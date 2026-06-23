"""Latency × accuracy frontier sweep on the real AMI meeting fixture.

The accuracy baseline reports latency and WER as two *separate* numbers. But
"optimize the speed/accuracy tradeoff" is about the **frontier** — the curve of
(caption latency vs WER) you get as you sweep the segmentation knobs. You cannot
optimize a tradeoff you cannot see, so this script measures the curve.

For each grid point it replays the SAME clip twice under identical config:
  - realtime  -> end-to-end caption latency p50/p95 (reuses _run_latency)
  - offline   -> WER/CER against ground truth     (reuses _run_accuracy)

Knobs swept (both are SessionConfig fields; balanced-preset defaults shown):
  - pre_vad_max_segment_seconds (default 8.0) — the dominant lever: during
    continuous speech pre-VAD holds a segment until this cap before the formal
    lane transcribes it, so it sets the floor on formal-lane latency.
  - asr_accumulation_seconds    (default 2.0)

Why AMI and not the synthetic fixtures: AMI is real conversational English with a
natural pause distribution. Tuning endpointing on adversarial synthetic clips
(rapped lyrics, tongue-twisters) tunes to the wrong distribution — exactly why
issue #13's endpointing pass was reverted (see docs/optimization-2026-06.md).

Run with the MeetingBro conda env:
  python scripts/benchmark_latency_accuracy_frontier.py
  python scripts/benchmark_latency_accuracy_frontier.py --pre-vad-max 8,6,4,3,2
  python scripts/benchmark_latency_accuracy_frontier.py --accumulation 2.0,1.5 --skip-latency
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import asyncio
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_accuracy as ba  # noqa: E402  (reuse the production-faithful runners)
from meetingbro.asr.faster_whisper_adapter import FasterWhisperAdapter  # noqa: E402

DEFAULT_MANIFEST = ROOT / "data" / "benchmark" / "real" / "manifest.json"
DEFAULT_FIXTURE_ID = "ami_en2002a"
DEFAULT_OUT = ROOT / "docs" / "benchmarks" / "frontier-latency-accuracy.md"
DEFAULT_PRE_VAD_MAX = "8,6,4,3,2"
DEFAULT_ACCUMULATION = "2.0"


@dataclass
class _Point:
    pre_vad_max_segment_seconds: float
    asr_accumulation_seconds: float
    wer: float | None
    metric: str
    accuracy_segments: int
    accuracy_rtf: float | None
    # Latency p50/p95 computed on samples POOLED across all runs (robust).
    latency_p50: float | None
    latency_p95: float | None
    latency_segments: int
    # Spread of the per-run p50 (min/max across runs) — exposes residual noise.
    latency_p50_min: float | None = None
    latency_p50_max: float | None = None
    runs: int = 1
    is_baseline: bool = False
    pareto: bool = False


def _floats(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _mark_pareto(points: list[_Point]) -> None:
    """A point is Pareto-optimal if no other measured point has both lower (or
    equal) latency p50 AND lower (or equal) WER, with at least one strictly
    better. Only points with both axes measured participate."""
    scored = [p for p in points if p.latency_p50 is not None and p.wer is not None]
    for p in scored:
        dominated = any(
            q is not p
            and q.latency_p50 <= p.latency_p50
            and q.wer <= p.wer
            and (q.latency_p50 < p.latency_p50 or q.wer < p.wer)
            for q in scored
        )
        p.pareto = not dominated


async def _measure(
    adapter: FasterWhisperAdapter,
    fixture: dict,
    wav: Path,
    *,
    pre_vad_max: float,
    accumulation: float,
    skip_latency: bool,
    skip_accuracy: bool,
    isolate_governor: bool,
    runs: int,
    is_baseline: bool,
) -> _Point:
    overrides = {
        "pre_vad_max_segment_seconds": pre_vad_max,
        "asr_accumulation_seconds": accumulation,
    }
    if isolate_governor:
        # Disable the ASR safeguard so the knob's effect is isolated. The
        # safeguard fires non-deterministically on short segments and changes
        # weak-rescue/retry behaviour, which confounded the first sweep
        # (asr_rtf spiked to 9.27). The resource governor only gates background
        # work (off in this benchmark), so this is the relevant pin.
        overrides["asr_safeguard_enabled"] = False

    wer: float | None = None
    metric = "wer"
    acc_segments = 0
    acc_rtf: float | None = None
    if not skip_accuracy:
        # WER is deterministic for a fixed config, so score it once.
        acc = await ba._run_accuracy(adapter, fixture, apply_vocab=True, extra_overrides=overrides)
        wer = acc.error_rate
        metric = acc.metric
        acc_segments = acc.segments
        acc_rtf = acc.asr_rtf

    p50 = p95 = p50_min = p50_max = None
    lat_segments = 0
    if not skip_latency:
        # Realtime latency has run-to-run variance, so repeat and pool the raw
        # per-segment samples across runs before taking percentiles.
        pooled: list[float] = []
        per_run_p50: list[float] = []
        for _ in range(max(1, runs)):
            lat = await ba._run_latency(adapter, wav, extra_overrides=overrides)
            pooled.extend(lat.samples)
            if lat.p50 == lat.p50:  # not NaN
                per_run_p50.append(lat.p50)
            lat_segments = lat.segments
        if pooled:
            p50 = float(statistics.median(pooled))
            p95 = float(np.percentile(pooled, 95))
        if per_run_p50:
            p50_min, p50_max = min(per_run_p50), max(per_run_p50)

    return _Point(
        pre_vad_max_segment_seconds=pre_vad_max,
        asr_accumulation_seconds=accumulation,
        wer=wer,
        metric=metric,
        accuracy_segments=acc_segments,
        accuracy_rtf=acc_rtf,
        latency_p50=p50,
        latency_p95=p95,
        latency_segments=lat_segments,
        latency_p50_min=p50_min,
        latency_p50_max=p50_max,
        runs=max(1, runs),
        is_baseline=is_baseline,
    )


def _build_report(args, fixture: dict, points: list[_Point]) -> str:
    L: list[str] = []
    L.append("# MeetingBro 延迟 × 准确率前沿 (latency–accuracy frontier)")
    L.append("")
    L.append("> 由 `scripts/benchmark_latency_accuracy_frontier.py` 生成。在**真实 AMI 会议片段**上")
    L.append("> 扫描分段旋钮,对每个配置在**相同音频**上分别测字幕延迟(实时回放)与 WER(离线评分),")
    L.append("> 给出可优化的 Pareto 前沿,而非两个孤立数字。`★` 标记 Pareto 最优点。")
    L.append("")
    L.append("## 配置")
    L.append("")
    L.append(f"- fixture: `{fixture.get('id')}` ({fixture.get('language')}, "
             f"{fixture.get('source', 'n/a')})")
    L.append(f"- Whisper: `{args.model_size}` / `{args.device}` / `{args.compute_type}` / beam `{args.beam_size}`")
    L.append("- runtime profile: `balanced` (formal lane only, fast_preview disabled), forced_language=`en`")
    isolation = "ASR safeguard DISABLED (knob isolated)" if not args.keep_safeguard else "production-faithful (safeguard on)"
    L.append(f"- governor: {isolation};  runs/point: {args.runs} (latency samples pooled across runs)")
    L.append(f"- swept: pre_vad_max_segment_seconds={args.pre_vad_max}  asr_accumulation_seconds={args.accumulation}")
    if args.label:
        L.append(f"- label: {args.label}")
    L.append("")
    L.append("## 前沿点")
    L.append("")
    L.append("| | pre_vad_max (s) | accum (s) | latency p50 (s) | p50 range | p95 (s) | WER | seg(acc/lat) | asr_rtf |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in points:
        flag = "★" if p.pareto else ("•" if p.is_baseline else "")
        base = " (baseline)" if p.is_baseline else ""
        spread = (
            f"{ba._fmt(p.latency_p50_min, 2)}–{ba._fmt(p.latency_p50_max, 2)}"
            if p.latency_p50_min is not None else "n/a"
        )
        L.append(
            f"| {flag} | {p.pre_vad_max_segment_seconds:g}{base} | {p.asr_accumulation_seconds:g} | "
            f"{ba._fmt(p.latency_p50, 2)} | {spread} | {ba._fmt(p.latency_p95, 2)} | {ba._fmt(p.wer)} | "
            f"{p.accuracy_segments}/{p.latency_segments} | {ba._fmt(p.accuracy_rtf, 2)} |"
        )
    L.append("")
    L.append("## 解读")
    L.append("")
    L.append("- `★` = Pareto 最优(没有任何其它点同时在延迟和 WER 上都不差且至少一项更好)。")
    L.append("- 沿 `pre_vad_max_segment_seconds` 下降,延迟应下降;若 WER 不升或仅微升,即为**免费的延迟收益**,")
    L.append("  说明当前 8.0s 上限对真实会议语音偏保守。若 WER 明显恶化,则该点是 tradeoff 的拐点。")
    L.append("- 这是真实会议语音上的相对比较;绝对 WER 受 AMI 远场单麦(SDM)与重叠说话影响,看趋势而非绝对值。")
    L.append("")
    return "\n".join(L)


def _to_json(args, fixture: dict, points: list[_Point]) -> dict:
    return {
        "config": {
            "fixture_id": fixture.get("id"),
            "language": fixture.get("language"),
            "model_size": args.model_size,
            "device": args.device,
            "compute_type": args.compute_type,
            "beam_size": args.beam_size,
            "runtime_profile": "balanced",
            "pre_vad_max": args.pre_vad_max,
            "accumulation": args.accumulation,
            "runs": args.runs,
            "safeguard_disabled": not args.keep_safeguard,
            "label": args.label,
        },
        "points": [
            {
                "pre_vad_max_segment_seconds": p.pre_vad_max_segment_seconds,
                "asr_accumulation_seconds": p.asr_accumulation_seconds,
                "wer": p.wer,
                "metric": p.metric,
                "accuracy_segments": p.accuracy_segments,
                "accuracy_rtf": p.accuracy_rtf,
                "latency_p50_s": p.latency_p50,
                "latency_p95_s": p.latency_p95,
                "latency_p50_min_s": p.latency_p50_min,
                "latency_p50_max_s": p.latency_p50_max,
                "latency_segments": p.latency_segments,
                "runs": p.runs,
                "is_baseline": p.is_baseline,
                "pareto": p.pareto,
            }
            for p in points
        ],
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latency × accuracy frontier sweep on the AMI fixture.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--fixture-id", default=DEFAULT_FIXTURE_ID)
    p.add_argument("--pre-vad-max", default=DEFAULT_PRE_VAD_MAX,
                   help="Comma list of pre_vad_max_segment_seconds to sweep (default: 8,6,4,3,2).")
    p.add_argument("--accumulation", default=DEFAULT_ACCUMULATION,
                   help="Comma list of asr_accumulation_seconds to sweep (default: 2.0).")
    p.add_argument("--model-size", default="small")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--runs", type=int, default=1,
                   help="Realtime latency passes per point; samples pooled across runs (default: 1).")
    p.add_argument("--keep-safeguard", action="store_true",
                   help="Keep the ASR safeguard on (production-faithful, but confounds the knob sweep). "
                        "Default isolates the knob by disabling it.")
    p.add_argument("--skip-latency", action="store_true", help="Measure WER only (fast; no realtime passes).")
    p.add_argument("--skip-accuracy", action="store_true", help="Measure latency only.")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--json-out", default=None)
    p.add_argument("--label", default=None)
    return p.parse_args(argv[1:])


async def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        print("  (regenerate the AMI fixture with: python scripts/fetch_ami_fixture.py)", file=sys.stderr)
        return 2

    fixtures = json.loads(manifest_path.read_text(encoding="utf-8")).get("fixtures", [])
    fixture = next((f for f in fixtures if f.get("id") == args.fixture_id), None)
    if fixture is None:
        print(f"fixture id {args.fixture_id!r} not in {manifest_path}", file=sys.stderr)
        return 2
    wav = ROOT / fixture["path"] if not Path(fixture["path"]).is_absolute() else Path(fixture["path"])
    if not wav.exists():
        print(f"fixture audio missing: {wav}", file=sys.stderr)
        print("  (audio is gitignored; regenerate with: python scripts/fetch_ami_fixture.py)", file=sys.stderr)
        return 2

    pre_vad_values = _floats(args.pre_vad_max)
    accum_values = _floats(args.accumulation)
    grid = [(pvm, acc) for pvm in pre_vad_values for acc in accum_values]

    adapter = FasterWhisperAdapter(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
    )
    print("Warming up Whisper model…")
    adapter.transcribe(np.zeros(ba.TARGET_SR, dtype=np.float32), ba.TARGET_SR, quality_preset="realtime")

    print(f"Sweeping {len(grid)} grid point(s) on {fixture['id']} "
          f"({'latency+accuracy' if not (args.skip_latency or args.skip_accuracy) else 'partial'})…")
    points: list[_Point] = []
    for pvm, acc in grid:
        is_baseline = pvm == 8.0 and acc == 2.0
        print(f"  pre_vad_max={pvm:g} accum={acc:g} …", flush=True)
        pt = await _measure(
            adapter, fixture, wav,
            pre_vad_max=pvm, accumulation=acc,
            skip_latency=args.skip_latency, skip_accuracy=args.skip_accuracy,
            isolate_governor=not args.keep_safeguard, runs=args.runs,
            is_baseline=is_baseline,
        )
        points.append(pt)
        print(f"      WER={ba._fmt(pt.wer)} p50={ba._fmt(pt.latency_p50, 2)}s p95={ba._fmt(pt.latency_p95, 2)}s")

    _mark_pareto(points)

    report = _build_report(args, fixture, points)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nMarkdown report → {out}")

    json_out = Path(args.json_out) if args.json_out else out.with_suffix(".json")
    if not json_out.is_absolute():
        json_out = ROOT / json_out
    json_out.write_text(json.dumps(_to_json(args, fixture, points), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON report     → {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
