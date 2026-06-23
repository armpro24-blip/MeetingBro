"""Accuracy + latency baseline for MeetingBro's production ASR pipeline.

Two passes, both LLM-free (Noop summarizer/translator) and formal-lane only
(fast preview disabled) so the numbers are deterministic and reflect what the
user ultimately reads:

  ACCURACY  — each manifest fixture is replayed through the full pipeline
              (WavFileSource -> SessionManager -> FasterWhisperAdapter) and the
              concatenated transcript is scored against ground truth with
              WER (Latin: en/de) or CER (CJK: zh) via scripts/asr_metrics.py.
  LATENCY   — one wav is replayed in real time and the end-to-end caption
              latency (audio-end -> segment emitted) is measured at p50/p95/max,
              mirroring scripts/benchmark_accumulation_latency.py.

The result is written as JSON and as a Markdown report. This is the baseline
that issues #2-#9 must compare their before/after numbers against.

Examples:
  python scripts/benchmark_accuracy.py
  python scripts/benchmark_accuracy.py --model-size medium --out docs/benchmarks/baseline-medium.md
  python scripts/benchmark_accuracy.py --skip-latency --json-out /tmp/acc.json
"""
from __future__ import annotations

import sys

# Windows cp1252 stdout chokes on CJK — force UTF-8 before any print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import asr_metrics  # noqa: E402  (local module, scripts/ on path)
from meetingbro.asr.faster_whisper_adapter import FasterWhisperAdapter  # noqa: E402
from meetingbro.audio.capture import AudioSource, WavFileSource  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.session.profiles import RUNTIME_PROFILE_PRESETS  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402

DATA_DIR = ROOT / "data"
DEFAULT_MANIFEST = DATA_DIR / "benchmark" / "manifest.json"
DEFAULT_LATENCY_WAV = DATA_DIR / "sample_en.wav"
DEFAULT_OUT = ROOT / "docs" / "benchmarks" / "baseline-2026-06.md"
LATENCY_MIN_SECONDS = 60.0
TARGET_SR = 16_000
# Measure the production default profile faithfully (its segmentation/latency
# params differ from the bare SessionConfig dataclass defaults).
RUNTIME_PROFILE = "balanced"


def _profile_settings(profile_name: str = RUNTIME_PROFILE) -> tuple[dict, float]:
    """Return (SessionConfig overrides, source chunk_seconds) for a runtime profile.

    Mirrors how main.py applies a profile preset to SessionConfig, so the
    benchmark exercises the real production behavior rather than the dataclass
    defaults. ``chunk_seconds`` is split out (it maps to the audio source, not a
    SessionConfig field) and ``fast_preview_enabled`` is dropped (forced off).
    """
    preset = dict(RUNTIME_PROFILE_PRESETS[profile_name])
    chunk = float(preset.pop("chunk_seconds", 0.75))
    preset.pop("fast_preview_enabled", None)
    return preset, chunk


PROFILE_CHUNK_SECONDS = _profile_settings()[1]

# zh/en/de are the project's first-class forced-language targets; anything else
# (e.g. "mixed") is replayed in auto-detect mode.
_FORCED_LANGS = {"zh", "en", "de"}


class _NoopSummarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None):
        return ""


class _NoopTranslator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


class _ArrivalProbeSource(AudioSource):
    """Wraps a source to record wall-clock arrival time of each chunk."""

    def __init__(self, inner: AudioSource) -> None:
        self._inner = inner
        self.first_chunk_wall: float | None = None

    @property
    def sample_rate(self) -> int:
        return self._inner.sample_rate

    async def stream(self):
        async for chunk in self._inner.stream():
            if self.first_chunk_wall is None:
                self.first_chunk_wall = time.monotonic()
            yield chunk

    def drain_drops(self) -> int:
        return self._inner.drain_drops()

    async def aclose(self) -> None:
        await self._inner.aclose()


@dataclass
class _AccuracyResult:
    fixture_id: str
    language: str
    metric: str
    error_rate: float | None
    asr_rtf: float | None
    duration_s: float
    segments: int
    transcript: str
    ground_truth: str
    missing: bool = False
    keywords_total: int = 0
    keywords_hit: int = 0
    vocab_applied: bool = False


@dataclass
class _LatencyResult:
    wav: str
    p50: float
    p95: float
    max: float
    segments: int
    asr_rtf: float | None
    # Preview (Qwen) lane, populated only when --preview is used.
    preview_p50: float | None = None
    preview_p95: float | None = None
    preview_segments: int = 0
    preview_rtf: float | None = None


def _wav_duration_seconds(path: Path) -> float:
    with sf.SoundFile(str(path), mode="r") as f:
        return len(f) / float(f.samplerate)


def _baseline_config(**overrides) -> dict:
    """Common SessionConfig kwargs for a deterministic, profile-faithful baseline.

    Applies the production runtime profile preset (so segmentation/latency params
    match what users actually run), then forces the bits that make a benchmark
    deterministic: no summaries, no live preview lane.
    """
    preset, _ = _profile_settings()
    cfg = dict(preset)  # production profile values (pre_vad_*, accumulation, etc.)
    cfg.update(
        summarizer=_NoopSummarizer(),
        translator=_NoopTranslator(),
        summary_language="en",
        runtime_profile=RUNTIME_PROFILE,
        audio_chunk_seconds=PROFILE_CHUNK_SECONDS,
        # Disable the preview lane so the scored transcript is pure formal Whisper.
        fast_preview_enabled=False,
        preview_asr=None,
        # Disable every summary worker (defensive; Noop already returns "").
        rolling_interval_seconds=10_000.0,
        memory_interval_seconds=10_000.0,
        cumulative_interval_seconds=10_000.0,
        refinement_interval_seconds=10_000.0,
        min_segments_for_rolling=10_000,
        min_segments_for_memory=10_000,
        min_segments_for_cumulative=10_000,
        min_segments_for_refinement=10_000,
    )
    cfg.update(overrides)
    return cfg


DEFAULT_QWEN_DIR = ROOT / "models" / "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"


def _build_qwen_preview(model_dir: Path = DEFAULT_QWEN_DIR):
    """Build the Qwen3 preview adapter for the --preview latency measurement."""
    from meetingbro.asr.qwen3_asr_adapter import Qwen3ASRAdapter

    if not model_dir.exists():
        raise FileNotFoundError(f"Qwen3 preview model not found: {model_dir}")
    return Qwen3ASRAdapter(model_dir=model_dir, num_threads=2, provider="cpu")


def _vocabulary_hint(fixture: dict) -> str | None:
    """Vocabulary hint for a fixture as the UI would send it: a raw, comma-
    separated term list. The backend (SessionManager._format_vocabulary_prompt)
    is responsible for turning it into a glossary-style Whisper prompt, so this
    exercises the real production formatting path."""
    hint = fixture.get("vocabulary_hint")
    if hint:
        return hint
    keywords = fixture.get("keywords")
    if keywords:
        return ", ".join(keywords)
    return None


def _keyword_recall(transcript: str, keywords: list[str]) -> tuple[int, int]:
    folded = transcript.casefold()
    hit = sum(1 for k in keywords if k.casefold() in folded)
    return len(keywords), hit


async def _run_accuracy(
    adapter: FasterWhisperAdapter,
    fixture: dict,
    *,
    apply_vocab: bool = True,
    extra_overrides: dict | None = None,
) -> _AccuracyResult:
    fid = fixture["id"]
    language = fixture.get("language", "")
    ground_truth = fixture.get("ground_truth", "")
    keywords = fixture.get("keywords") or []
    path = ROOT / fixture["path"] if not Path(fixture["path"]).is_absolute() else Path(fixture["path"])

    # `metric` may be overridden in the manifest (e.g. a zh-dominant code-switch
    # clip wants cer); otherwise pick by language/script.
    metric = fixture.get("metric") or asr_metrics.metric_for(language, ground_truth)
    if not path.exists():
        return _AccuracyResult(fid, language, metric, None, None, 0.0, 0, "", ground_truth, missing=True,
                               keywords_total=len(keywords))

    # `auto: true` forces auto-detect regardless of the (metric-only) language label.
    forced_language = None if fixture.get("auto") else (language if language in _FORCED_LANGS else None)
    vocab = _vocabulary_hint(fixture) if apply_vocab else None
    duration = _wav_duration_seconds(path)
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "bench.db")
        try:
            source = WavFileSource(path, sample_rate=TARGET_SR, chunk_seconds=PROFILE_CHUNK_SECONDS, realtime=False)
            manager = SessionManager(SessionConfig(
                audio_source=source,
                asr=adapter,
                storage=storage,
                forced_language=forced_language,
                vocabulary_hint=vocab,
                # Offline accuracy replay floods the queue (chunks arrive as fast
                # as the file reads). The production 8 s bound would drop the
                # oldest audio and silently truncate the start of longer clips,
                # so we make the queue effectively unbounded for scoring.
                audio_input_queue_max_seconds=100_000.0,
                **_baseline_config(**(extra_overrides or {})),
            ))
            await manager.start()
            if manager._task is not None:
                await manager._task
            await manager.stop()

            segments = storage.list_segments(manager.meeting_id)
            transcript = " ".join(seg.text for seg in segments).strip()
            asr_rtf = manager._state.asr_realtime_factor
        finally:
            storage.close()

    error_rate = asr_metrics.cer(ground_truth, transcript) if metric == "cer" else asr_metrics.wer(ground_truth, transcript)
    kw_total, kw_hit = _keyword_recall(transcript, keywords)
    return _AccuracyResult(
        fixture_id=fid,
        language=language,
        metric=metric,
        error_rate=error_rate,
        asr_rtf=asr_rtf,
        duration_s=duration,
        segments=len(segments),
        transcript=transcript,
        ground_truth=ground_truth,
        keywords_total=kw_total,
        keywords_hit=kw_hit,
        vocab_applied=vocab is not None,
    )


def _prepare_latency_wav(path: Path) -> tuple[Path, bool]:
    """Ensure the latency wav is >= LATENCY_MIN_SECONDS by tiling. Returns (path, is_temp)."""
    info = sf.info(str(path))
    if info.duration >= LATENCY_MIN_SECONDS:
        return path, False
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    target_frames = int(LATENCY_MIN_SECONDS * sample_rate)
    repeats = max(1, int(np.ceil(target_frames / len(samples))))
    extended = np.tile(samples, repeats)[:target_frames]
    tmp = tempfile.NamedTemporaryFile(prefix="mb-latency-", suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, extended, sample_rate)
    return Path(tmp.name), True


async def _run_latency(
    adapter: FasterWhisperAdapter,
    wav: Path,
    *,
    preview_asr=None,
    extra_overrides: dict | None = None,
) -> _LatencyResult:
    prepared, is_temp = _prepare_latency_wav(wav)
    cfg_overrides = dict(forced_language="en")
    if preview_asr is not None:
        # Enable the preview (Qwen) lane on a dedicated executor — this is what
        # delivers sub-second captions. Overrides the formal-only baseline.
        cfg_overrides.update(
            fast_preview_enabled=True,
            preview_asr=preview_asr,
            preview_asr_backend_name="qwen3",
        )
    if extra_overrides:
        cfg_overrides.update(extra_overrides)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp) / "bench.db")
            source = _ArrivalProbeSource(
                WavFileSource(prepared, sample_rate=TARGET_SR, chunk_seconds=PROFILE_CHUNK_SECONDS, realtime=True)
            )
            manager = SessionManager(SessionConfig(
                audio_source=source,
                asr=adapter,
                storage=storage,
                **_baseline_config(**cfg_overrides),
            ))
            latencies: list[float] = []
            preview_latencies: list[float] = []
            bench_start = time.monotonic()

            async def collect() -> None:
                async for event in manager.events():
                    base = source.first_chunk_wall or bench_start
                    if event.type == "transcript_segment":
                        latencies.append(time.monotonic() - base - event.payload["end_time"])
                    elif event.type == "transcript_preview":
                        seg = event.payload.get("segment")
                        if seg:
                            preview_latencies.append(time.monotonic() - base - seg["end_time"])

            collector = asyncio.create_task(collect())
            try:
                await manager.start()
                if manager._task is not None:
                    await manager._task
                await manager.stop()
            finally:
                collector.cancel()
                try:
                    await collector
                except asyncio.CancelledError:
                    pass
                asr_rtf = manager._state.asr_realtime_factor
                preview_rtf = manager._state.fast_preview_realtime_factor
                storage.close()

            def _stats(xs):
                if not xs:
                    return float("nan"), float("nan")
                return statistics.median(xs), float(np.percentile(xs, 95))

            p50, p95 = _stats(latencies)
            mx = max(latencies) if latencies else float("nan")
            pv_p50, pv_p95 = _stats(preview_latencies)
            return _LatencyResult(
                str(wav.name), p50, p95, mx, len(latencies), asr_rtf,
                preview_p50=(None if not preview_latencies else pv_p50),
                preview_p95=(None if not preview_latencies else pv_p95),
                preview_segments=len(preview_latencies),
                preview_rtf=preview_rtf,
            )
    finally:
        if is_temp and prepared.exists():
            prepared.unlink(missing_ok=True)


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or value != value:  # None or NaN
        return "n/a"
    return f"{value:.{digits}f}"


def _aggregate(results: list[_AccuracyResult]) -> dict[str, dict]:
    groups: dict[str, list[float]] = {}
    for r in results:
        if r.missing or r.error_rate is None:
            continue
        groups.setdefault(r.language, []).append(r.error_rate)
    return {
        lang: {"metric": "cer" if lang == "zh" else "wer", "mean": sum(v) / len(v), "n": len(v)}
        for lang, v in sorted(groups.items())
    }


def _build_report(args, accuracy: list[_AccuracyResult], latency: _LatencyResult | None) -> str:
    agg = _aggregate(accuracy)
    lines: list[str] = []
    lines.append("# MeetingBro ASR 基线报告 (baseline)")
    lines.append("")
    lines.append("> 由 `scripts/benchmark_accuracy.py` 生成。仅正式通道(formal Whisper),已禁用预览(preview)与摘要,")
    lines.append("> 确保结果可复现。后续优化 issue (#2-#9) 须引用此基线给出前/后对比。")
    lines.append("")
    lines.append("## 配置")
    lines.append("")
    lines.append(f"- Whisper model size: `{args.model_size}`")
    lines.append(f"- device: `{args.device}`  compute type: `{args.compute_type}`  beam size: `{args.beam_size}`")
    lines.append(f"- runtime profile: `balanced`  (formal lane only, fast_preview disabled)")
    lines.append(f"- manifest: `{Path(args.manifest).relative_to(ROOT) if Path(args.manifest).is_relative_to(ROOT) else args.manifest}`")
    if args.label:
        lines.append(f"- label: {args.label}")
    lines.append("")

    lines.append("## 准确率 (WER / CER)")
    lines.append("")
    lines.append("| fixture | lang | metric | error_rate | kw recall | vocab | asr_rtf | dur (s) | seg |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in accuracy:
        kw = f"{r.keywords_hit}/{r.keywords_total}" if r.keywords_total else "—"
        vocab = "on" if r.vocab_applied else "off"
        if r.missing:
            lines.append(f"| {r.fixture_id} | {r.language} | {r.metric} | _audio missing_ | {kw} | {vocab} | — | — | — |")
        else:
            lines.append(
                f"| {r.fixture_id} | {r.language} | {r.metric} | {_fmt(r.error_rate)} | {kw} | {vocab} | "
                f"{_fmt(r.asr_rtf, 2)} | {r.duration_s:.1f} | {r.segments} |"
            )
    lines.append("")
    lines.append("### 按语言聚合")
    lines.append("")
    if agg:
        lines.append("| lang | metric | mean | n |")
        lines.append("| --- | --- | --- | --- |")
        for lang, info in agg.items():
            lines.append(f"| {lang} | {info['metric']} | {_fmt(info['mean'])} | {info['n']} |")
    else:
        lines.append("_无可评分的 fixture(音频缺失?运行 Qwen3 预览模型下载以获取 test_wavs)。_")
    missing = [r.fixture_id for r in accuracy if r.missing]
    if missing:
        lines.append("")
        lines.append(f"> ⚠ 缺失音频(已跳过): {', '.join(missing)}")
    lines.append("")

    lines.append("## 端到端字幕延迟 (formal lane)")
    lines.append("")
    if latency is None:
        lines.append("_已跳过 (--skip-latency)。_")
    else:
        lines.append(f"源文件: `{latency.wav}` (循环铺到 ≥{LATENCY_MIN_SECONDS:.0f}s),实时回放。延迟 = 音频结束 → 字幕发出。")
        lines.append("")
        lines.append("| lane | p50 (s) | p95 (s) | max (s) | segments | rtf |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        lines.append(
            f"| formal | {_fmt(latency.p50, 2)} | {_fmt(latency.p95, 2)} | {_fmt(latency.max, 2)} | "
            f"{latency.segments} | {_fmt(latency.asr_rtf, 2)} |"
        )
        if latency.preview_segments:
            lines.append(
                f"| preview (Qwen) | {_fmt(latency.preview_p50, 2)} | {_fmt(latency.preview_p95, 2)} | — | "
                f"{latency.preview_segments} | {_fmt(latency.preview_rtf, 2)} |"
            )
    lines.append("")
    lines.append("## 结果解读 (重要)")
    lines.append("")
    lines.append("- **这些 fixture 是对抗性压力样本**(背景噪声 / 配乐说唱 / 歌唱 / 绕口令 / 多语 code-switch),")
    lines.append("  **不代表干净会议语音**。绝对错误率偏高是预期的;唯一的干净样本 `de_de` (WER 0.14) 才是")
    lines.append("  干净语音的现实参考。本基线的价值在于**相对比较**(优化前/后),而非绝对达标线。")
    lines.append("- 离线评分关闭了音频输入队列上限(`audio_input_queue_max_seconds`),否则非实时回放会冲垮 8s")
    lines.append("  队列、丢掉长片段的开头,污染准确率。生产实时路径不受影响。")
    lines.append("- 正式通道字幕延迟为多秒级:连续无停顿语音会让 pre-VAD 把片段攒到上限才发出。**亚秒级**")
    lines.append("  实时字幕由预览(preview)通道提供,需单独度量(见 #6/#13)。`sample_en.wav` 为循环铺长。")
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- WER 用于 Latin 语言(en/de + code-switch),CER 用于 CJK(zh);见 `scripts/asr_metrics.py`。")
    lines.append("- 此基线为**正式通道**质量;预览(preview)延迟与质量另行度量(见 #6/#13)。")
    lines.append("- ground truth 提交于 `data/benchmark/manifest.json`;音频随 Qwen3 模型下载,gitignored。")
    lines.append("- 复现: `python scripts/benchmark_accuracy.py`(MeetingBro conda 环境)。")
    lines.append("")
    return "\n".join(lines)


def _to_json(args, accuracy: list[_AccuracyResult], latency: _LatencyResult | None) -> dict:
    return {
        "config": {
            "model_size": args.model_size,
            "device": args.device,
            "compute_type": args.compute_type,
            "beam_size": args.beam_size,
            "runtime_profile": "balanced",
            "formal_lane_only": True,
            "label": args.label,
        },
        "accuracy": [
            {
                "id": r.fixture_id,
                "language": r.language,
                "metric": r.metric,
                "error_rate": r.error_rate,
                "asr_rtf": r.asr_rtf,
                "duration_s": round(r.duration_s, 2),
                "segments": r.segments,
                "missing": r.missing,
                "keywords_total": r.keywords_total,
                "keywords_hit": r.keywords_hit,
                "vocab_applied": r.vocab_applied,
                "transcript": r.transcript,
                "ground_truth": r.ground_truth,
            }
            for r in accuracy
        ],
        "accuracy_by_language": _aggregate(accuracy),
        "latency": (
            None if latency is None else {
                "wav": latency.wav,
                "p50_s": latency.p50,
                "p95_s": latency.p95,
                "max_s": latency.max,
                "segments": latency.segments,
                "asr_rtf": latency.asr_rtf,
                "preview_p50_s": latency.preview_p50,
                "preview_p95_s": latency.preview_p95,
                "preview_segments": latency.preview_segments,
                "preview_rtf": latency.preview_rtf,
            }
        ),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MeetingBro ASR accuracy + latency baseline.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--model-size", default="small", help="Faster-Whisper model size (default: small, the CPU default).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--latency-wav", default=str(DEFAULT_LATENCY_WAV))
    p.add_argument("--skip-latency", action="store_true", help="Skip the latency pass (accuracy only).")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Markdown report path.")
    p.add_argument("--json-out", default=None, help="Optional JSON report path (defaults beside --out).")
    p.add_argument("--label", default=None, help="Optional run label recorded in the report.")
    p.add_argument("--ignore-vocab", action="store_true",
                   help="Disable vocabulary_hint/keyword biasing (A/B baseline for #5).")
    p.add_argument("--preview", action="store_true",
                   help="Also run the latency pass with the Qwen preview lane enabled and report its latency.")
    return p.parse_args(argv[1:])


async def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    fixtures = json.loads(manifest_path.read_text(encoding="utf-8")).get("fixtures", [])
    if not fixtures:
        print("manifest has no fixtures", file=sys.stderr)
        return 2

    adapter = FasterWhisperAdapter(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
    )
    # Warm up so model-load time does not contaminate the first fixture's RTF.
    print("Warming up Whisper model…")
    warm = np.zeros(TARGET_SR, dtype=np.float32)
    adapter.transcribe(warm, TARGET_SR, quality_preset="realtime")

    print(f"Accuracy pass: {len(fixtures)} fixtures…")
    accuracy: list[_AccuracyResult] = []
    for fx in fixtures:
        r = await _run_accuracy(adapter, fx, apply_vocab=not args.ignore_vocab)
        accuracy.append(r)
        if r.missing:
            print(f"  [skip] {r.fixture_id}: audio missing")
        else:
            kw = f" kw={r.keywords_hit}/{r.keywords_total}" if r.keywords_total else ""
            vocab = " vocab=on" if r.vocab_applied else ""
            print(f"  {r.fixture_id} [{r.language}] {r.metric}={_fmt(r.error_rate)} asr_rtf={_fmt(r.asr_rtf, 2)}{kw}{vocab}")

    latency: _LatencyResult | None = None
    if not args.skip_latency:
        lat_wav = Path(args.latency_wav)
        if not lat_wav.is_absolute():
            lat_wav = ROOT / lat_wav
        if lat_wav.exists():
            preview_asr = None
            if args.preview:
                print("Building Qwen3 preview adapter…")
                preview_asr = _build_qwen_preview()
            print(f"Latency pass: {lat_wav.name} (realtime){' +preview' if args.preview else ''}…")
            latency = await _run_latency(adapter, lat_wav, preview_asr=preview_asr)
            print(f"  formal  p50={_fmt(latency.p50, 2)}s p95={_fmt(latency.p95, 2)}s")
            if latency.preview_segments:
                print(f"  preview p50={_fmt(latency.preview_p50, 2)}s p95={_fmt(latency.preview_p95, 2)}s "
                      f"segs={latency.preview_segments} rtf={_fmt(latency.preview_rtf, 2)}")
        else:
            print(f"  [skip] latency wav not found: {lat_wav}")

    report = _build_report(args, accuracy, latency)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nMarkdown report → {out}")

    json_out = Path(args.json_out) if args.json_out else out.with_suffix(".json")
    if not json_out.is_absolute():
        json_out = ROOT / json_out
    json_out.write_text(json.dumps(_to_json(args, accuracy, latency), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON report     → {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
