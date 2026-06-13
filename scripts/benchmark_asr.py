"""Benchmark MeetingBro's local ASR path on WAV files.

This is intentionally LLM-free: it runs WavFileSource -> SessionManager ->
FasterWhisperAdapter with no-op summarization/translation so the metrics mostly
reflect audio preprocessing, VAD, ASR speed, weak-speech rescue, and safeguards.

Examples:
  python scripts/benchmark_asr.py data/sample_en.wav --language en
  python scripts/benchmark_asr.py data/*.wav --language auto --model-size small
  python scripts/benchmark_asr.py data/sample_en.wav --keywords Alice Bob budget
  python scripts/benchmark_asr.py data/sample_en.wav --json-out benchmark.json
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import asr_metrics  # noqa: E402  (local module, scripts/ on path)
from meetingbro.asr.faster_whisper_adapter import FasterWhisperAdapter  # noqa: E402
from meetingbro.audio.capture import WavFileSource  # noqa: E402
from meetingbro.llm.openai_compatible import _load_dotenv_if_present  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.session.profiles import RUNTIME_PROFILE_PRESETS  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402


PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    **{name: dict(values) for name, values in RUNTIME_PROFILE_PRESETS.items()},
    "legacy": {
        "chunk_seconds": 5.0,
        "asr_accumulation_seconds": 2.5,
        "pre_vad_enabled": False,
        "asr_early_flush_enabled": False,
        "silence_commit_min_confidence": 1.1,
        "silence_commit_min_duration_seconds": 10.0,
        "pre_vad_adaptive_trailing_silence_enabled": False,
        "language_lock_enabled": False,
    },
    "tuned_base": {
        "chunk_seconds": 0.5,
        "asr_accumulation_seconds": 1.5,
        "pre_vad_enabled": False,
        "asr_early_flush_enabled": False,
        "silence_commit_min_confidence": 1.1,
        "silence_commit_min_duration_seconds": 10.0,
        "pre_vad_adaptive_trailing_silence_enabled": False,
        "language_lock_enabled": False,
    },
    "tuned_commit": {
        "chunk_seconds": 0.5,
        "asr_accumulation_seconds": 1.5,
        "pre_vad_enabled": False,
        "asr_early_flush_enabled": True,
        "silence_commit_min_confidence": 0.75,
        "silence_commit_min_duration_seconds": 0.6,
        "pre_vad_adaptive_trailing_silence_enabled": False,
        "language_lock_enabled": False,
    },
    "full_current": {
        **RUNTIME_PROFILE_PRESETS["balanced"],
    },
}


class NoopSummarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None):
        return ""


class NoopTranslator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cfg_value(overrides: dict[str, Any], key: str, env_name: str, default: Any) -> Any:
    if key in overrides:
        return overrides[key]
    return os.environ.get(env_name, default)


def _cfg_float(overrides: dict[str, Any], key: str, env_name: str, default: float) -> float:
    if key in overrides:
        return float(overrides[key])
    return _env_float(env_name, default)


def _cfg_int(overrides: dict[str, Any], key: str, env_name: str, default: int) -> int:
    if key in overrides:
        return int(overrides[key])
    return _env_int(env_name, default)


def _cfg_bool(overrides: dict[str, Any], key: str, env_name: str, default: bool) -> bool:
    if key in overrides:
        return bool(overrides[key])
    return _env_bool(env_name, default)


def _parse_profiles(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one profile name is required")
    invalid = [name for name in names if name not in PROFILE_PRESETS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown profile(s): {', '.join(invalid)}. valid: {', '.join(PROFILE_PRESETS)}"
        )
    return names


def _resolve_inputs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in patterns:
        has_glob = any(ch in item for ch in "*?[")
        if has_glob:
            pattern = item if Path(item).is_absolute() else str(ROOT / item)
            matches = [Path(p) for p in glob.glob(pattern)]
        else:
            matches = []
        if not matches:
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            matches = [path]
        for path in matches:
            path = path.resolve()
            if path.suffix.lower() == ".wav" and path.exists():
                out.append(path)
    return sorted(dict.fromkeys(out))


def _wav_duration_seconds(path: Path) -> float:
    import soundfile as sf

    with sf.SoundFile(str(path), mode="r") as f:
        return len(f) / float(f.samplerate)


def _keyword_recall(text: str, keywords: list[str]) -> dict[str, Any]:
    if not keywords:
        return {"expected": 0, "matched": 0, "recall": None, "missing": []}
    folded = text.casefold()
    missing = [kw for kw in keywords if kw.casefold() not in folded]
    matched = len(keywords) - len(missing)
    return {
        "expected": len(keywords),
        "matched": matched,
        "recall": matched / max(1, len(keywords)),
        "missing": missing,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MeetingBro ASR on WAV files.")
    parser.add_argument("wav", nargs="+", help="WAV file(s) or repo-relative glob(s).")
    parser.add_argument("--language", choices=("auto", "zh", "en", "de"), default="auto")
    parser.add_argument("--model-size", default=os.environ.get("MEETINGBRO_WHISPER_SIZE", "tiny"))
    parser.add_argument("--device", default=os.environ.get("MEETINGBRO_WHISPER_DEVICE", "cpu"))
    parser.add_argument("--compute-type", default=os.environ.get("MEETINGBRO_WHISPER_COMPUTE_TYPE", "int8"))
    parser.add_argument("--beam-size", type=int, default=_env_int("MEETINGBRO_WHISPER_BEAM_SIZE", 3))
    parser.add_argument("--chunk-seconds", type=float, default=_env_float("MEETINGBRO_FILE_CHUNK_SECONDS", 5.0))
    parser.add_argument(
        "--profiles",
        type=_parse_profiles,
        default=["full_current"],
        help="Comma-separated named config profiles to compare. Default: full_current (alias of balanced)",
    )
    parser.add_argument("--keywords", nargs="*", default=[], help="Expected keywords for rough recall.")
    parser.add_argument("--reference", default=None, help="Ground-truth transcript for WER/CER scoring.")
    parser.add_argument("--reference-file", default=None, help="Path to a file holding the ground-truth transcript.")
    parser.add_argument("--json-out", default=None, help="Optional JSON report path.")
    parser.add_argument("--text-out-dir", default=None, help="Optional directory for transcript .txt files.")
    return parser.parse_args(argv[1:])


async def _run_one(path: Path, args: argparse.Namespace, *, profile_name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    forced_language = None if args.language == "auto" else args.language
    audio_duration = _wav_duration_seconds(path)
    chunk_seconds = float(overrides.get("chunk_seconds", args.chunk_seconds))
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "benchmark.db")
        try:
            asr = FasterWhisperAdapter(
                model_size=args.model_size,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
                cpu_threads=_cfg_int(overrides, "whisper_cpu_threads", "MEETINGBRO_WHISPER_CPU_THREADS", 0),
                num_workers=_cfg_int(overrides, "whisper_num_workers", "MEETINGBRO_WHISPER_NUM_WORKERS", 1),
                vad_threshold=_cfg_float(overrides, "whisper_vad_threshold", "MEETINGBRO_WHISPER_VAD_THRESHOLD", 0.3),
                vad_min_speech_ms=_cfg_int(overrides, "whisper_vad_min_speech_ms", "MEETINGBRO_WHISPER_VAD_MIN_SPEECH_MS", 100),
                vad_min_silence_ms=_cfg_int(overrides, "whisper_vad_min_silence_ms", "MEETINGBRO_WHISPER_VAD_MIN_SILENCE_MS", 300),
                vad_speech_pad_ms=_cfg_int(overrides, "whisper_vad_speech_pad_ms", "MEETINGBRO_WHISPER_VAD_SPEECH_PAD_MS", 400),
                multilingual=_cfg_bool(overrides, "whisper_multilingual", "MEETINGBRO_WHISPER_MULTILINGUAL", False),
            )
            cfg = SessionConfig(
                audio_source=WavFileSource(path, sample_rate=16_000, chunk_seconds=chunk_seconds, realtime=False),
                audio_chunk_seconds=chunk_seconds,
                runtime_profile=profile_name,
                asr=asr,
                summarizer=NoopSummarizer(),
                translator=NoopTranslator(),
                storage=storage,
                forced_language=forced_language,
                summary_language="en",
                rolling_interval_seconds=999999.0,
                memory_interval_seconds=999999.0,
                cumulative_interval_seconds=999999.0,
                min_segments_for_rolling=999999,
                min_segments_for_memory=999999,
                min_segments_for_cumulative=999999,
                asr_accumulation_seconds=_cfg_float(overrides, "asr_accumulation_seconds", "MEETINGBRO_ASR_ACCUM_SECONDS", 2.5),
                silence_rms_threshold=_cfg_float(overrides, "silence_rms_threshold", "MEETINGBRO_SILENCE_RMS_THRESHOLD", 0.001),
                asr_overlap_seconds=_cfg_float(overrides, "asr_overlap_seconds", "MEETINGBRO_ASR_OVERLAP_SECONDS", 0.0),
                vocabulary_hint=_cfg_value(overrides, "vocabulary_hint", "MEETINGBRO_VOCABULARY_HINT", "") or None,
                asr_retry_enabled=_cfg_bool(overrides, "asr_retry_enabled", "MEETINGBRO_ASR_RETRY_ENABLED", True),
                asr_safeguard_enabled=_cfg_bool(overrides, "asr_safeguard_enabled", "MEETINGBRO_ASR_SAFEGUARD_ENABLED", True),
                asr_safeguard_rtf_threshold=_cfg_float(overrides, "asr_safeguard_rtf_threshold", "MEETINGBRO_ASR_SAFEGUARD_RTF_THRESHOLD", 0.9),
                asr_safeguard_cooldown_windows=_cfg_int(overrides, "asr_safeguard_cooldown_windows", "MEETINGBRO_ASR_SAFEGUARD_COOLDOWN_WINDOWS", 5),
                asr_early_flush_enabled=_cfg_bool(overrides, "asr_early_flush_enabled", "MEETINGBRO_ASR_EARLY_FLUSH_ENABLED", True),
                asr_early_flush_min_seconds=_cfg_float(overrides, "asr_early_flush_min_seconds", "MEETINGBRO_ASR_EARLY_FLUSH_MIN_SECONDS", 0.8),
                silence_commit_min_confidence=_cfg_float(overrides, "silence_commit_min_confidence", "MEETINGBRO_SILENCE_COMMIT_MIN_CONFIDENCE", 0.75),
                silence_commit_min_duration_seconds=_cfg_float(overrides, "silence_commit_min_duration_seconds", "MEETINGBRO_SILENCE_COMMIT_MIN_DURATION_SECONDS", 0.6),
                denoise_enabled=_cfg_bool(overrides, "denoise_enabled", "MEETINGBRO_DENOISE_ENABLED", False),
                denoise_strength=_cfg_float(overrides, "denoise_strength", "MEETINGBRO_DENOISE_STRENGTH", 1.1),
                denoise_noise_update_rms_threshold=_cfg_float(overrides, "denoise_noise_update_rms_threshold", "MEETINGBRO_DENOISE_NOISE_UPDATE_RMS_THRESHOLD", 0.02),
                audio_conditioning_enabled=_cfg_bool(overrides, "audio_conditioning_enabled", "MEETINGBRO_AUDIO_CONDITIONING_ENABLED", True),
                audio_conditioning_target_rms=_cfg_float(overrides, "audio_conditioning_target_rms", "MEETINGBRO_AUDIO_CONDITIONING_TARGET_RMS", 0.035),
                audio_conditioning_min_rms=_cfg_float(overrides, "audio_conditioning_min_rms", "MEETINGBRO_AUDIO_CONDITIONING_MIN_RMS", 0.003),
                audio_conditioning_max_gain=_cfg_float(overrides, "audio_conditioning_max_gain", "MEETINGBRO_AUDIO_CONDITIONING_MAX_GAIN", 2.5),
                audio_conditioning_peak_limit=_cfg_float(overrides, "audio_conditioning_peak_limit", "MEETINGBRO_AUDIO_CONDITIONING_PEAK_LIMIT", 0.98),
                pre_vad_enabled=_cfg_bool(overrides, "pre_vad_enabled", "MEETINGBRO_PRE_VAD_ENABLED", True),
                pre_vad_conditioning_enabled=_cfg_bool(overrides, "pre_vad_conditioning_enabled", "MEETINGBRO_PRE_VAD_CONDITIONING_ENABLED", True),
                pre_vad_conditioning_target_rms=_cfg_float(overrides, "pre_vad_conditioning_target_rms", "MEETINGBRO_PRE_VAD_CONDITIONING_TARGET_RMS", 0.03),
                pre_vad_conditioning_min_rms=_cfg_float(overrides, "pre_vad_conditioning_min_rms", "MEETINGBRO_PRE_VAD_CONDITIONING_MIN_RMS", 0.001),
                pre_vad_conditioning_max_gain=_cfg_float(overrides, "pre_vad_conditioning_max_gain", "MEETINGBRO_PRE_VAD_CONDITIONING_MAX_GAIN", 4.0),
                pre_vad_threshold=_cfg_float(overrides, "pre_vad_threshold", "MEETINGBRO_PRE_VAD_THRESHOLD", 0.38),
                pre_vad_energy_rms_threshold=_cfg_float(overrides, "pre_vad_energy_rms_threshold", "MEETINGBRO_PRE_VAD_ENERGY_RMS_THRESHOLD", 0.005),
                pre_vad_trailing_silence_seconds=_cfg_float(overrides, "pre_vad_trailing_silence_seconds", "MEETINGBRO_PRE_VAD_TRAILING_SILENCE_SECONDS", 0.45),
                pre_vad_adaptive_trailing_silence_enabled=_cfg_bool(overrides, "pre_vad_adaptive_trailing_silence_enabled", "MEETINGBRO_PRE_VAD_ADAPTIVE_TRAILING_SILENCE_ENABLED", True),
                pre_vad_adaptive_fast_trailing_silence_seconds=_cfg_float(overrides, "pre_vad_adaptive_fast_trailing_silence_seconds", "MEETINGBRO_PRE_VAD_ADAPTIVE_FAST_TRAILING_SILENCE_SECONDS", 0.30),
                pre_vad_adaptive_max_realtime_factor=_cfg_float(overrides, "pre_vad_adaptive_max_realtime_factor", "MEETINGBRO_PRE_VAD_ADAPTIVE_MAX_REALTIME_FACTOR", 0.5),
                pre_vad_max_segment_seconds=_cfg_float(overrides, "pre_vad_max_segment_seconds", "MEETINGBRO_PRE_VAD_MAX_SEGMENT_SECONDS", 8.0),
                weak_speech_rescue_enabled=_cfg_bool(overrides, "weak_speech_rescue_enabled", "MEETINGBRO_WEAK_SPEECH_RESCUE_ENABLED", True),
                weak_speech_rescue_rms_min=_cfg_float(overrides, "weak_speech_rescue_rms_min", "MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MIN", 0.0008),
                weak_speech_rescue_rms_max=_cfg_float(overrides, "weak_speech_rescue_rms_max", "MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MAX", 0.02),
                weak_speech_rescue_fast_rms_max=_cfg_float(overrides, "weak_speech_rescue_fast_rms_max", "MEETINGBRO_WEAK_SPEECH_RESCUE_FAST_RMS_MAX", 0.01),
                weak_speech_rescue_fast_window_seconds=_cfg_float(overrides, "weak_speech_rescue_fast_window_seconds", "MEETINGBRO_WEAK_SPEECH_RESCUE_FAST_WINDOW_SECONDS", 2.5),
                weak_speech_rescue_window_seconds=_cfg_float(overrides, "weak_speech_rescue_window_seconds", "MEETINGBRO_WEAK_SPEECH_RESCUE_WINDOW_SECONDS", 6.0),
                weak_speech_rescue_cooldown_seconds=_cfg_float(overrides, "weak_speech_rescue_cooldown_seconds", "MEETINGBRO_WEAK_SPEECH_RESCUE_COOLDOWN_SECONDS", 8.0),
                language_lock_enabled=_cfg_bool(overrides, "language_lock_enabled", "MEETINGBRO_LANGUAGE_LOCK_ENABLED", False),
            )
            manager = SessionManager(cfg)
            t0 = time.perf_counter()
            await manager.start()
            if manager._task is not None:
                await manager._task
            await manager.stop()
            wall_seconds = time.perf_counter() - t0

            segments = storage.list_segments(manager.meeting_id)
            transcript = "\n".join(seg.text for seg in segments)
            language_counts = Counter(seg.original_language for seg in segments)
            keyword = _keyword_recall(transcript, args.keywords)
            reference = getattr(args, "reference", None)
            if reference:
                score_lang = None if args.language == "auto" else args.language
                metric_name, error_rate = asr_metrics.score(reference, transcript, language=score_lang)
                wer_cer = {"metric": metric_name, "error_rate": error_rate}
            else:
                wer_cer = {"metric": None, "error_rate": None}
            if args.text_out_dir:
                out_dir = Path(args.text_out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{path.stem}.{profile_name}.txt").write_text(transcript, encoding="utf-8")

            return {
                "profile": profile_name,
                "file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                "audio_seconds": audio_duration,
                "wall_seconds": wall_seconds,
                "pipeline_realtime_factor": wall_seconds / max(audio_duration, 1e-6),
                "chunk_seconds": chunk_seconds,
                "asr_accumulation_seconds": cfg.asr_accumulation_seconds,
                "segments": len(segments),
                "transcript_chars": len(transcript),
                "languages": dict(language_counts),
                "wer_cer": wer_cer,
                "asr_last_realtime_factor": manager._state.asr_realtime_factor,
                "asr_safeguard_events": manager._state.asr_safeguard_events,
                "weak_rescue_attempts": manager._state.weak_rescue_attempts,
                "weak_rescue_emitted": manager._state.weak_rescue_emitted,
                "retry_windows_total": manager._state.retry_windows_total,
                "retry_windows_improved": manager._state.retry_windows_improved,
                "retry_windows_diverged": manager._state.retry_windows_diverged,
                "audio_drop_total": manager._audio_drop_total,
                "keyword": keyword,
                "first_segments": [
                    {
                        "start": round(seg.start_time, 2),
                        "end": round(seg.end_time, 2),
                        "language": seg.original_language,
                        "text": seg.text,
                    }
                    for seg in segments[:5]
                ],
            }
        finally:
            storage.close()


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "profile",
        "file",
        "audio",
        "chunk",
        "accum",
        "wall",
        "rtf",
        "seg",
        "chars",
        "asr_rtf",
        "safe",
        "rescue",
        "kw",
        "err",
    ]
    print("\t".join(headers))
    for row in rows:
        kw = row["keyword"]
        kw_text = "—" if kw["recall"] is None else f"{kw['matched']}/{kw['expected']}"
        wer_cer = row.get("wer_cer", {})
        err_text = (
            "—"
            if wer_cer.get("error_rate") is None
            else f"{wer_cer['metric']} {wer_cer['error_rate']:.3f}"
        )
        print(
            "\t".join(
                [
                    row["profile"],
                    row["file"],
                    f"{row['audio_seconds']:.1f}",
                    f"{row['chunk_seconds']:.1f}",
                    f"{row['asr_accumulation_seconds']:.1f}",
                    f"{row['wall_seconds']:.1f}",
                    f"{row['pipeline_realtime_factor']:.2f}",
                    str(row["segments"]),
                    str(row["transcript_chars"]),
                    "—" if row["asr_last_realtime_factor"] is None else f"{row['asr_last_realtime_factor']:.2f}",
                    str(row["asr_safeguard_events"]),
                    f"{row['weak_rescue_emitted']}/{row['weak_rescue_attempts']}",
                    kw_text,
                    err_text,
                ]
            )
        )


async def main(argv: list[str]) -> int:
    _load_dotenv_if_present()
    args = _parse_args(argv)
    if args.reference_file and not args.reference:
        ref_path = Path(args.reference_file)
        if not ref_path.is_absolute():
            ref_path = ROOT / ref_path
        args.reference = ref_path.read_text(encoding="utf-8").strip()
    wavs = _resolve_inputs(args.wav)
    if not wavs:
        print("No WAV files found.", file=sys.stderr)
        return 2

    rows = []
    for profile_name in args.profiles:
        overrides = PROFILE_PRESETS[profile_name]
        for wav in wavs:
            rows.append(await _run_one(wav, args, profile_name=profile_name, overrides=overrides))

    _print_table(rows)
    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
