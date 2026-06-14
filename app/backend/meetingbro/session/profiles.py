from __future__ import annotations

from typing import Any


DEFAULT_RUNTIME_PROFILE = "balanced"
RUNTIME_PROFILE_NAMES = {"balanced", "performance", "summary_only"}


RUNTIME_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    # Whisper-only meeting mode with live captions and full summaries.
    # Tuned for stability over speed.
    "balanced": {
        "resource_governor_policy": "balanced",
        "resource_pressure_rtf_threshold": 0.90,
        "resource_critical_rtf_threshold": 1.25,
        "resource_pressure_backlog_seconds": 3.0,
        "resource_critical_backlog_seconds": 6.0,
        "chunk_seconds": 0.75,
        "asr_accumulation_seconds": 2.0,
        "asr_early_flush_enabled": True,
        "asr_early_flush_min_seconds": 1.0,
        "silence_commit_min_confidence": 0.75,
        "silence_commit_min_duration_seconds": 0.8,
        "pre_vad_enabled": True,
        "pre_vad_trailing_silence_seconds": 0.60,
        "pre_vad_adaptive_trailing_silence_enabled": False,
        "pre_vad_adaptive_fast_trailing_silence_seconds": 0.30,
        "language_lock_enabled": False,
        "weak_speech_rescue_enabled": True,
        "weak_speech_rescue_fast_window_seconds": 3.0,
        "rolling_window_seconds": 300.0,
        "summary_tail_seconds": 300.0,
        # Live preview captions (Qwen lane) give ~1s captions vs ~5s for the
        # formal lane (measured, CPU). Degrades gracefully to a shared preview
        # when the optional Qwen model is not installed.
        "fast_preview_enabled": True,
    },
    # Performance mode: Whisper + Qwen preview lane for faster live captions
    # and dual-source context for refined clean notes.
    "performance": {
        "resource_governor_policy": "balanced",
        "resource_pressure_rtf_threshold": 1.00,
        "resource_critical_rtf_threshold": 1.35,
        "resource_pressure_backlog_seconds": 4.0,
        "resource_critical_backlog_seconds": 7.0,
        "chunk_seconds": 0.5,
        "asr_accumulation_seconds": 1.5,
        "asr_early_flush_enabled": True,
        "asr_early_flush_min_seconds": 0.7,
        "silence_commit_min_confidence": 0.80,
        "silence_commit_min_duration_seconds": 0.6,
        "pre_vad_enabled": True,
        "pre_vad_trailing_silence_seconds": 0.40,
        "pre_vad_adaptive_trailing_silence_enabled": True,
        "pre_vad_adaptive_fast_trailing_silence_seconds": 0.25,
        "language_lock_enabled": False,
        "weak_speech_rescue_enabled": True,
        "weak_speech_rescue_fast_window_seconds": 2.5,
        "rolling_window_seconds": 300.0,
        "summary_tail_seconds": 300.0,
        "fast_preview_enabled": True,
    },
    # Lowest-overhead meeting mode: no live captions/translation, slower ASR
    # cadence, and summary generation tuned for low-spec machines.
    "summary_only": {
        "resource_governor_policy": "conservative",
        "resource_pressure_rtf_threshold": 0.75,
        "resource_critical_rtf_threshold": 1.05,
        "resource_pressure_backlog_seconds": 2.5,
        "resource_critical_backlog_seconds": 4.5,
        "chunk_seconds": 1.0,
        "asr_accumulation_seconds": 4.0,
        "asr_early_flush_enabled": False,
        "asr_early_flush_min_seconds": 1.2,
        "silence_commit_min_confidence": 0.75,
        "silence_commit_min_duration_seconds": 0.9,
        "pre_vad_enabled": True,
        "pre_vad_trailing_silence_seconds": 0.75,
        "pre_vad_adaptive_trailing_silence_enabled": False,
        "pre_vad_adaptive_fast_trailing_silence_seconds": 0.35,
        "language_lock_enabled": False,
        "weak_speech_rescue_enabled": True,
        "weak_speech_rescue_fast_window_seconds": 4.0,
        "rolling_window_seconds": 300.0,
        "summary_tail_seconds": 300.0,
        "rolling_interval_seconds": 45.0,
        "memory_interval_seconds": 120.0,
        "cumulative_interval_seconds": 180.0,
        "refinement_interval_seconds": 90.0,
        "min_segments_for_rolling": 3,
        "min_segments_for_memory": 4,
        "min_segments_for_cumulative": 4,
        "min_segments_for_refinement": 2,
        "live_translation_backfill_limit": 0,
        "live_translation_max_pending": 1,
        "live_translation_safeguard_max_pending": 1,
        "fast_preview_enabled": False,
    },
}


def normalize_runtime_profile(name: str | None) -> str:
    normalized = (name or DEFAULT_RUNTIME_PROFILE).strip().lower().replace("-", "_")
    aliases = {
        "low_latency": "balanced",
        "multilingual": "balanced",
        "single_language": "balanced",
        "robust": "performance",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in RUNTIME_PROFILE_NAMES:
        return normalized
    return DEFAULT_RUNTIME_PROFILE


def runtime_profile_defaults(name: str | None) -> dict[str, Any]:
    return dict(RUNTIME_PROFILE_PRESETS[normalize_runtime_profile(name)])
