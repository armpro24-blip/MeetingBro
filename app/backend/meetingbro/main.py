from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .asr.base import ASRAdapter
from .asr.faster_whisper_adapter import FasterWhisperAdapter
from .audio import AudioSource, MicrophoneSource, MixedAudioSource, SystemAudioLoopbackSource, WavFileSource
from .exporter import export_meeting
from .hardware import HardwareProfile, detect_hardware_profile
from .llm.openai_compatible import _load_dotenv_if_present
from .schemas import (
    CreateNoteRequest,
    ErrorPayload,
    ExportMeetingRequest,
    ExportMeetingResponse,
    LanguageCode,
    Note,
    SessionStatePayload,
    SummarySnapshot,
    TranscriptSegment,
)
from .session.manager import SessionConfig, SessionManager
from .session.profiles import DEFAULT_RUNTIME_PROFILE, normalize_runtime_profile, runtime_profile_defaults
from .storage.db import Storage
from .summarization.llm import LLMSummarizer
from .translation.llm import LLMTranslator

logger = logging.getLogger("meetingbro")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "meetingbro.db"
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "exports"


def _recommended_asr_executor_workers() -> int:
    # SessionManager drives at most one ASR call at a time per session, so more
    # than one ASR worker per session adds thread overhead without throughput gain.
    return 1


def _recommended_preview_asr_executor_workers() -> int:
    return 1


def _recommended_summary_executor_workers() -> int:
    cores = os.cpu_count() or 4
    if cores >= 12:
        return 2
    return 1


def _recommended_translation_executor_workers() -> int:
    return 2


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("invalid %s=%r, using default %.3f", name, value, default)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("invalid %s=%r, using default %d", name, value, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("invalid %s=%r, using default %s", name, value, default)
    return default


def _env_str_auto(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip()
    if not normalized or normalized.lower() == "auto":
        return default
    return normalized


def _profile_value(profile: dict[str, object], key: str, env_name: str, default: object) -> object:
    if env_name in os.environ:
        return os.environ[env_name]
    return profile.get(key, default)


def _profile_float(profile: dict[str, object], key: str, env_name: str, default: float) -> float:
    value = _profile_value(profile, key, env_name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("invalid %s/%s=%r, using default %.3f", key, env_name, value, default)
        return default


def _profile_bool(profile: dict[str, object], key: str, env_name: str, default: bool) -> bool:
    value = _profile_value(profile, key, env_name, default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("invalid %s/%s=%r, using default %s", key, env_name, value, default)
    return default


def _runtime_settings_from_profile(profile_name: str) -> dict[str, object]:
    profile = runtime_profile_defaults(profile_name)
    return {
        "asr_accumulation_seconds": _profile_float(profile, "asr_accumulation_seconds", "MEETINGBRO_ASR_ACCUM_SECONDS", 1.5),
        "asr_early_flush_enabled": _profile_bool(profile, "asr_early_flush_enabled", "MEETINGBRO_ASR_EARLY_FLUSH_ENABLED", True),
        "asr_early_flush_min_seconds": _profile_float(profile, "asr_early_flush_min_seconds", "MEETINGBRO_ASR_EARLY_FLUSH_MIN_SECONDS", 0.8),
        "silence_commit_min_confidence": _profile_float(profile, "silence_commit_min_confidence", "MEETINGBRO_SILENCE_COMMIT_MIN_CONFIDENCE", 0.75),
        "silence_commit_min_duration_seconds": _profile_float(profile, "silence_commit_min_duration_seconds", "MEETINGBRO_SILENCE_COMMIT_MIN_DURATION_SECONDS", 0.6),
        "pre_vad_enabled": _profile_bool(profile, "pre_vad_enabled", "MEETINGBRO_PRE_VAD_ENABLED", True),
        "pre_vad_trailing_silence_seconds": _profile_float(profile, "pre_vad_trailing_silence_seconds", "MEETINGBRO_PRE_VAD_TRAILING_SILENCE_SECONDS", 0.45),
        "pre_vad_adaptive_trailing_silence_enabled": _profile_bool(profile, "pre_vad_adaptive_trailing_silence_enabled", "MEETINGBRO_PRE_VAD_ADAPTIVE_TRAILING_SILENCE_ENABLED", True),
        "pre_vad_adaptive_fast_trailing_silence_seconds": _profile_float(profile, "pre_vad_adaptive_fast_trailing_silence_seconds", "MEETINGBRO_PRE_VAD_ADAPTIVE_FAST_TRAILING_SILENCE_SECONDS", 0.30),
        "language_lock_enabled": _profile_bool(profile, "language_lock_enabled", "MEETINGBRO_LANGUAGE_LOCK_ENABLED", False),
        "weak_speech_rescue_enabled": _profile_bool(profile, "weak_speech_rescue_enabled", "MEETINGBRO_WEAK_SPEECH_RESCUE_ENABLED", True),
        "weak_speech_rescue_fast_window_seconds": _profile_float(profile, "weak_speech_rescue_fast_window_seconds", "MEETINGBRO_WEAK_SPEECH_RESCUE_FAST_WINDOW_SECONDS", 2.5),
        "resource_governor_policy": str(_profile_value(profile, "resource_governor_policy", "MEETINGBRO_RESOURCE_GOVERNOR_POLICY", "balanced")),
        "resource_pressure_rtf_threshold": _profile_float(profile, "resource_pressure_rtf_threshold", "MEETINGBRO_RESOURCE_PRESSURE_RTF_THRESHOLD", 0.9),
        "resource_critical_rtf_threshold": _profile_float(profile, "resource_critical_rtf_threshold", "MEETINGBRO_RESOURCE_CRITICAL_RTF_THRESHOLD", 1.25),
        "resource_pressure_backlog_seconds": _profile_float(profile, "resource_pressure_backlog_seconds", "MEETINGBRO_RESOURCE_PRESSURE_BACKLOG_SECONDS", 3.0),
        "resource_critical_backlog_seconds": _profile_float(profile, "resource_critical_backlog_seconds", "MEETINGBRO_RESOURCE_CRITICAL_BACKLOG_SECONDS", 6.0),
        "rolling_window_seconds": _profile_float(profile, "rolling_window_seconds", "MEETINGBRO_ROLLING_WINDOW_SECONDS", 180.0),
        "summary_tail_seconds": _profile_float(profile, "summary_tail_seconds", "MEETINGBRO_SUMMARY_TAIL_SECONDS", 120.0),
        "rolling_interval_seconds": _profile_float(profile, "rolling_interval_seconds", "MEETINGBRO_ROLLING_INTERVAL_SECONDS", 60.0),
        "memory_interval_seconds": _profile_float(profile, "memory_interval_seconds", "MEETINGBRO_MEMORY_INTERVAL_SECONDS", 120.0),
        "cumulative_interval_seconds": _profile_float(profile, "cumulative_interval_seconds", "MEETINGBRO_CUMULATIVE_INTERVAL_SECONDS", 180.0),
        "refinement_interval_seconds": _profile_float(profile, "refinement_interval_seconds", "MEETINGBRO_REFINEMENT_INTERVAL_SECONDS", 180.0),
        "min_segments_for_rolling": int(_profile_value(profile, "min_segments_for_rolling", "MEETINGBRO_MIN_SEGMENTS_FOR_ROLLING", 1)),
        "min_segments_for_memory": int(_profile_value(profile, "min_segments_for_memory", "MEETINGBRO_MIN_SEGMENTS_FOR_MEMORY", 3)),
        "min_segments_for_cumulative": int(_profile_value(profile, "min_segments_for_cumulative", "MEETINGBRO_MIN_SEGMENTS_FOR_CUMULATIVE", 3)),
        "min_segments_for_refinement": int(_profile_value(profile, "min_segments_for_refinement", "MEETINGBRO_MIN_SEGMENTS_FOR_REFINEMENT", 2)),
        "live_translation_backfill_limit": int(_profile_value(profile, "live_translation_backfill_limit", "MEETINGBRO_LIVE_TRANSLATION_BACKFILL_LIMIT", 20)),
        "live_translation_max_pending": int(_profile_value(profile, "live_translation_max_pending", "MEETINGBRO_LIVE_TRANSLATION_MAX_PENDING", 12)),
        "live_translation_safeguard_max_pending": int(_profile_value(profile, "live_translation_safeguard_max_pending", "MEETINGBRO_LIVE_TRANSLATION_SAFEGUARD_MAX_PENDING", 4)),
        "fast_preview_enabled": _profile_bool(profile, "fast_preview_enabled", "MEETINGBRO_FAST_PREVIEW_ENABLED", True),
    }


def _chunk_seconds_for_profile(profile_name: str) -> float:
    profile = runtime_profile_defaults(profile_name)
    return _profile_float(profile, "chunk_seconds", "MEETINGBRO_CHUNK_SECONDS", 0.5)


def _language_lock_enabled_for_speech(forced_language: Optional[str]) -> bool:
    return forced_language in {"zh", "en", "de"}


def _resolve_preview_runtime(
    *,
    profile_settings: dict[str, object],
    preview_asr: Optional[ASRAdapter],
    preview_backend_name: str,
    preview_device: Optional[str],
    formal_device: str,
) -> tuple[Optional[ASRAdapter], str, Optional[str]]:
    if bool(profile_settings.get("fast_preview_enabled", True)):
        return preview_asr, preview_backend_name, preview_device
    return None, "shared", formal_device


def _configured_preview_runtime(
    hardware: HardwareProfile | None,
    *,
    formal_device: str,
) -> tuple[str, str]:
    preview_backend = os.environ.get("MEETINGBRO_PREVIEW_ASR_BACKEND", "").strip().lower()
    if preview_backend == "qwen3":
        return (
            "qwen3",
            _env_str_auto(
                "MEETINGBRO_PREVIEW_QWEN3_PROVIDER",
                hardware.recommended_qwen_provider if hardware is not None else "cpu",
            ),
        )

    preview_size = os.environ.get("MEETINGBRO_PREVIEW_WHISPER_SIZE", "").strip()
    if not preview_size or preview_size.lower() == "shared":
        return "shared", formal_device

    preview_default_device = "cpu"
    if hardware is not None and hardware.recommended_whisper_device != "cuda":
        preview_default_device = hardware.recommended_whisper_device
    return (
        "faster_whisper",
        _env_str_auto("MEETINGBRO_PREVIEW_WHISPER_DEVICE", preview_default_device),
    )


def _should_load_preview_asr(
    *,
    profile_settings: dict[str, object],
    preview_backend_name: str,
) -> bool:
    return bool(profile_settings.get("fast_preview_enabled", True)) and preview_backend_name != "shared"


def _enforce_performance_requires_qwen(
    profile_name: str,
    *,
    preview_backend_name: str,
) -> tuple[str, Optional[str]]:
    if profile_name == "performance" and preview_backend_name != "qwen3":
        return (
            "balanced",
            "performance mode requires Qwen preview backend; switched to Balanced",
        )
    return profile_name, None


def _maybe_start_preview_prewarm(
    app: FastAPI,
    *,
    preview_asr: Optional[ASRAdapter],
    preview_backend_name: str,
    profile_name: str,
    reason: str,
) -> None:
    if not (
        preview_backend_name == "qwen3"
        and preview_asr is not None
        and _env_bool("MEETINGBRO_PREVIEW_QWEN3_PREWARM", True)
        and hasattr(preview_asr, "prewarm")
    ):
        app.state.preview_asr_prewarm_future = None
        return

    loop = asyncio.get_running_loop()
    prewarm_future = loop.run_in_executor(None, preview_asr.prewarm)  # type: ignore[attr-defined]

    def _log_preview_prewarm_done(task) -> None:
        try:
            task.result()
        except Exception as exc:
            logger.warning("Qwen3 preview ASR prewarm failed: %s", exc)
        else:
            logger.info("Qwen3 preview ASR prewarm completed")

    prewarm_future.add_done_callback(_log_preview_prewarm_done)
    app.state.preview_asr_prewarm_future = prewarm_future
    logger.info(
        "starting Qwen3 preview ASR prewarm reason=%s runtime_profile=%s",
        reason,
        profile_name,
    )


async def _ensure_preview_asr_loaded(app: FastAPI, *, profile_name: str) -> None:
    if getattr(app.state, "preview_asr", None) is not None:
        return
    if getattr(app.state, "preview_asr_backend_name", "shared") == "shared":
        return

    async with app.state.preview_asr_build_lock:
        if getattr(app.state, "preview_asr", None) is not None:
            return

        preview_asr = _build_preview_asr(getattr(app.state, "hardware_profile", None))
        app.state.preview_asr = preview_asr
        if preview_asr is None:
            app.state.preview_asr_backend_name = "shared"
            app.state.preview_asr_device = getattr(app.state, "formal_asr_device", "cpu")
            app.state.preview_asr_prewarm_future = None
            return

        logger.info(
            "loaded preview ASR on demand runtime_profile=%s backend=%s device=%s",
            profile_name,
            app.state.preview_asr_backend_name,
            app.state.preview_asr_device,
        )
        _maybe_start_preview_prewarm(
            app,
            preview_asr=preview_asr,
            preview_backend_name=app.state.preview_asr_backend_name,
            profile_name=profile_name,
            reason="on_demand",
        )


def _build_preview_asr(hardware: HardwareProfile | None = None) -> ASRAdapter | None:
    backend = os.environ.get("MEETINGBRO_PREVIEW_ASR_BACKEND", "").strip().lower()

    if backend == "qwen3":
        # Lazy import so that a missing sherpa-onnx never affects normal startup.
        from .asr.qwen3_asr_adapter import Qwen3ASRAdapter

        model_dir = os.environ.get("MEETINGBRO_PREVIEW_QWEN3_MODEL_DIR", "").strip()
        if not model_dir:
            raise RuntimeError(
                "MEETINGBRO_PREVIEW_ASR_BACKEND=qwen3 requires "
                "MEETINGBRO_PREVIEW_QWEN3_MODEL_DIR to be set."
            )
        # Resolve relative paths against the project root so the backend can
        # be started from any working directory.
        model_dir_path = Path(model_dir)
        if not model_dir_path.is_absolute():
            model_dir_path = PROJECT_ROOT / model_dir_path
        return Qwen3ASRAdapter(
            model_dir=model_dir_path,
            num_threads=_env_int(
                "MEETINGBRO_PREVIEW_QWEN3_NUM_THREADS",
                hardware.recommended_qwen_threads if hardware is not None else 2,
            ),
            provider=_env_str_auto(
                "MEETINGBRO_PREVIEW_QWEN3_PROVIDER",
                hardware.recommended_qwen_provider if hardware is not None else "cpu",
            ),
            max_total_len=_env_int("MEETINGBRO_PREVIEW_QWEN3_MAX_TOTAL_LEN", 256),
            max_new_tokens=_env_int("MEETINGBRO_PREVIEW_QWEN3_MAX_NEW_TOKENS", 96),
            filter_language_script=_env_bool(
                "MEETINGBRO_PREVIEW_QWEN3_FILTER_LANGUAGE_SCRIPT", True
            ),
            suppress_fillers=_env_bool("MEETINGBRO_PREVIEW_QWEN3_SUPPRESS_FILLERS", True),
        )

    # Default: dedicated faster-whisper model, or None → fall back to shared model.
    model_size = os.environ.get("MEETINGBRO_PREVIEW_WHISPER_SIZE", "").strip()
    if not model_size or model_size.lower() == "shared":
        return None
    preview_default_device = "cpu"
    if hardware is not None and hardware.recommended_whisper_device != "cuda":
        preview_default_device = hardware.recommended_whisper_device
    preview_device = _env_str_auto("MEETINGBRO_PREVIEW_WHISPER_DEVICE", preview_default_device)
    preview_compute_default = "int8" if preview_device == "cpu" else (
        hardware.recommended_whisper_compute_type if hardware is not None else "float16"
    )

    return FasterWhisperAdapter(
        model_size=model_size,
        device=preview_device,
        compute_type=_env_str_auto("MEETINGBRO_PREVIEW_WHISPER_COMPUTE_TYPE", preview_compute_default),
        beam_size=_env_int("MEETINGBRO_PREVIEW_WHISPER_BEAM_SIZE", 1),
        cpu_threads=_env_int("MEETINGBRO_PREVIEW_WHISPER_CPU_THREADS", 0),
        num_workers=_env_int("MEETINGBRO_PREVIEW_WHISPER_NUM_WORKERS", 1),
        vad_threshold=_env_float(
            "MEETINGBRO_PREVIEW_WHISPER_VAD_THRESHOLD",
            _env_float("MEETINGBRO_WHISPER_VAD_THRESHOLD", 0.3),
        ),
        vad_min_speech_ms=_env_int(
            "MEETINGBRO_PREVIEW_WHISPER_VAD_MIN_SPEECH_MS",
            _env_int("MEETINGBRO_WHISPER_VAD_MIN_SPEECH_MS", 100),
        ),
        vad_min_silence_ms=_env_int(
            "MEETINGBRO_PREVIEW_WHISPER_VAD_MIN_SILENCE_MS",
            _env_int("MEETINGBRO_WHISPER_VAD_MIN_SILENCE_MS", 300),
        ),
        vad_speech_pad_ms=_env_int(
            "MEETINGBRO_PREVIEW_WHISPER_VAD_SPEECH_PAD_MS",
            _env_int("MEETINGBRO_WHISPER_VAD_SPEECH_PAD_MS", 400),
        ),
        multilingual=_env_bool("MEETINGBRO_PREVIEW_WHISPER_MULTILINGUAL", True),
        language_detection_threshold=_env_float(
            "MEETINGBRO_PREVIEW_WHISPER_LANGUAGE_DETECTION_THRESHOLD",
            0.5,
        ),
        language_detection_segments=_env_int(
            "MEETINGBRO_PREVIEW_WHISPER_LANGUAGE_DETECTION_SEGMENTS",
            1,
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_dotenv_if_present()
    hardware = detect_hardware_profile()
    startup_runtime_profile = normalize_runtime_profile(
        os.environ.get("MEETINGBRO_RUNTIME_PROFILE")
        or getattr(hardware, "recommended_runtime_profile", DEFAULT_RUNTIME_PROFILE)
    )
    startup_profile_settings = _runtime_settings_from_profile(startup_runtime_profile)
    storage = Storage(Path(os.environ.get("MEETINGBRO_DB", str(DEFAULT_DB_PATH))))
    whisper_size = _env_str_auto("MEETINGBRO_WHISPER_SIZE", hardware.recommended_whisper_size)
    whisper_device = _env_str_auto("MEETINGBRO_WHISPER_DEVICE", hardware.recommended_whisper_device)
    whisper_compute_type = _env_str_auto("MEETINGBRO_WHISPER_COMPUTE_TYPE", hardware.recommended_whisper_compute_type)
    asr = FasterWhisperAdapter(
        model_size=whisper_size,
        device=whisper_device,
        compute_type=whisper_compute_type,
        beam_size=_env_int("MEETINGBRO_WHISPER_BEAM_SIZE", 1),
        cpu_threads=_env_int("MEETINGBRO_WHISPER_CPU_THREADS", 0),
        num_workers=_env_int("MEETINGBRO_WHISPER_NUM_WORKERS", 1),
        vad_threshold=_env_float("MEETINGBRO_WHISPER_VAD_THRESHOLD", 0.3),
        vad_min_speech_ms=_env_int("MEETINGBRO_WHISPER_VAD_MIN_SPEECH_MS", 100),
        vad_min_silence_ms=_env_int("MEETINGBRO_WHISPER_VAD_MIN_SILENCE_MS", 300),
        vad_speech_pad_ms=_env_int("MEETINGBRO_WHISPER_VAD_SPEECH_PAD_MS", 400),
        multilingual=_env_bool("MEETINGBRO_WHISPER_MULTILINGUAL", False),
        language_detection_threshold=_env_float("MEETINGBRO_WHISPER_LANGUAGE_DETECTION_THRESHOLD", 0.5),
        language_detection_segments=_env_int("MEETINGBRO_WHISPER_LANGUAGE_DETECTION_SEGMENTS", 1),
    )
    _preview_asr_backend_name, _preview_asr_device = _configured_preview_runtime(
        hardware,
        formal_device=whisper_device,
    )
    preview_asr = None
    if _should_load_preview_asr(
        profile_settings=startup_profile_settings,
        preview_backend_name=_preview_asr_backend_name,
    ):
        preview_asr = _build_preview_asr(hardware)
    app.state.storage = storage
    app.state.hardware_profile = hardware
    app.state.asr = asr
    app.state.formal_asr_device = whisper_device
    app.state.preview_asr = preview_asr
    app.state.preview_asr_backend_name = _preview_asr_backend_name
    app.state.preview_asr_device = _preview_asr_device
    app.state.preview_asr_build_lock = asyncio.Lock()
    if preview_asr is not None:
        _maybe_start_preview_prewarm(
            app,
            preview_asr=preview_asr,
            preview_backend_name=_preview_asr_backend_name,
            profile_name=startup_runtime_profile,
            reason="startup",
        )
    else:
        if _preview_asr_backend_name != "shared" and not bool(startup_profile_settings.get("fast_preview_enabled", True)):
            logger.info(
                "skipping preview ASR load because runtime profile %s disables fast preview",
                startup_runtime_profile,
            )
        app.state.preview_asr_prewarm_future = None
    logger.info(
        "MeetingBro backend starting db=%s hardware=%s hardware_summary=%s whisper_size=%s whisper_device=%s compute=%s beam=%d cpu_threads=%d whisper_workers=%d preview_backend=%s preview_device=%s preview_compute=%s preview_asr_workers=%d preview_loaded=%s chunk=%.2fs accum=%.2fs silence_rms=%.4f audio_conditioning=%s denoise=%s pre_vad=%s pre_vad_conditioning=%s pre_vad_threshold=%.2f pre_vad_energy_rms=%.4f pre_vad_max=%.1fs weak_rescue=%s weak_rescue_rms=%.4f..%.4f language_lock=%s asr_retry=%s asr_safeguard=%s safeguard_rtf=%.2f suspicious_no_speech=%.2f suspicious_avg_logprob=%.2f suspicious_compression=%.2f mixed_mic_gain=%.2f mixed_system_gain=%.2f mixed_auto_balance=%s mixed_max_mic_boost=%.2f asr_workers=%d summary_workers=%d translation_workers=%d translation_backfill=%d translation_max_pending=%d",
        storage._db_path,
        hardware.label,
        hardware.summary,
        whisper_size,
        whisper_device,
        whisper_compute_type,
        _env_int("MEETINGBRO_WHISPER_BEAM_SIZE", 1),
        _env_int("MEETINGBRO_WHISPER_CPU_THREADS", 0),
        _env_int("MEETINGBRO_WHISPER_NUM_WORKERS", 1),
        _preview_asr_backend_name,
        _preview_asr_device,
        _env_str_auto("MEETINGBRO_PREVIEW_WHISPER_COMPUTE_TYPE", "int8"),
        _env_int(
            "MEETINGBRO_PREVIEW_ASR_EXECUTOR_WORKERS",
            _recommended_preview_asr_executor_workers(),
        ),
        preview_asr is not None,
        _env_float("MEETINGBRO_CHUNK_SECONDS", 0.5),
        _env_float("MEETINGBRO_ASR_ACCUM_SECONDS", 1.5),
        _env_float("MEETINGBRO_SILENCE_RMS_THRESHOLD", 0.002),
        _env_bool("MEETINGBRO_AUDIO_CONDITIONING_ENABLED", True),
        _env_bool("MEETINGBRO_DENOISE_ENABLED", False),
        _env_bool("MEETINGBRO_PRE_VAD_ENABLED", True),
        _env_bool("MEETINGBRO_PRE_VAD_CONDITIONING_ENABLED", True),
        _env_float("MEETINGBRO_PRE_VAD_THRESHOLD", 0.38),
        _env_float("MEETINGBRO_PRE_VAD_ENERGY_RMS_THRESHOLD", 0.005),
        _env_float("MEETINGBRO_PRE_VAD_MAX_SEGMENT_SECONDS", 8.0),
        _env_bool("MEETINGBRO_WEAK_SPEECH_RESCUE_ENABLED", True),
        _env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MIN", 0.0008),
        _env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MAX", 0.02),
        _env_bool("MEETINGBRO_LANGUAGE_LOCK_ENABLED", False),
        _env_bool("MEETINGBRO_ASR_RETRY_ENABLED", True),
        _env_bool("MEETINGBRO_ASR_SAFEGUARD_ENABLED", True),
        _env_float("MEETINGBRO_ASR_SAFEGUARD_RTF_THRESHOLD", 0.9),
        _env_float("MEETINGBRO_SUSPICIOUS_SEGMENT_NO_SPEECH_PROB", 0.6),
        _env_float("MEETINGBRO_SUSPICIOUS_SEGMENT_AVG_LOGPROB", -0.9),
        _env_float("MEETINGBRO_SUSPICIOUS_SEGMENT_COMPRESSION_RATIO", 2.1),
        _env_float("MEETINGBRO_MIXED_MIC_GAIN", 1.2),
        _env_float("MEETINGBRO_MIXED_SYSTEM_GAIN", 1.0),
        _env_bool("MEETINGBRO_MIXED_AUTO_BALANCE_ENABLED", True),
        _env_float("MEETINGBRO_MIXED_MAX_MIC_BOOST", 1.8),
        _env_int("MEETINGBRO_ASR_EXECUTOR_WORKERS", _recommended_asr_executor_workers()),
        _env_int("MEETINGBRO_SUMMARY_EXECUTOR_WORKERS", _recommended_summary_executor_workers()),
        _env_int("MEETINGBRO_TRANSLATION_EXECUTOR_WORKERS", _recommended_translation_executor_workers()),
        _env_int("MEETINGBRO_LIVE_TRANSLATION_BACKFILL_LIMIT", 20),
        _env_int("MEETINGBRO_LIVE_TRANSLATION_MAX_PENDING", 12),
    )
    try:
        yield
    finally:
        storage.close()
        logger.info("MeetingBro backend stopping")


app = FastAPI(title="MeetingBro backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/notes", response_model=Note)
async def create_note(req: CreateNoteRequest) -> Note:
    storage: Storage = app.state.storage
    note = Note(
        id=str(uuid.uuid4()),
        meeting_id=req.meeting_id,
        content=req.content,
        source_type=req.source_type,
        source_id=req.source_id,
        created_at=datetime.now(tz=timezone.utc),
    )
    storage.insert_note(note)
    return note


@app.get("/meetings/{meeting_id}/notes", response_model=list[Note])
async def list_notes(meeting_id: str) -> list[Note]:
    storage: Storage = app.state.storage
    return storage.list_notes(meeting_id)


@app.get("/meetings/{meeting_id}/transcript", response_model=list[TranscriptSegment])
async def list_transcript(meeting_id: str) -> list[TranscriptSegment]:
    storage: Storage = app.state.storage
    return storage.list_segments(meeting_id)


@app.get("/meetings/{meeting_id}/summaries", response_model=list[SummarySnapshot])
async def list_summaries(
    meeting_id: str,
    summary_type: Optional[str] = Query(default=None),
) -> list[SummarySnapshot]:
    storage: Storage = app.state.storage
    return storage.list_snapshots(meeting_id, summary_type=summary_type)  # type: ignore[arg-type]


@app.post("/meetings/{meeting_id}/export", response_model=ExportMeetingResponse)
async def export_meeting_endpoint(
    meeting_id: str,
    bilingual: bool = Query(False),
    target_language: Optional[LanguageCode] = Query(None),
    req: ExportMeetingRequest | None = None,
) -> ExportMeetingResponse:
    storage: Storage = app.state.storage
    req_data = req.model_dump(exclude_none=True) if req is not None else {}
    requested_export_root = req_data.pop("export_root", None)
    requested_export_dir = req_data.pop("export_dir", None)
    export_root = Path(requested_export_root or os.environ.get("MEETINGBRO_EXPORT_ROOT", str(DEFAULT_EXPORT_ROOT)))
    try:
        return export_meeting(
            storage,
            meeting_id=meeting_id,
            export_root=export_root,
            export_dir=Path(requested_export_dir) if requested_export_dir else None,
            client_metadata=req_data,
            bilingual=bilingual,
            target_language=target_language,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="meeting not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"export failed: {exc}")


def _build_audio_source(source: str, *, chunk_seconds: Optional[float] = None) -> AudioSource:
    """Build an AudioSource from a URI-like spec.

    - "mic" or empty: microphone capture (offline / in-person mode).
    - "loopback" / "system": platform system-audio loopback (Windows WASAPI,
      macOS virtual loopback device, Linux PulseAudio/PipeWire monitor).
    - "mixed": microphone + system loopback mixed together.
    - "file:<path>": replay a WAV file for the E2E vertical-slice test.
    """
    chunk_seconds = chunk_seconds if chunk_seconds is not None else _env_float("MEETINGBRO_CHUNK_SECONDS", 0.5)
    if not source or source == "mic":
        return MicrophoneSource(sample_rate=16_000, chunk_seconds=chunk_seconds)
    if source in ("loopback", "system"):
        return SystemAudioLoopbackSource(sample_rate=16_000, chunk_seconds=chunk_seconds)
    if source == "mixed":
        return MixedAudioSource(
            sample_rate=16_000,
            chunk_seconds=chunk_seconds,
            microphone_gain=_env_float("MEETINGBRO_MIXED_MIC_GAIN", 1.2),
            system_gain=_env_float("MEETINGBRO_MIXED_SYSTEM_GAIN", 1.0),
            auto_balance_enabled=_env_bool("MEETINGBRO_MIXED_AUTO_BALANCE_ENABLED", True),
            max_microphone_boost=_env_float("MEETINGBRO_MIXED_MAX_MIC_BOOST", 1.8),
            microphone_activity_floor=_env_float("MEETINGBRO_MIXED_MIC_ACTIVITY_FLOOR", 0.008),
            balance_smoothing=_env_float("MEETINGBRO_MIXED_BALANCE_SMOOTHING", 0.35),
        )
    if source.startswith("file:"):
        path = Path(source[len("file:") :]).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return WavFileSource(
            path,
            sample_rate=16_000,
            chunk_seconds=_env_float("MEETINGBRO_FILE_CHUNK_SECONDS", 5.0),
            realtime=False,
        )
    raise ValueError(f"unsupported audio source: {source}")


async def _forward_events(manager: SessionManager, ws: WebSocket) -> None:
    async for ev in manager.events():
        await ws.send_text(json.dumps({"type": ev.type, "payload": ev.payload}))


@app.websocket("/ws/session")
async def session_ws(
    ws: WebSocket,
    source: str = Query(default="mic"),
    summary_language: str = Query(default="en"),
    forced_language: Optional[str] = Query(default=None),
    vocabulary_hint: Optional[str] = Query(default=None),
    runtime_profile: Optional[str] = Query(default=None),
) -> None:
    await ws.accept()

    summary_lang: LanguageCode = summary_language if summary_language in ("zh", "en", "de") else "en"  # type: ignore[assignment]
    hardware_profile = getattr(app.state, "hardware_profile", None)
    default_runtime_profile = (
        hardware_profile.recommended_runtime_profile
        if hardware_profile is not None
        else DEFAULT_RUNTIME_PROFILE
    )
    profile_name = normalize_runtime_profile(
        runtime_profile or os.environ.get("MEETINGBRO_RUNTIME_PROFILE") or default_runtime_profile
    )
    profile_settings = _runtime_settings_from_profile(profile_name)
    profile_settings["language_lock_enabled"] = _language_lock_enabled_for_speech(forced_language)
    chunk_seconds = _chunk_seconds_for_profile(profile_name)
    if _should_load_preview_asr(
        profile_settings=profile_settings,
        preview_backend_name=getattr(app.state, "preview_asr_backend_name", "shared"),
    ):
        await _ensure_preview_asr_loaded(app, profile_name=profile_name)
    session_preview_asr, session_preview_backend_name, session_preview_device = _resolve_preview_runtime(
        profile_settings=profile_settings,
        preview_asr=app.state.preview_asr,
        preview_backend_name=app.state.preview_asr_backend_name,
        preview_device=getattr(app.state, "preview_asr_device", None),
        formal_device=getattr(app.state, "formal_asr_device", "cpu"),
    )
    profile_name, performance_warning = _enforce_performance_requires_qwen(
        profile_name,
        preview_backend_name=session_preview_backend_name,
    )
    if performance_warning is not None:
        profile_settings = _runtime_settings_from_profile(profile_name)
        profile_settings["language_lock_enabled"] = _language_lock_enabled_for_speech(forced_language)
        chunk_seconds = _chunk_seconds_for_profile(profile_name)
        session_preview_asr, session_preview_backend_name, session_preview_device = _resolve_preview_runtime(
            profile_settings=profile_settings,
            preview_asr=app.state.preview_asr,
            preview_backend_name=app.state.preview_asr_backend_name,
            preview_device=getattr(app.state, "preview_asr_device", None),
            formal_device=getattr(app.state, "formal_asr_device", "cpu"),
        )
        await ws.send_text(
            json.dumps(
                {
                    "type": "error",
                    "payload": ErrorPayload(
                        code="performance_requires_qwen",
                        message=performance_warning,
                    ).model_dump(),
                }
            )
        )

    try:
        audio_source = _build_audio_source(source, chunk_seconds=chunk_seconds)
    except Exception as exc:
        await ws.send_text(
            json.dumps(
                {
                    "type": "error",
                    "payload": ErrorPayload(code="bad_source", message=str(exc)).model_dump(),
                }
            )
        )
        await ws.close()
        return

    config = SessionConfig(
        audio_source=audio_source,
        audio_source_name=source,
        audio_chunk_seconds=chunk_seconds,
        runtime_profile=profile_name,
        asr=app.state.asr,
        preview_asr=session_preview_asr,
        preview_asr_backend_name=session_preview_backend_name,
        hardware_profile_label=getattr(app.state.hardware_profile, "label", "unknown"),
        hardware_summary=getattr(app.state.hardware_profile, "summary", None),
        formal_asr_device=getattr(app.state, "formal_asr_device", "cpu"),
        preview_asr_device=session_preview_device,
        compute_gpu_available=getattr(app.state.hardware_profile, "ctranslate2_cuda_available", False),
        preview_asr_fallback_on_error=_env_bool(
            "MEETINGBRO_PREVIEW_FALLBACK_ON_ERROR",
            session_preview_backend_name != "qwen3",
        ),
        summarizer=LLMSummarizer(),
        translator=LLMTranslator(),
        storage=app.state.storage,
        forced_language=forced_language,
        summary_language=summary_lang,
        live_translation_language=None,
        asr_accumulation_seconds=float(profile_settings["asr_accumulation_seconds"]),
        silence_rms_threshold=_env_float("MEETINGBRO_SILENCE_RMS_THRESHOLD", 0.002),
        asr_overlap_seconds=_env_float("MEETINGBRO_ASR_OVERLAP_SECONDS", 0.0),
        vocabulary_hint=vocabulary_hint or os.environ.get("MEETINGBRO_VOCABULARY_HINT") or None,
        suspicious_segment_no_speech_prob=_env_float(
            "MEETINGBRO_SUSPICIOUS_SEGMENT_NO_SPEECH_PROB",
            0.6,
        ),
        suspicious_segment_avg_logprob=_env_float(
            "MEETINGBRO_SUSPICIOUS_SEGMENT_AVG_LOGPROB",
            -0.9,
        ),
        suspicious_segment_compression_ratio=_env_float(
            "MEETINGBRO_SUSPICIOUS_SEGMENT_COMPRESSION_RATIO",
            2.1,
        ),
        asr_retry_enabled=_env_bool("MEETINGBRO_ASR_RETRY_ENABLED", True),
        asr_safeguard_enabled=_env_bool("MEETINGBRO_ASR_SAFEGUARD_ENABLED", True),
        asr_safeguard_rtf_threshold=_env_float("MEETINGBRO_ASR_SAFEGUARD_RTF_THRESHOLD", 0.9),
        asr_safeguard_cooldown_windows=_env_int("MEETINGBRO_ASR_SAFEGUARD_COOLDOWN_WINDOWS", 5),
        asr_early_flush_enabled=bool(profile_settings["asr_early_flush_enabled"]),
        asr_early_flush_min_seconds=float(profile_settings["asr_early_flush_min_seconds"]),
        silence_commit_min_confidence=float(profile_settings["silence_commit_min_confidence"]),
        silence_commit_min_duration_seconds=float(profile_settings["silence_commit_min_duration_seconds"]),
        denoise_enabled=_env_bool("MEETINGBRO_DENOISE_ENABLED", False),
        denoise_strength=_env_float("MEETINGBRO_DENOISE_STRENGTH", 1.1),
        denoise_noise_update_rms_threshold=_env_float(
            "MEETINGBRO_DENOISE_NOISE_UPDATE_RMS_THRESHOLD",
            0.02,
        ),
        audio_conditioning_enabled=_env_bool("MEETINGBRO_AUDIO_CONDITIONING_ENABLED", True),
        audio_conditioning_target_rms=_env_float("MEETINGBRO_AUDIO_CONDITIONING_TARGET_RMS", 0.035),
        audio_conditioning_min_rms=_env_float("MEETINGBRO_AUDIO_CONDITIONING_MIN_RMS", 0.003),
        audio_conditioning_max_gain=_env_float("MEETINGBRO_AUDIO_CONDITIONING_MAX_GAIN", 2.5),
        audio_conditioning_peak_limit=_env_float("MEETINGBRO_AUDIO_CONDITIONING_PEAK_LIMIT", 0.98),
        pre_vad_enabled=bool(profile_settings["pre_vad_enabled"]),
        pre_vad_conditioning_enabled=_env_bool("MEETINGBRO_PRE_VAD_CONDITIONING_ENABLED", True),
        pre_vad_conditioning_target_rms=_env_float("MEETINGBRO_PRE_VAD_CONDITIONING_TARGET_RMS", 0.03),
        pre_vad_conditioning_min_rms=_env_float("MEETINGBRO_PRE_VAD_CONDITIONING_MIN_RMS", 0.001),
        pre_vad_conditioning_max_gain=_env_float("MEETINGBRO_PRE_VAD_CONDITIONING_MAX_GAIN", 4.0),
        pre_vad_threshold=_env_float("MEETINGBRO_PRE_VAD_THRESHOLD", 0.30),
        pre_vad_energy_rms_threshold=_env_float("MEETINGBRO_PRE_VAD_ENERGY_RMS_THRESHOLD", 0.005),
        pre_vad_trailing_silence_seconds=float(profile_settings["pre_vad_trailing_silence_seconds"]),
        pre_vad_adaptive_trailing_silence_enabled=bool(profile_settings["pre_vad_adaptive_trailing_silence_enabled"]),
        pre_vad_adaptive_fast_trailing_silence_seconds=float(profile_settings["pre_vad_adaptive_fast_trailing_silence_seconds"]),
        pre_vad_max_segment_seconds=_env_float("MEETINGBRO_PRE_VAD_MAX_SEGMENT_SECONDS", 8.0),
        rolling_window_seconds=float(profile_settings["rolling_window_seconds"]),
        refinement_interval_seconds=float(profile_settings["refinement_interval_seconds"]),
        refinement_window_seconds=_env_float("MEETINGBRO_REFINEMENT_WINDOW_SECONDS", 0.0),
        summary_tail_seconds=float(profile_settings["summary_tail_seconds"]),
        min_segments_for_refinement=int(profile_settings["min_segments_for_refinement"]),
        rolling_interval_seconds=float(profile_settings["rolling_interval_seconds"]),
        memory_interval_seconds=float(profile_settings["memory_interval_seconds"]),
        cumulative_interval_seconds=float(profile_settings["cumulative_interval_seconds"]),
        min_segments_for_rolling=int(profile_settings["min_segments_for_rolling"]),
        min_segments_for_memory=int(profile_settings["min_segments_for_memory"]),
        min_segments_for_cumulative=int(profile_settings["min_segments_for_cumulative"]),
        weak_speech_rescue_enabled=bool(profile_settings["weak_speech_rescue_enabled"]),
        weak_speech_rescue_rms_min=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MIN", 0.0008),
        weak_speech_rescue_rms_max=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MAX", 0.02),
        weak_speech_rescue_fast_window_seconds=float(profile_settings["weak_speech_rescue_fast_window_seconds"]),
        resource_governor_policy=str(profile_settings["resource_governor_policy"]),
        resource_pressure_rtf_threshold=float(profile_settings["resource_pressure_rtf_threshold"]),
        resource_critical_rtf_threshold=float(profile_settings["resource_critical_rtf_threshold"]),
        resource_pressure_backlog_seconds=float(profile_settings["resource_pressure_backlog_seconds"]),
        resource_critical_backlog_seconds=float(profile_settings["resource_critical_backlog_seconds"]),
        weak_speech_rescue_window_seconds=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_WINDOW_SECONDS", 6.0),
        weak_speech_rescue_cooldown_seconds=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_COOLDOWN_SECONDS", 8.0),
        language_lock_enabled=bool(profile_settings["language_lock_enabled"]),
        live_translation_backfill_limit=int(profile_settings["live_translation_backfill_limit"]),
        live_translation_max_pending=int(profile_settings["live_translation_max_pending"]),
        live_translation_safeguard_max_pending=int(profile_settings["live_translation_safeguard_max_pending"]),
        audio_input_queue_max_seconds=_env_float("MEETINGBRO_AUDIO_INPUT_QUEUE_MAX_SECONDS", 8.0),
        audio_input_queue_warning_seconds=_env_float("MEETINGBRO_AUDIO_INPUT_QUEUE_WARNING_SECONDS", 3.0),
        fast_preview_enabled=bool(profile_settings["fast_preview_enabled"]),
        fast_preview_interval_seconds=_env_float("MEETINGBRO_FAST_PREVIEW_INTERVAL_SECONDS", 0.5),
        fast_preview_window_seconds=_env_float("MEETINGBRO_FAST_PREVIEW_WINDOW_SECONDS", 3.0),
        fast_preview_max_backlog_seconds=_env_float("MEETINGBRO_FAST_PREVIEW_MAX_BACKLOG_SECONDS", 0.5),
        fast_preview_max_asr_realtime_factor=_env_float("MEETINGBRO_FAST_PREVIEW_MAX_ASR_RTF", 0.65),
        fast_preview_min_rms=_env_float("MEETINGBRO_FAST_PREVIEW_MIN_RMS", 0.002),
        qwen_orphan_max_age_seconds=_env_float("MEETINGBRO_QWEN_ORPHAN_MAX_AGE_SECONDS", 2.0),
        qwen_startup_draft_enabled=_env_bool("MEETINGBRO_QWEN_STARTUP_DRAFT_ENABLED", True),
        qwen_startup_draft_window_seconds=_env_float("MEETINGBRO_QWEN_STARTUP_DRAFT_WINDOW_SECONDS", 20.0),
        qwen_startup_draft_grace_seconds=_env_float("MEETINGBRO_QWEN_STARTUP_DRAFT_GRACE_SECONDS", 1.2),
        qwen_targeted_retry_enabled=_env_bool("MEETINGBRO_QWEN_TARGETED_RETRY_ENABLED", True),
        qwen_targeted_retry_margin_seconds=_env_float("MEETINGBRO_QWEN_TARGETED_RETRY_MARGIN_SECONDS", 0.35),
        qwen_targeted_retry_max_audio_seconds=_env_float("MEETINGBRO_QWEN_TARGETED_RETRY_MAX_AUDIO_SECONDS", 5.0),
        preview_stale_tolerance_seconds=_env_float("MEETINGBRO_PREVIEW_STALE_TOLERANCE_SECONDS", 0.30),
        asr_executor_workers=_env_int(
            "MEETINGBRO_ASR_EXECUTOR_WORKERS",
            _recommended_asr_executor_workers(),
        ),
        preview_asr_executor_workers=_env_int(
            "MEETINGBRO_PREVIEW_ASR_EXECUTOR_WORKERS",
            _recommended_preview_asr_executor_workers(),
        ),
        summary_executor_workers=_env_int(
            "MEETINGBRO_SUMMARY_EXECUTOR_WORKERS",
            _recommended_summary_executor_workers(),
        ),
        translation_executor_workers=_env_int(
            "MEETINGBRO_TRANSLATION_EXECUTOR_WORKERS",
            _recommended_translation_executor_workers(),
        ),
    )
    manager = SessionManager(config)
    await manager.start()
    logger.info(
        "session ws accepted meeting_id=%s source=%s profile=%s chunk=%.2fs accum=%.2fs",
        manager.meeting_id,
        source,
        profile_name,
        chunk_seconds,
        config.asr_accumulation_seconds,
    )

    forward_task = asyncio.create_task(_forward_events(manager, ws))
    stop_requested = False

    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "save_note":
                payload = data.get("payload") or {}
                content = (payload.get("content") or "").strip()
                if content or payload.get("source_type") == "bookmark":
                    note = manager.save_note(
                        content=content,
                        source_type=payload.get("source_type"),
                        source_id=payload.get("source_id"),
                    )
                    await ws.send_text(
                        json.dumps(
                            {"type": "note_saved", "payload": note.model_dump(mode="json")}
                        )
                    )
            elif data.get("type") == "update_settings":
                payload = data.get("payload") or {}
                next_summary_language = payload.get("summary_language")
                summary_lang = (
                    next_summary_language
                    if next_summary_language in ("zh", "en", "de")
                    else None
                )
                next_profile = normalize_runtime_profile(payload.get("runtime_profile") or manager._cfg.runtime_profile)
                forced_language = payload.get("forced_language")
                if forced_language == "auto":
                    forced_language = None
                elif forced_language not in (None, "zh", "en", "de"):
                    forced_language = manager._cfg.forced_language
                live_translation_language = payload.get("subtitle_language")
                if next_profile == "summary_only":
                    live_translation_language = None
                if live_translation_language == "off":
                    live_translation_language = None
                elif live_translation_language not in (None, "zh", "en", "de"):
                    live_translation_language = manager._cfg.live_translation_language
                if next_profile == "summary_only":
                    live_translation_language = None
                next_profile_settings = _runtime_settings_from_profile(next_profile)
                next_profile_settings["language_lock_enabled"] = _language_lock_enabled_for_speech(forced_language)
                next_chunk_seconds = _chunk_seconds_for_profile(next_profile)
                if _should_load_preview_asr(
                    profile_settings=next_profile_settings,
                    preview_backend_name=getattr(app.state, "preview_asr_backend_name", "shared"),
                ):
                    await _ensure_preview_asr_loaded(app, profile_name=next_profile)
                next_preview_asr, next_preview_backend_name, next_preview_device = _resolve_preview_runtime(
                    profile_settings=next_profile_settings,
                    preview_asr=app.state.preview_asr,
                    preview_backend_name=app.state.preview_asr_backend_name,
                    preview_device=getattr(app.state, "preview_asr_device", None),
                    formal_device=getattr(app.state, "formal_asr_device", "cpu"),
                )
                next_profile, performance_warning = _enforce_performance_requires_qwen(
                    next_profile,
                    preview_backend_name=next_preview_backend_name,
                )
                if performance_warning is not None:
                    next_profile_settings = _runtime_settings_from_profile(next_profile)
                    next_profile_settings["language_lock_enabled"] = _language_lock_enabled_for_speech(forced_language)
                    next_chunk_seconds = _chunk_seconds_for_profile(next_profile)
                    next_preview_asr, next_preview_backend_name, next_preview_device = _resolve_preview_runtime(
                        profile_settings=next_profile_settings,
                        preview_asr=app.state.preview_asr,
                        preview_backend_name=app.state.preview_asr_backend_name,
                        preview_device=getattr(app.state, "preview_asr_device", None),
                        formal_device=getattr(app.state, "formal_asr_device", "cpu"),
                    )
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "payload": ErrorPayload(
                                    code="performance_requires_qwen",
                                    message=performance_warning,
                                ).model_dump(),
                            }
                        )
                    )
                next_vocabulary_hint = payload.get("vocabulary_hint", manager._cfg.vocabulary_hint)
                if next_vocabulary_hint is not None:
                    next_vocabulary_hint = str(next_vocabulary_hint).strip() or None
                next_source = payload.get("source")
                profile_changed = next_profile != manager._cfg.runtime_profile
                chunk_changed = abs(next_chunk_seconds - manager._cfg.audio_chunk_seconds) > 1e-6
                if next_source and (next_source != source or (profile_changed and chunk_changed)):
                    try:
                        next_audio_source = _build_audio_source(next_source, chunk_seconds=next_chunk_seconds)
                    except Exception as exc:
                        await ws.send_text(
                            json.dumps(
                                {
                                    "type": "error",
                                    "payload": ErrorPayload(code="bad_source", message=str(exc)).model_dump(),
                                }
                            )
                        )
                    else:
                        manager.update_audio_source(
                            next_audio_source,
                            source_name=next_source,
                            chunk_seconds=next_chunk_seconds,
                        )
                        source = next_source
                manager.update_preview_runtime(
                    preview_asr=next_preview_asr,
                    preview_asr_backend_name=next_preview_backend_name,
                    preview_asr_device=next_preview_device,
                    preview_asr_fallback_on_error=_env_bool(
                        "MEETINGBRO_PREVIEW_FALLBACK_ON_ERROR",
                        next_preview_backend_name != "qwen3",
                    ),
                )
                manager.update_runtime_settings(
                    forced_language=forced_language,
                    summary_language=summary_lang,
                    live_translation_language=live_translation_language,
                    vocabulary_hint=next_vocabulary_hint,
                    runtime_profile=next_profile,
                    runtime_settings={
                        **next_profile_settings,
                        "audio_chunk_seconds": next_chunk_seconds,
                    },
                )
            elif data.get("type") == "pause":
                await manager.pause()
            elif data.get("type") == "resume":
                await manager.resume()
            elif data.get("type") == "request_summary":
                payload = data.get("payload") or {}
                summary_type = payload.get("summary_type")
                if summary_type in {"rolling_summary", "cumulative_meeting_summary"}:
                    await manager.request_summary(summary_type)
            elif data.get("type") == "stop":
                stop_requested = True
                break
    except WebSocketDisconnect:
        pass
    finally:
        await manager.stop(graceful=stop_requested)
        forward_task.cancel()
        try:
            await forward_task
        except (asyncio.CancelledError, Exception):
            pass
        try:
            await ws.close()
        except Exception:
            pass


def run() -> None:
    uvicorn.run(
        "meetingbro.main:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
