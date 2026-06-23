from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable, Optional

import numpy as np

from ..asr.base import ASRAdapter, ASRSegment
from ..audio.capture import AudioChunk, AudioSource
from ..audio.mixed import MixedAudioSource
from ..audio.enhancement import AdaptiveNoiseReducer, AudioConditioner
from ..audio.vad import PreVadSegmenter
from ..diarization.base import Diarizer
from ..schemas import (
    ErrorPayload,
    LanguageCode,
    Note,
    SessionStatePayload,
    Speaker,
    SummarySnapshot,
    SummaryType,
    TranscriptPreviewPayload,
    TranscriptSegmentRemovedPayload,
    TranscriptTranslationPayload,
    TranscriptSegment,
)
from ..storage.db import Storage
from ..summarization.base import Summarizer
from ..translation.base import Translator

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    audio_source: AudioSource
    asr: ASRAdapter
    summarizer: Summarizer
    translator: Translator
    storage: Storage
    preview_asr: Optional[ASRAdapter] = None
    preview_asr_backend_name: str = "unknown"
    hardware_profile_label: str = "unknown"
    hardware_summary: Optional[str] = None
    formal_asr_device: str = "cpu"
    preview_asr_device: Optional[str] = None
    compute_gpu_available: bool = False
    preview_asr_fallback_on_error: bool = True
    preview_stale_tolerance_seconds: float = 0.30
    audio_source_name: str = "mic"
    audio_chunk_seconds: float = 0.5
    runtime_profile: str = "balanced"
    diarizer: Optional[Diarizer] = None
    forced_language: Optional[str] = None  # None => auto-detect
    summary_language: LanguageCode = "en"
    live_translation_language: Optional[LanguageCode] = None
    rolling_window_seconds: float = 180.0  # last ~3 minutes for rolling summary input
    rolling_interval_seconds: float = 60.0  # cadence target for rolling refresh
    memory_interval_seconds: float = 120.0  # cadence target for compressed meeting memory refresh
    cumulative_interval_seconds: float = 180.0  # cadence target for cumulative refresh
    summary_tail_seconds: float = 120.0  # recent raw transcript kept beside compressed memory
    refinement_interval_seconds: float = 180.0  # cadence target for clean-conversation snapshots
    refinement_window_seconds: float = 0.0  # <=0 means full conversation so far
    min_segments_for_rolling: int = 1
    min_segments_for_memory: int = 3
    min_segments_for_cumulative: int = 3
    min_segments_for_refinement: int = 2
    asr_accumulation_seconds: float = 2.0  # was 2.5; benchmark showed equal char-count at lower latency
    asr_early_flush_enabled: bool = True
    asr_early_flush_min_seconds: float = 0.8
    silence_commit_min_confidence: float = 0.75
    silence_commit_min_duration_seconds: float = 0.6
    silence_rms_threshold: float = 0.002  # chunks below this RMS are skipped (~-54 dBFS); low enough to pass quiet loopback / soft speech
    # Overlap tail prepended to next ASR window. DEFAULT 0.0 (disabled) — Whisper's
    # internal VAD padding (~400 ms speech_pad_ms) handles boundary words on its own,
    # and any non-zero overlap interacts badly with that padding: VAD pads segment
    # starts backwards into the overlap zone and the filter then drops legitimate
    # speech. Set this >0 only if you have measured a specific boundary-truncation
    # problem the VAD padding does not cover.
    asr_overlap_seconds: float = 0.0
    # Optional static vocabulary hint fed to Whisper as initial_prompt. Use this for
    # proper nouns, names, jargon — NEVER pass running transcript here, that re-creates
    # the conditioning loop hallucination this codebase explicitly defends against.
    vocabulary_hint: Optional[str] = None
    suspicious_segment_no_speech_prob: float = 0.6
    suspicious_segment_avg_logprob: float = -0.9
    suspicious_segment_compression_ratio: float = 2.1
    asr_retry_enabled: bool = True
    asr_safeguard_enabled: bool = True
    asr_safeguard_rtf_threshold: float = 0.9
    asr_safeguard_cooldown_windows: int = 5
    denoise_enabled: bool = False
    denoise_strength: float = 1.1
    denoise_noise_update_rms_threshold: float = 0.02
    audio_conditioning_enabled: bool = True
    audio_conditioning_target_rms: float = 0.035
    audio_conditioning_min_rms: float = 0.003
    audio_conditioning_max_gain: float = 2.5
    audio_conditioning_peak_limit: float = 0.98
    pre_vad_enabled: bool = True
    pre_vad_conditioning_enabled: bool = True
    pre_vad_conditioning_target_rms: float = 0.03
    pre_vad_conditioning_min_rms: float = 0.001
    pre_vad_conditioning_max_gain: float = 4.0
    pre_vad_threshold: float = 0.30
    pre_vad_energy_rms_threshold: float = 0.005
    pre_vad_trailing_silence_seconds: float = 0.45
    pre_vad_adaptive_trailing_silence_enabled: bool = True
    pre_vad_adaptive_fast_trailing_silence_seconds: float = 0.30
    pre_vad_adaptive_max_realtime_factor: float = 0.5
    pre_vad_max_segment_seconds: float = 8.0
    weak_speech_rescue_enabled: bool = True
    weak_speech_rescue_rms_min: float = 0.0008
    weak_speech_rescue_rms_max: float = 0.02
    weak_speech_rescue_fast_rms_max: float = 0.01
    weak_speech_rescue_fast_window_seconds: float = 2.5
    resource_governor_policy: str = "balanced"
    resource_pressure_rtf_threshold: float = 0.90
    resource_critical_rtf_threshold: float = 1.25
    resource_pressure_backlog_seconds: float = 3.0
    resource_critical_backlog_seconds: float = 6.0
    weak_speech_rescue_window_seconds: float = 6.0
    weak_speech_rescue_cooldown_seconds: float = 8.0
    # Default off: mixed-language meetings should not be forced into one language.
    language_lock_enabled: bool = False
    live_translation_backfill_limit: int = 20
    live_translation_max_pending: int = 12
    live_translation_safeguard_max_pending: int = 4
    watchdog_enabled: bool = True
    watchdog_tick_seconds: float = 2.0
    watchdog_no_chunk_warning_seconds: float = 5.0
    watchdog_no_chunk_error_seconds: float = 15.0
    watchdog_all_silent_seconds: float = 45.0
    watchdog_drop_burst_count: int = 50
    watchdog_drop_burst_window_seconds: float = 30.0
    audio_input_queue_max_seconds: float = 8.0
    audio_input_queue_warning_seconds: float = 3.0
    fast_preview_enabled: bool = True
    fast_preview_interval_seconds: float = 0.5
    fast_preview_window_seconds: float = 3.0
    fast_preview_max_backlog_seconds: float = 0.5
    fast_preview_max_asr_realtime_factor: float = 0.65
    fast_preview_min_rms: float = 0.002
    # How long (seconds) a Qwen preview may go uncovered by Whisper before it
    # is promoted live as a formal segment.  This is the "no missed sentences"
    # safety net for chatty meetings with no silence pauses.
    qwen_orphan_max_age_seconds: float = 2.0
    # Softer startup protection: Qwen often emits useful text before Whisper has
    # produced the first formal segment. During the first few seconds, allow one
    # earlier low-quality Qwen draft so the meeting opening is not lost; Whisper
    # still replaces the draft when it later covers the same time span.
    qwen_startup_draft_enabled: bool = True
    qwen_startup_draft_window_seconds: float = 20.0
    qwen_startup_draft_grace_seconds: float = 1.2
    # Before a stranded Qwen preview is promoted as a low-quality draft, try one
    # small Whisper retry on the same time span.  This preserves Qwen's realtime
    # responsiveness while letting Whisper close gaps when it can.
    qwen_targeted_retry_enabled: bool = True
    qwen_targeted_retry_margin_seconds: float = 0.35
    qwen_targeted_retry_max_audio_seconds: float = 5.0
    # Executor sizing is not a user-facing tuning surface. Defaults are chosen
    # automatically in main.py and may be overridden only via hidden env vars.
    asr_executor_workers: int = 1
    preview_asr_executor_workers: int = 1
    summary_executor_workers: int = 1
    translation_executor_workers: int = 1
    filler_filter_enabled: bool = True
    filler_tokens: dict[str, list[str]] = field(default_factory=lambda: {
        "zh": ["嗯", "嗯嗯", "嗯嗯嗯", "呃", "呃呃", "哦", "哦哦", "噢", "唔"],
        "en": ["um", "uh", "er", "ah", "hmm", "mm", "mhm", "mmhm", "mm-hmm"],
        "de": ["äh", "ähm", "hm", "hmm", "öh", "öhm"],
    })


@dataclass
class SessionEvent:
    type: str
    payload: dict


@dataclass
class _State:
    meeting_id: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    last_rolling_at: float = 0.0
    last_rolling_end_time: float = 0.0  # transcript end_time of the last segment included in the previous rolling summary
    latest_rolling_text: Optional[str] = None
    last_memory_at: float = 0.0
    last_cumulative_at: float = 0.0
    last_refinement_at: float = 0.0
    latest_meeting_memory: Optional[str] = None
    latest_cumulative_text: Optional[str] = None
    last_memory_segment_index: int = 0
    memory_in_flight: bool = False
    rolling_in_flight: bool = False
    cumulative_in_flight: bool = False
    refinement_in_flight: bool = False
    elapsed_seconds: float = 0.0
    speakers: dict[str, Speaker] = field(default_factory=dict)  # label -> Speaker
    # Repetition guard: count consecutive emissions with identical text so we can
    # detect Whisper conditioning-loop hallucinations and reset state.
    last_emitted_text: str = ""
    repetition_streak: int = 0
    # Language vote stickiness (active only when forced_language is None / auto mode).
    language_votes: dict[str, int] = field(default_factory=dict)
    locked_language: Optional[str] = None
    language_dissent_streak: int = 0
    pending_segments: list[TranscriptSegment] = field(default_factory=list)
    last_progress_emit_second: int = -1
    retry_windows_total: int = 0
    retry_windows_improved: int = 0
    retry_windows_unchanged: int = 0
    retry_windows_diverged: int = 0
    last_backpressure_elapsed_seconds: Optional[float] = None
    asr_last_audio_seconds: Optional[float] = None
    asr_last_wall_seconds: Optional[float] = None
    asr_realtime_factor: Optional[float] = None
    asr_inflight_start_wall_time: Optional[float] = None
    asr_inflight_audio_seconds: Optional[float] = None
    asr_inflight_phase: Optional[str] = None
    asr_safeguard_reason: Optional[str] = None
    asr_safeguard_cooldown_windows: int = 0
    asr_safeguard_events: int = 0
    resource_pressure_level: str = "normal"
    resource_governor_reason: Optional[str] = None
    resource_governor_skips: int = 0
    weak_rescue_attempts: int = 0
    weak_rescue_emitted: int = 0
    weak_rescue_buffer_seconds: float = 0.0
    translation_backlog_trim_total: int = 0
    filler_filtered_total: int = 0
    audio_input_backlog_seconds: float = 0.0
    audio_input_queue_drop_total: int = 0
    audio_queue_wait_seconds: Optional[float] = None
    fast_preview_attempts: int = 0
    fast_preview_emitted: int = 0
    fast_preview_skipped: int = 0
    fast_preview_last_audio_seconds: Optional[float] = None
    fast_preview_last_wall_seconds: Optional[float] = None
    fast_preview_realtime_factor: Optional[float] = None
    fast_preview_schedule_delay_seconds: Optional[float] = None
    snapshot_concat_wall_seconds: Optional[float] = None
    fast_preview_inflight: bool = False
    fast_preview_segment: Optional[TranscriptSegment] = None
    preview_continued_during_formal: int = 0
    last_emitted_end_time: Optional[float] = None
    preview_stale_suppressed: int = 0
    preview_alignment_compared: int = 0
    preview_alignment_similarity_sum: float = 0.0
    preview_alignment_similarity_last: Optional[float] = None
    preview_unconfirmed_after_formal: int = 0
    preview_unconfirmed_last_text: Optional[str] = None
    # Ring buffer of recent preview segments for alignment diagnostics (memory-only).
    recent_preview_segments: list = field(default_factory=list)
    # Qwen previews that Whisper passed over without coverage, queued for promotion.
    qwen_orphan_queue: list = field(default_factory=list)
    # Latest end_time (session-absolute seconds) covered by a Qwen promotion.
    # Used to prevent Whisper from re-emitting speech already promoted by Qwen.
    qwen_covered_until: Optional[float] = None
    # Map of segment_id → (start_time, end_time) for Qwen-promoted draft segments.
    # When Whisper commits a segment that overlaps a draft, the draft is removed
    # from the DB and a transcript_segment_removed event is emitted so Whisper's
    # higher-quality version takes its place in the frontend.
    qwen_committed_drafts: dict = field(default_factory=dict)
    qwen_startup_draft_promoted: bool = False
    qwen_targeted_retry_attempts: int = 0
    qwen_targeted_retry_recovered: int = 0
    qwen_targeted_retry_failed: int = 0
    qwen_targeted_retry_skipped: int = 0
    qwen_targeted_retry_last_reason: Optional[str] = None
    first_chunk_wall_time: Optional[float] = None
    last_chunk_wall_time: Optional[float] = None
    last_voiced_chunk_wall_time: Optional[float] = None
    watchdog_active_episodes: set[str] = field(default_factory=set)
    watchdog_drop_history: list[tuple[float, int]] = field(default_factory=list)
    preview_candidate_segment: Optional[TranscriptSegment] = None
    preview_candidate_confirmations: int = 0


class SessionManager:
    """Orchestrates one live meeting session.

        Pipeline per audio chunk:
            audio_source -> ASR -> TranscriptSegment persisted & emitted ->
            (on cadence) rolling_summary built from last N seconds of segments ->
            (on cadence) cumulative_meeting_summary built from all segments so far.
        """

    _PREVIEW_CONFIRMATION_WINDOWS = 2
    _STOP_TASK_TIMEOUT_SECONDS = 6.0
    _STOP_SUMMARY_WAIT_SECONDS = 1.5
    _STOP_TRANSLATION_WAIT_SECONDS = 1.0

    def __init__(self, config: SessionConfig) -> None:
        self._cfg = config
        # Dedupe key for surfacing LLM summary failures to the UI once per
        # distinct error (reset when the LLM recovers).
        self._last_summary_error_emitted: Optional[str] = None
        # Same, for live-translation failures (e.g. rejected key while subtitles on).
        self._last_translation_error_emitted: Optional[str] = None
        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue(maxsize=1024)
        self._state = _State(meeting_id=str(uuid.uuid4()))
        self._stopped = asyncio.Event()
        self._pause_gate = asyncio.Event()
        self._pause_gate.set()
        self._task: Optional[asyncio.Task[None]] = None
        self._watchdog_task: Optional[asyncio.Task[None]] = None
        self._summary_tasks: set[asyncio.Task[None]] = set()
        self._translation_tasks: set[asyncio.Task[None]] = set()
        self._translation_task_order: list[asyncio.Task[None]] = []
        self._translation_in_flight: set[tuple[str, str]] = set()
        self._translation_requested_segment_ids: dict[str, str] = {}
        self._translation_workers_by_language: dict[str, asyncio.Task[None]] = {}
        self._pending_speaker_updates: list[Speaker] = []
        self._queue_drop_count: int = 0
        self._audio_drop_total: int = 0
        self._audio_source_generation: int = 0
        # Keep ASR isolated from summarization so periodic summary refreshes do
        # not steal the default executor workers used by the realtime path.
        self._asr_executor = ThreadPoolExecutor(
            max_workers=max(1, self._cfg.asr_executor_workers),
            thread_name_prefix="meetingbro-asr",
        )
        self._preview_asr_executor = (
            ThreadPoolExecutor(
                max_workers=max(1, self._cfg.preview_asr_executor_workers),
                thread_name_prefix="meetingbro-preview-asr",
            )
            if self._cfg.preview_asr is not None
            else None
        )
        self._summary_executor = ThreadPoolExecutor(
            max_workers=max(1, self._cfg.summary_executor_workers),
            thread_name_prefix="meetingbro-summary",
        )
        self._translation_executor = ThreadPoolExecutor(
            max_workers=max(1, self._cfg.translation_executor_workers),
            thread_name_prefix="meetingbro-translation",
        )
        self._preview_asr_disabled = False
        # True when user explicitly requested a normal stop from the UI. False
        # for transport loss / Ctrl+C shutdown where fast exit is preferred.
        self._graceful_stop_requested = True

    @property
    def meeting_id(self) -> str:
        return self._state.meeting_id

    def _active_state_name(self) -> str:
        return "running" if self._pause_gate.is_set() else "paused"

    def _compute_activity_status(self) -> tuple[bool, bool, Optional[str]]:
        formal_device = (self._cfg.formal_asr_device or "cpu").lower()
        preview_device = (self._cfg.preview_asr_device or "cpu").lower()
        formal_active = self._state.asr_inflight_start_wall_time is not None
        preview_active = self._state.fast_preview_inflight
        gpu_active = (formal_active and formal_device == "cuda") or (preview_active and preview_device == "cuda")
        cpu_active = (formal_active and formal_device != "cuda") or (preview_active and preview_device != "cuda")
        labels: list[str] = []
        if formal_active:
            phase = self._state.asr_inflight_phase or "formal"
            labels.append(f"Whisper {phase} on {formal_device}")
        if preview_active:
            backend = self._cfg.preview_asr_backend_name or "preview"
            labels.append(f"{backend} preview on {preview_device}")
        return cpu_active, gpu_active, " ? ".join(labels) if labels else None

    def _session_state_payload(self, *, state: str) -> SessionStatePayload:
        self._prune_background_task_lists()
        compute_cpu_active, compute_gpu_active, compute_activity_label = self._compute_activity_status()
        mixed_microphone_gain = None
        mixed_system_gain = None
        mixed_effective_microphone_gain = None
        mixed_auto_balance_enabled = None
        if isinstance(self._cfg.audio_source, MixedAudioSource):
            mixed_microphone_gain = self._cfg.audio_source.microphone_gain
            mixed_system_gain = self._cfg.audio_source.system_gain
            mixed_effective_microphone_gain = self._cfg.audio_source.effective_microphone_gain
            mixed_auto_balance_enabled = self._cfg.audio_source.auto_balance_enabled
        return SessionStatePayload(
            state=state,
            meeting_id=self._state.meeting_id,
            elapsed_seconds=self._state.elapsed_seconds,
            source=self._cfg.audio_source_name,
            runtime_profile=self._cfg.runtime_profile,
            hardware_profile=self._cfg.hardware_profile_label,
            hardware_summary=self._cfg.hardware_summary,
            compute_cpu_active=compute_cpu_active,
            compute_cpu_available=True,
            compute_cpu_configured=(
                (self._cfg.formal_asr_device or "cpu").lower() != "cuda"
                or (self._cfg.preview_asr_device or "cpu").lower() != "cuda"
            ),
            compute_gpu_active=compute_gpu_active,
            compute_gpu_available=self._cfg.compute_gpu_available,
            compute_gpu_configured=(
                (self._cfg.formal_asr_device or "cpu").lower() == "cuda"
                or (self._cfg.preview_asr_device or "cpu").lower() == "cuda"
            ),
            compute_activity_label=compute_activity_label,
            audio_chunk_seconds=self._cfg.audio_chunk_seconds,
            asr_accumulation_seconds=self._cfg.asr_accumulation_seconds,
            language_lock_enabled=self._cfg.language_lock_enabled,
            live_translation_language=self._cfg.live_translation_language,
            retry_windows_total=self._state.retry_windows_total,
            retry_windows_improved=self._state.retry_windows_improved,
            retry_windows_unchanged=self._state.retry_windows_unchanged,
            retry_windows_diverged=self._state.retry_windows_diverged,
            last_backpressure_elapsed_seconds=self._state.last_backpressure_elapsed_seconds,
            asr_last_audio_seconds=self._state.asr_last_audio_seconds,
            asr_last_wall_seconds=self._state.asr_last_wall_seconds,
            asr_realtime_factor=self._state.asr_realtime_factor,
            asr_safeguard_active=self._asr_safeguard_active(),
            asr_safeguard_reason=self._state.asr_safeguard_reason,
            asr_safeguard_events=self._state.asr_safeguard_events,
            resource_pressure_level=self._state.resource_pressure_level,
            resource_governor_policy=self._cfg.resource_governor_policy,
            resource_governor_reason=self._state.resource_governor_reason,
            resource_governor_skips=self._state.resource_governor_skips,
            weak_rescue_attempts=self._state.weak_rescue_attempts,
            weak_rescue_emitted=self._state.weak_rescue_emitted,
            weak_rescue_buffer_seconds=self._state.weak_rescue_buffer_seconds,
            summary_pending_count=sum(1 for task in self._summary_tasks if not task.done()),
            translation_pending_count=sum(1 for task in self._translation_task_order if not task.done()),
            translation_backlog_trim_total=self._state.translation_backlog_trim_total,
            audio_drop_total=self._audio_drop_total,
            audio_input_backlog_seconds=self._state.audio_input_backlog_seconds,
            audio_input_queue_drop_total=self._state.audio_input_queue_drop_total,
            audio_queue_wait_seconds=self._state.audio_queue_wait_seconds,
            event_queue_depth=self._event_queue.qsize(),
            fast_preview_enabled=self._cfg.fast_preview_enabled,
            fast_preview_attempts=self._state.fast_preview_attempts,
            fast_preview_emitted=self._state.fast_preview_emitted,
            fast_preview_skipped=self._state.fast_preview_skipped,
            fast_preview_last_audio_seconds=self._state.fast_preview_last_audio_seconds,
            fast_preview_last_wall_seconds=self._state.fast_preview_last_wall_seconds,
            fast_preview_realtime_factor=self._state.fast_preview_realtime_factor,
            fast_preview_schedule_delay_seconds=self._state.fast_preview_schedule_delay_seconds,
            snapshot_concat_wall_seconds=self._state.snapshot_concat_wall_seconds,
            preview_continued_during_formal=self._state.preview_continued_during_formal,
            preview_stale_suppressed=self._state.preview_stale_suppressed,
            preview_alignment_compared=self._state.preview_alignment_compared,
            preview_alignment_similarity_avg=(
                round(
                    self._state.preview_alignment_similarity_sum
                    / self._state.preview_alignment_compared,
                    4,
                )
                if self._state.preview_alignment_compared > 0
                else None
            ),
            preview_alignment_similarity_last=self._state.preview_alignment_similarity_last,
            preview_unconfirmed_after_formal=self._state.preview_unconfirmed_after_formal,
            preview_unconfirmed_last_text=self._state.preview_unconfirmed_last_text,
            qwen_targeted_retry_attempts=self._state.qwen_targeted_retry_attempts,
            qwen_targeted_retry_recovered=self._state.qwen_targeted_retry_recovered,
            qwen_targeted_retry_failed=self._state.qwen_targeted_retry_failed,
            qwen_targeted_retry_skipped=self._state.qwen_targeted_retry_skipped,
            qwen_targeted_retry_last_reason=self._state.qwen_targeted_retry_last_reason,
            mixed_microphone_gain=mixed_microphone_gain,
            mixed_system_gain=mixed_system_gain,
            mixed_effective_microphone_gain=mixed_effective_microphone_gain,
            mixed_auto_balance_enabled=mixed_auto_balance_enabled,
        )

    async def _emit_compute_activity_state(self) -> None:
        await self._emit(
            "session_state",
            self._session_state_payload(state=self._active_state_name()),
        )

    def update_runtime_settings(
        self,
        *,
        forced_language: Optional[str],
        summary_language: Optional[LanguageCode] = None,
        live_translation_language: Optional[LanguageCode] = None,
        vocabulary_hint: Optional[str] = None,
        runtime_profile: Optional[str] = None,
        runtime_settings: Optional[dict[str, object]] = None,
    ) -> None:
        runtime_changed = False
        if runtime_profile is not None and runtime_profile != self._cfg.runtime_profile:
            self._cfg.runtime_profile = runtime_profile
            runtime_changed = True

        if runtime_settings:
            language_lock_before = self._cfg.language_lock_enabled
            for key, value in runtime_settings.items():
                if hasattr(self._cfg, key):
                    current = getattr(self._cfg, key)
                    if current != value:
                        setattr(self._cfg, key, value)
                        runtime_changed = True
            if self._cfg.language_lock_enabled != language_lock_before:
                self._state.locked_language = None
                self._state.language_votes.clear()
                self._state.language_dissent_streak = 0

        if runtime_changed:
            logger.info(
                "updated runtime_profile=%s chunk=%.2fs accum=%.2fs language_lock=%s for meeting_id=%s",
                self._cfg.runtime_profile,
                self._cfg.audio_chunk_seconds,
                self._cfg.asr_accumulation_seconds,
                self._cfg.language_lock_enabled,
                self._state.meeting_id,
            )

        forced_changed = forced_language != self._cfg.forced_language
        if forced_changed:
            self._cfg.forced_language = forced_language
            self._state.locked_language = None
            self._state.language_votes.clear()
            self._state.language_dissent_streak = 0
            logger.info("updated forced_language=%s for meeting_id=%s", forced_language or "auto", self._state.meeting_id)

        if summary_language is not None and summary_language != self._cfg.summary_language:
            self._cfg.summary_language = summary_language
            self._cfg.storage.update_meeting_summary_language(
                self._state.meeting_id,
                preferred_summary_language=summary_language,
            )
            logger.info("updated summary_language=%s for meeting_id=%s", summary_language, self._state.meeting_id)

        translation_changed = live_translation_language != self._cfg.live_translation_language
        if translation_changed:
            self._cfg.live_translation_language = live_translation_language
            logger.info(
                "updated live_translation_language=%s for meeting_id=%s",
                live_translation_language or "off",
                self._state.meeting_id,
            )

        if vocabulary_hint != self._cfg.vocabulary_hint:
            self._cfg.vocabulary_hint = vocabulary_hint
            logger.info(
                "updated vocabulary_hint=%s for meeting_id=%s",
                "set" if vocabulary_hint else "cleared",
                self._state.meeting_id,
            )
            if live_translation_language is not None:
                self._backfill_live_translations()

        if runtime_changed or translation_changed:
            asyncio.create_task(self._emit("session_state", self._session_state_payload(state=self._active_state_name())))

    def update_preview_runtime(
        self,
        *,
        preview_asr: Optional[ASRAdapter],
        preview_asr_backend_name: str,
        preview_asr_device: Optional[str],
        preview_asr_fallback_on_error: bool,
    ) -> None:
        lane_changed = (
            self._cfg.preview_asr is not preview_asr
            or self._cfg.preview_asr_backend_name != preview_asr_backend_name
            or self._cfg.preview_asr_device != preview_asr_device
            or self._cfg.preview_asr_fallback_on_error != preview_asr_fallback_on_error
        )
        if not lane_changed:
            return

        previous_executor = self._preview_asr_executor
        self._cfg.preview_asr = preview_asr
        self._cfg.preview_asr_backend_name = preview_asr_backend_name
        self._cfg.preview_asr_device = preview_asr_device
        self._cfg.preview_asr_fallback_on_error = preview_asr_fallback_on_error
        self._preview_asr_disabled = False
        self._state.fast_preview_segment = None
        self._clear_preview_candidate()

        if preview_asr is not None:
            self._preview_asr_executor = ThreadPoolExecutor(
                max_workers=max(1, self._cfg.preview_asr_executor_workers),
                thread_name_prefix="meetingbro-preview-asr",
            )
        else:
            self._preview_asr_executor = None

        if previous_executor is not None:
            previous_executor.shutdown(wait=False, cancel_futures=True)

        logger.info(
            "updated preview runtime backend=%s dedicated=%s device=%s meeting_id=%s",
            self._cfg.preview_asr_backend_name,
            self._cfg.preview_asr is not None,
            self._cfg.preview_asr_device or "cpu",
            self._state.meeting_id,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        asyncio.create_task(self._emit_transcript_preview())
        asyncio.create_task(self._emit("session_state", self._session_state_payload(state=self._active_state_name())))

    def update_audio_source(
        self,
        audio_source: AudioSource,
        *,
        source_name: str,
        chunk_seconds: Optional[float] = None,
    ) -> None:
        previous_source = self._cfg.audio_source
        self._cfg.audio_source = audio_source
        self._cfg.audio_source_name = source_name
        if chunk_seconds is not None:
            self._cfg.audio_chunk_seconds = chunk_seconds
        self._audio_source_generation += 1
        self._reset_watchdog_tracking()
        self._clear_preview_candidate()
        self._state.fast_preview_segment = None
        self._state.pending_segments.clear()
        self._state.last_emitted_text = ""
        self._state.repetition_streak = 0
        logger.info("updated audio source=%s for meeting_id=%s", source_name, self._state.meeting_id)
        asyncio.create_task(self._aclose_specific_source(previous_source))
        asyncio.create_task(self._emit("session_state", self._session_state_payload(state=self._active_state_name())))

    async def pause(self) -> None:
        if self._stopped.is_set() or not self._pause_gate.is_set():
            return
        self._pause_gate.clear()
        self._state.pending_segments.clear()
        self._state.fast_preview_segment = None
        self._clear_preview_candidate()
        await self._emit_transcript_preview()
        await self._emit(
            "session_state",
            self._session_state_payload(state="paused"),
        )

    async def resume(self) -> None:
        if self._stopped.is_set() or self._pause_gate.is_set():
            return
        self._pause_gate.set()
        await self._emit(
            "session_state",
            self._session_state_payload(state="running"),
        )

    async def request_summary(self, summary_type: str) -> None:
        if self._stopped.is_set():
            return
        if summary_type == "rolling_summary":
            await self._emit_rolling(force=True)
            return
        if summary_type == "cumulative_meeting_summary":
            await self._emit_memory(force=True)
            await self._emit_cumulative(force=True)
            return
        if summary_type == "refined_transcript":
            await self._emit_refined_transcript(force=True)
            return

    async def events(self) -> AsyncIterator[SessionEvent]:
        while not (self._stopped.is_set() and self._event_queue.empty()):
            try:
                ev = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            yield ev

    # Event types that must never be dropped even when the queue is full.
    _CRITICAL_EVENT_TYPES = frozenset({
        "summary_snapshot", "error", "session_state", "speaker_update",
        "transcript_segment", "transcript_segment_removed",
    })

    def _drop_one_queued_noncritical_event(self) -> bool:
        """Make room for a critical event without losing committed transcript.

        If the queue is saturated by preview/diagnostic traffic, drop the oldest
        non-critical event and keep the relative order of everything else.
        """
        buffered: list[SessionEvent] = []
        dropped: Optional[SessionEvent] = None
        while True:
            try:
                queued = self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if dropped is None and queued.type not in self._CRITICAL_EVENT_TYPES:
                dropped = queued
                continue
            buffered.append(queued)
        for queued in buffered:
            self._event_queue.put_nowait(queued)
        if dropped is None:
            return False
        self._queue_drop_count += 1
        logger.debug(
            "event queue full — dropped queued %s to make room for critical event (total drops: %d)",
            dropped.type,
            self._queue_drop_count,
        )
        return True

    async def _emit(self, event_type: str, payload_model) -> None:
        payload = (
            payload_model.model_dump(mode="json")
            if hasattr(payload_model, "model_dump")
            else dict(payload_model)
        )
        event = SessionEvent(type=event_type, payload=payload)
        if self._event_queue.full():
            if event_type in self._CRITICAL_EVENT_TYPES:
                self._drop_one_queued_noncritical_event()
            else:
                # Drop low-priority events (e.g. preview updates) to prevent OOM.
                self._queue_drop_count += 1
                logger.debug("event queue full — dropping %s (total drops: %d)", event_type, self._queue_drop_count)
                if self._queue_drop_count % 50 == 0:
                    drop_notice = ErrorPayload(
                        code="event_queue_drop",
                        message=f"{self._queue_drop_count} events dropped (queue full); consumer may be too slow",
                    )
                    try:
                        self._event_queue.put_nowait(
                            SessionEvent(type="error", payload=drop_notice.model_dump(mode="json"))
                        )
                    except asyncio.QueueFull:
                        pass  # critical slot also full; silent skip
                return
        await self._event_queue.put(event)

    def save_note(self, content: str, source_type: Optional[str] = None, source_id: Optional[str] = None) -> Note:
        note = Note(
            id=str(uuid.uuid4()),
            meeting_id=self._state.meeting_id,
            content=content,
            source_type=source_type,
            source_id=source_id,
            time_seconds=self._state.elapsed_seconds if source_type == "bookmark" else None,
            created_at=datetime.now(tz=timezone.utc),
        )
        self._cfg.storage.insert_note(note)
        return note

    async def start(self) -> None:
        self._cfg.storage.create_meeting(
            self._state.meeting_id,
            preferred_summary_language=self._cfg.summary_language,
        )
        await self._emit(
            "session_state",
            self._session_state_payload(state="running"),
        )
        self._task = asyncio.create_task(self._run())
        self._watchdog_task = asyncio.create_task(self._audio_watchdog())

    async def stop(self, *, graceful: bool = True) -> None:
        self._graceful_stop_requested = graceful
        self._stopped.set()
        # Fire aclose() without blocking — for mic/loopback it just sets a threading.Event,
        # so it returns almost instantly.  We don't await here so that if it ever blocks
        # the task timeout below still fires at 15 s.
        asyncio.create_task(self._aclose_source())
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await asyncio.wait_for(self._watchdog_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self._STOP_TASK_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning(
                    "session task did not finish within %.1f s (likely blocked in ASR executor) — cancelling",
                    self._STOP_TASK_TIMEOUT_SECONDS,
                )
                self._task.cancel()
        await self._drain_background_tasks(
            self._summary_tasks,
            timeout=self._STOP_SUMMARY_WAIT_SECONDS,
            label="summary",
        )
        await self._drain_background_tasks(
            self._translation_tasks,
            timeout=self._STOP_TRANSLATION_WAIT_SECONDS,
            label="translation",
        )
        self._cfg.storage.end_meeting(self._state.meeting_id)
        self._shutdown_executors()
        await self._emit(
            "session_state",
            self._session_state_payload(state="ended"),
        )

    async def _drain_background_tasks(
        self,
        tasks: set[asyncio.Task[None]],
        *,
        timeout: float,
        label: str,
    ) -> None:
        """Wait briefly for background tasks, then cancel leftovers.

        Ctrl+C should stop quickly; we keep a short grace period for tasks that
        are about to finish and then cancel the rest.
        """
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        if pending:
            logger.info(
                "stop: cancelling %d pending %s task(s) after %.1fs grace",
                len(pending),
                label,
                timeout,
            )
            for task in pending:
                task.cancel()
        # Keep only tasks still running; callbacks will prune as they complete.
        tasks.difference_update(done)

    async def _aclose_source(self) -> None:
        try:
            await self._cfg.audio_source.aclose()
        except Exception:
            logger.exception("audio_source.aclose failed")

    async def _aclose_specific_source(self, source: AudioSource) -> None:
        try:
            await source.aclose()
        except Exception:
            logger.exception("audio_source.aclose failed during source switch")

    def _shutdown_executors(self) -> None:
        self._asr_executor.shutdown(wait=False, cancel_futures=True)
        if self._preview_asr_executor is not None:
            self._preview_asr_executor.shutdown(wait=False, cancel_futures=True)
        self._summary_executor.shutdown(wait=False, cancel_futures=True)
        self._translation_executor.shutdown(wait=False, cancel_futures=True)

    def _has_dedicated_preview_asr(self) -> bool:
        return (
            self._cfg.preview_asr is not None
            and self._preview_asr_executor is not None
            and not self._preview_asr_disabled
        )

    def _can_continue_preview_during_formal(self) -> bool:
        """Allow only the isolated Qwen preview lane to keep updating during formal work."""
        return (
            self._cfg.preview_asr_backend_name == "qwen3"
            and self._has_dedicated_preview_asr()
        )

    def _disable_dedicated_preview_asr(self, exc: Exception) -> None:
        if self._preview_asr_disabled:
            return
        self._preview_asr_disabled = True
        fallback_note = (
            "falling back to shared preview path"
            if self._cfg.preview_asr_fallback_on_error
            else "skipping preview until session restart"
        )
        logger.warning(
            "dedicated preview ASR unavailable, %s: %s",
            fallback_note,
            exc,
        )

    def _reset_watchdog_tracking(self) -> None:
        self._state.first_chunk_wall_time = None
        self._state.last_chunk_wall_time = None
        self._state.last_voiced_chunk_wall_time = None
        self._state.watchdog_active_episodes.clear()
        self._state.watchdog_drop_history.clear()

    def _clear_preview_candidate(self) -> None:
        self._state.preview_candidate_segment = None
        self._state.preview_candidate_confirmations = 0

    def _preview_text_key(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def _preview_candidate_matches(self, left: TranscriptSegment, right: TranscriptSegment) -> bool:
        if left.original_language != right.original_language:
            return False
        if abs(left.start_time - right.start_time) > 1.0:
            return False
        if right.end_time + 0.25 < left.end_time:
            return False
        left_key = self._preview_text_key(left.text)
        right_key = self._preview_text_key(right.text)
        if not left_key or not right_key:
            return False
        return (
            left_key == right_key
            or left_key.startswith(right_key)
            or right_key.startswith(left_key)
        )

    def _merge_preview_candidate(self, left: TranscriptSegment, right: TranscriptSegment) -> TranscriptSegment:
        merged_text = right.text if len(self._preview_text_key(right.text)) >= len(self._preview_text_key(left.text)) else left.text
        return TranscriptSegment(
            id=left.id,
            meeting_id=left.meeting_id,
            start_time=min(left.start_time, right.start_time),
            end_time=max(left.end_time, right.end_time),
            text=merged_text,
            original_language=right.original_language,
            speaker_id=right.speaker_id or left.speaker_id,
            confidence=max(left.confidence, right.confidence),
            quality=self._merge_quality(left.quality, right.quality),
            translations=dict(left.translations),
            created_at=right.created_at,
            emitted_at_elapsed_seconds=right.emitted_at_elapsed_seconds,
        )

    def _compute_quality(self, asr_seg: ASRSegment) -> str:
        suspicious_count = int((asr_seg.no_speech_prob or 0.0) >= self._cfg.suspicious_segment_no_speech_prob)
        suspicious_count += int(asr_seg.avg_logprob is not None and asr_seg.avg_logprob <= self._cfg.suspicious_segment_avg_logprob)
        suspicious_count += int((asr_seg.compression_ratio or 0.0) >= self._cfg.suspicious_segment_compression_ratio)
        if suspicious_count >= 3:
            return "low"
        if suspicious_count >= 1:
            return "uncertain"
        return "ok"

    def _merge_quality(self, left: str, right: str) -> str:
        rank = {"ok": 0, "uncertain": 1, "low": 2}
        return left if rank[left] >= rank[right] else right

    def _is_redundant_preview_candidate(self, seg: TranscriptSegment) -> bool:
        if not self._state.last_emitted_text:
            return False
        if self._preview_text_key(seg.text) != self._preview_text_key(self._state.last_emitted_text):
            return False
        last_end = self._state.last_emitted_end_time
        if last_end is None:
            return False
        return seg.start_time <= last_end + 1.0

    def _is_pure_filler(self, text: str, language: str) -> bool:
        if not self._cfg.filler_filter_enabled:
            return False
        if not text:
            return False
        normalized = text.strip().lower()
        for ch in '.,!?;:。，！？、；：…—–"\'(){}[]':
            normalized = normalized.replace(ch, "")
        normalized = normalized.strip()
        if not normalized:
            return True
        return normalized in self._cfg.filler_tokens.get(language, [])

    async def _queue_preview_candidate(self, seg: TranscriptSegment) -> None:
        self._state.fast_preview_segment = None
        self._state.preview_candidate_segment = seg
        self._state.preview_candidate_confirmations = 1
        await self._emit_transcript_preview(seg)

    async def _flush_preview_candidate(self) -> None:
        preview = self._state.preview_candidate_segment
        if preview is None:
            return
        self._clear_preview_candidate()
        await self._persist_and_emit_segment(preview)
        await self._emit_transcript_preview()

    def _should_commit_on_silence(self, seg: TranscriptSegment) -> bool:
        duration = max(0.0, seg.end_time - seg.start_time)
        if seg.confidence < self._cfg.silence_commit_min_confidence:
            return False
        if duration < self._cfg.silence_commit_min_duration_seconds:
            return False
        return True

    async def _flush_silence_boundary_segments(self) -> None:
        while self._state.pending_segments:
            oldest = self._state.pending_segments[0]
            if not self._should_commit_on_silence(oldest):
                break
            await self._emit_oldest_pending_segment()

        preview = self._state.preview_candidate_segment
        if preview is not None and self._should_commit_on_silence(preview):
            await self._flush_preview_candidate()

        await self._drain_qwen_orphan_queue()

    @staticmethod
    def _format_vocabulary_prompt(hint: Optional[str]) -> str:
        """Format user vocabulary terms into a glossary-style Whisper prompt.

        The vocabulary dock collects a separator-delimited list of names,
        acronyms, and project terms. Presenting them as a short glossary sentence
        biases Whisper toward the spellings far more effectively than the raw
        delimited string (measured on the proper-noun benchmark fixture: WER
        0.19 -> 0.095, keyword recall 3/5 -> 5/5). If the hint is already free
        text with no separators it is passed through unchanged.
        """
        if not hint:
            return ""
        terms = [t.strip() for t in re.split(r"[\n,，;；、]+", hint) if t.strip()]
        if not terms:
            return ""
        if len(terms) == 1:
            return terms[0]
        return "Glossary: " + ", ".join(terms) + "."

    def _build_whisper_prompt(self) -> Optional[str]:
        """Combine static vocabulary hint with recent Qwen preview context.

        Qwen's current preview is passed as Whisper's initial_prompt so Whisper
        is biased toward the vocabulary and phrasing already heard — the main
        "1+1>2" synergy between the two engines.  The combined string is capped
        at 200 characters to stay well within Whisper's ~224-token prompt budget.

        IMPORTANT: only the *current* rolling preview is included, never an
        accumulating running transcript.  Accumulating context re-creates the
        conditioning-loop hallucination this codebase explicitly defends against.
        """
        base = self._format_vocabulary_prompt(self._cfg.vocabulary_hint)
        if self._has_dedicated_preview_asr():
            preview = self._state.fast_preview_segment
            if preview is not None:
                qwen_ctx = preview.text.strip()
                if qwen_ctx:
                    combined = f"{base} {qwen_ctx}".strip() if base else qwen_ctx
                    return combined[:200]
        return base or None

    def _merge_overlapping_qwen_previews(
        self,
        previews: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        """Merge overlapping rolling Qwen previews before draft promotion.

        Qwen preview windows overlap heavily. Keeping only the newest window can
        lose the beginning of a meeting; keeping every window creates duplicate
        draft transcript rows. This preserves the earliest covered time span
        while using the newest/most-contextual text as the draft wording.
        """

        if not previews:
            return []
        ordered = sorted(previews, key=lambda s: (s.start_time, s.end_time))
        groups: list[list[TranscriptSegment]] = []
        current: list[TranscriptSegment] = []
        current_end = 0.0
        for preview in ordered:
            if not current or preview.start_time <= current_end + 0.05:
                current.append(preview)
                current_end = max(current_end, preview.end_time)
                continue
            groups.append(current)
            current = [preview]
            current_end = preview.end_time
        if current:
            groups.append(current)

        merged: list[TranscriptSegment] = []
        for group in groups:
            text_source = max(
                group,
                key=lambda s: (s.end_time, len(self._preview_text_key(s.text))),
            )
            merged.append(
                TranscriptSegment(
                    id=text_source.id,
                    meeting_id=text_source.meeting_id,
                    start_time=min(s.start_time for s in group),
                    end_time=max(s.end_time for s in group),
                    text=text_source.text,
                    original_language=text_source.original_language,
                    speaker_id=text_source.speaker_id,
                    confidence=text_source.confidence,
                    quality=text_source.quality,
                    translations=dict(text_source.translations),
                    created_at=text_source.created_at,
                    emitted_at_elapsed_seconds=text_source.emitted_at_elapsed_seconds,
                )
            )
        return merged

    async def _promote_time_stranded_previews(
        self,
        whisper_retry: Optional[Callable[[TranscriptSegment], Awaitable[bool]]] = None,
    ) -> None:
        """Live "no missed sentences" safety net: promote Qwen previews that are
        older than ``qwen_orphan_max_age_seconds`` with no formal Whisper coverage.

        Called from fast_preview_loop on every tick so stranded previews surface
        within one interval of going stale, even in chatty meetings without
        silence pauses (which is when ``_flush_silence_boundary_segments`` would
        otherwise have triggered the drain).
        """
        if not self._has_dedicated_preview_asr():
            return
        if not self._state.recent_preview_segments:
            return

        now = self._state.elapsed_seconds
        startup_active = (
            self._cfg.qwen_startup_draft_enabled
            and not self._state.qwen_startup_draft_promoted
            and self._state.last_emitted_end_time is None
            and now <= self._cfg.qwen_startup_draft_window_seconds
        )
        max_age = (
            self._cfg.qwen_startup_draft_grace_seconds
            if startup_active
            else self._cfg.qwen_orphan_max_age_seconds
        )
        covered_until = max(
            self._state.last_emitted_end_time or 0.0,
            self._state.qwen_covered_until or 0.0,
        )

        to_promote = []
        remaining = []
        for preview in self._state.recent_preview_segments:
            if (now - preview.end_time) >= max_age and preview.end_time > covered_until:
                to_promote.append(preview)
            else:
                remaining.append(preview)

        if not to_promote:
            return

        self._state.recent_preview_segments = remaining

        deduped = self._merge_overlapping_qwen_previews(to_promote)
        if startup_active and deduped:
            # One startup draft closes the opening gap; normal orphan promotion
            # handles later uncovered Qwen previews.
            deduped = deduped[:1]

        for orphan in deduped:
            text = orphan.text.strip()
            if not text:
                continue
            if len(re.sub(r"\s+", "", text)) < 2:
                continue
            if text == self._state.last_emitted_text:
                continue
            covered_until = max(
                self._state.last_emitted_end_time or 0.0,
                self._state.qwen_covered_until or 0.0,
            )
            if orphan.end_time <= covered_until + 0.1:
                continue

            if whisper_retry is not None:
                recovered = await whisper_retry(orphan)
                if recovered:
                    if startup_active:
                        self._state.qwen_startup_draft_promoted = True
                    continue

            promoted = TranscriptSegment(
                id=str(uuid.uuid4()),
                meeting_id=orphan.meeting_id,
                start_time=orphan.start_time,
                end_time=orphan.end_time,
                text=orphan.text,
                original_language=orphan.original_language,
                speaker_id=orphan.speaker_id,
                confidence=orphan.confidence,
                quality="low",
                created_at=datetime.now(tz=timezone.utc),
                emitted_at_elapsed_seconds=orphan.emitted_at_elapsed_seconds,
            )
            logger.info(
                "qwen_stranded_promoted [%.2f-%.2f] age=%.1fs text=%r",
                orphan.start_time,
                orphan.end_time,
                now - orphan.end_time,
                text[:80],
            )
            await self._persist_and_emit_segment(promoted, is_qwen_draft=True)
            self._state.qwen_committed_drafts[promoted.id] = (promoted.start_time, promoted.end_time)
            if startup_active:
                self._state.qwen_startup_draft_promoted = True
            self._state.qwen_covered_until = max(
                self._state.qwen_covered_until or 0.0,
                orphan.end_time,
            )

    async def _drain_qwen_orphan_queue(self) -> None:
        """Promote orphaned Qwen previews as low-quality formal segments.

        Runs at silence boundaries and session end.  Only active when a
        dedicated preview ASR (Qwen) is configured, so it never fires when
        Whisper is both the preview and formal engine (same-model duplication
        would be wrong).
        """
        if not self._state.qwen_orphan_queue:
            return

        queue = sorted(self._state.qwen_orphan_queue, key=lambda s: s.end_time)
        self._state.qwen_orphan_queue.clear()
        deduped = self._merge_overlapping_qwen_previews(queue)

        for orphan in deduped:
            text = orphan.text.strip()
            if not text:
                continue
            dense = re.sub(r"\s+", "", text)
            # Skip trivially short non-CJK (single letter, stray token).
            if len(dense) < 2:
                continue
            if text == self._state.last_emitted_text:
                continue
            # Don't reach back before the formal timeline — only promote if
            # the gap wasn't already closed by a formal segment committed after
            # the orphan was queued.
            if (
                self._state.last_emitted_end_time is not None
                and orphan.end_time <= self._state.last_emitted_end_time - 0.1
            ):
                continue

            promoted = TranscriptSegment(
                id=str(uuid.uuid4()),
                meeting_id=orphan.meeting_id,
                start_time=orphan.start_time,
                end_time=orphan.end_time,
                text=orphan.text,
                original_language=orphan.original_language,
                speaker_id=orphan.speaker_id,
                confidence=orphan.confidence,
                quality="low",
                created_at=datetime.now(tz=timezone.utc),
                emitted_at_elapsed_seconds=orphan.emitted_at_elapsed_seconds,
            )
            logger.info(
                "qwen_orphan_promoted [%.2f-%.2f] text=%r",
                orphan.start_time,
                orphan.end_time,
                text[:80],
            )
            await self._persist_and_emit_segment(promoted, is_qwen_draft=True)
            self._state.qwen_committed_drafts[promoted.id] = (promoted.start_time, promoted.end_time)
            self._state.qwen_covered_until = max(
                self._state.qwen_covered_until or 0.0,
                orphan.end_time,
            )

    async def _reconcile_preview_candidate(
        self,
        candidate_segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        preview = self._state.preview_candidate_segment
        if preview is None:
            return candidate_segments
        if not candidate_segments:
            return candidate_segments

        first = candidate_segments[0]
        if self._preview_candidate_matches(preview, first):
            merged = self._merge_preview_candidate(preview, first)
            self._state.preview_candidate_segment = merged
            self._state.preview_candidate_confirmations += 1
            if self._state.preview_candidate_confirmations >= self._PREVIEW_CONFIRMATION_WINDOWS:
                self._clear_preview_candidate()
                await self._persist_and_emit_segment(merged)
                await self._emit_transcript_preview()
            else:
                await self._emit_transcript_preview(merged)
            return candidate_segments[1:]

        await self._flush_preview_candidate()
        return candidate_segments

    async def _emit_watchdog_event(
        self,
        *,
        code: str,
        message: str,
        status: str,
        recovered_code: Optional[str] = None,
    ) -> None:
        payload = {
            "code": code,
            "message": f"[{status}] {message}",
        }
        if recovered_code is not None:
            payload["recovered_code"] = recovered_code
        await self._emit("error", payload)

    async def _arm_watchdog_episode(self, *, code: str, message: str, status: str) -> None:
        if code in self._state.watchdog_active_episodes:
            return
        self._state.watchdog_active_episodes.add(code)
        await self._emit_watchdog_event(code=code, message=message, status=status)

    async def _recover_watchdog_episode(self, code: str, message: str) -> None:
        if code not in self._state.watchdog_active_episodes:
            return
        self._state.watchdog_active_episodes.remove(code)
        await self._emit_watchdog_event(
            code="audio_recovered",
            message=message,
            status="info",
            recovered_code=code,
        )

    def _clear_watchdog_episode(self, code: str) -> None:
        self._state.watchdog_active_episodes.discard(code)

    def _record_audio_drops(self, now: float, drops: int) -> int:
        cutoff = now - self._cfg.watchdog_drop_burst_window_seconds
        self._state.watchdog_drop_history = [
            item for item in self._state.watchdog_drop_history if item[0] >= cutoff
        ]
        if drops > 0:
            prev_bucket = self._audio_drop_total // 10
            self._audio_drop_total += drops
            self._state.watchdog_drop_history.append((now, drops))
            return prev_bucket
        return self._audio_drop_total // 10

    def _active_transcript_item_count(self) -> int:
        return (
            len(self._state.segments)
            + len(self._state.pending_segments)
            + (1 if self._state.preview_candidate_segment is not None else 0)
        )

    async def _check_audio_health(self) -> None:
        now = time.monotonic()
        source = self._cfg.audio_source
        drops = source.drain_drops()
        prev_bucket = self._record_audio_drops(now, drops)
        if drops > 0 and self._audio_drop_total // 10 > prev_bucket:
            await self._emit(
                "error",
                ErrorPayload(
                    code="audio_drop",
                    message=f"audio queue has dropped {self._audio_drop_total} chunks total (source too fast for ASR)",
                ),
            )

        # audio_drop and audio_drops_sustained are intentionally separate signals — do not collapse
        drop_total = sum(count for _, count in self._state.watchdog_drop_history)
        if drop_total > self._cfg.watchdog_drop_burst_count:
            await self._arm_watchdog_episode(
                code="audio_drops_sustained",
                status="error",
                message=(
                    "audio queue dropped "
                    f"{drop_total} chunks in the last {self._cfg.watchdog_drop_burst_window_seconds:.0f}s"
                ),
            )
        else:
            await self._recover_watchdog_episode(
                "audio_drops_sustained",
                "audio drop rate returned to normal",
            )

        asr_inflight_start = self._state.asr_inflight_start_wall_time
        if asr_inflight_start is not None:
            asr_busy_seconds = now - asr_inflight_start
            if asr_busy_seconds >= self._cfg.watchdog_no_chunk_warning_seconds:
                phase = self._state.asr_inflight_phase or "realtime"
                audio_seconds = self._state.asr_inflight_audio_seconds
                audio_detail = (
                    f" for {audio_seconds:.1f}s audio"
                    if audio_seconds is not None
                    else ""
                )
                await self._arm_watchdog_episode(
                    code="asr_busy",
                    status="warning",
                    message=(
                        f"ASR has been processing {phase}{audio_detail} "
                        f"for {asr_busy_seconds:.1f}s; realtime transcription may lag"
                    ),
                )
        else:
            await self._recover_watchdog_episode(
                "asr_busy",
                "ASR processing returned to realtime cadence",
            )

        backlog_seconds = self._state.audio_input_backlog_seconds
        if backlog_seconds >= self._cfg.audio_input_queue_warning_seconds:
            await self._arm_watchdog_episode(
                code="audio_backlog",
                status="warning",
                message=(
                    f"audio input backlog is {backlog_seconds:.1f}s; "
                    "ASR is consuming slower than capture"
                ),
            )
        else:
            await self._recover_watchdog_episode(
                "audio_backlog",
                "audio input backlog returned to normal",
            )

        last_chunk = self._state.last_chunk_wall_time
        if last_chunk is not None:
            no_chunk_seconds = now - last_chunk
            if asr_inflight_start is not None:
                await self._recover_watchdog_episode(
                    "audio_source_silent",
                    "audio source wait is caused by ASR processing",
                )
                await self._recover_watchdog_episode(
                    "audio_source_slow",
                    "audio source wait is caused by ASR processing",
                )
            elif no_chunk_seconds >= self._cfg.watchdog_no_chunk_error_seconds:
                self._clear_watchdog_episode("audio_source_slow")
                await self._arm_watchdog_episode(
                    code="audio_source_silent",
                    status="error",
                    message=(
                        "audio source has not produced a chunk for "
                        f"{no_chunk_seconds:.1f}s"
                    ),
                )
            elif no_chunk_seconds >= self._cfg.watchdog_no_chunk_warning_seconds:
                await self._recover_watchdog_episode(
                    "audio_source_silent",
                    "audio source resumed chunk delivery",
                )
                await self._arm_watchdog_episode(
                    code="audio_source_slow",
                    status="warning",
                    message=(
                        "audio source has not produced a chunk for "
                        f"{no_chunk_seconds:.1f}s"
                    ),
                )
            else:
                await self._recover_watchdog_episode(
                    "audio_source_silent",
                    "audio source resumed chunk delivery",
                )
                await self._recover_watchdog_episode(
                    "audio_source_slow",
                    "audio source resumed chunk delivery",
                )

        voiced_anchor = self._state.last_voiced_chunk_wall_time or self._state.first_chunk_wall_time
        if voiced_anchor is not None:
            all_silent_seconds = now - voiced_anchor
            if all_silent_seconds >= self._cfg.watchdog_all_silent_seconds:
                await self._arm_watchdog_episode(
                    code="audio_all_silent",
                    status="warning",
                    message=(
                        "audio source has produced only silent chunks for "
                        f"{all_silent_seconds:.1f}s"
                    ),
                )
            else:
                await self._recover_watchdog_episode(
                    "audio_all_silent",
                    "audio levels recovered above silence threshold",
                )

    async def _audio_watchdog(self) -> None:
        if not self._cfg.watchdog_enabled:
            return
        tick = self._cfg.watchdog_tick_seconds
        try:
            while not self._stopped.is_set():
                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=tick)
                    return
                except asyncio.TimeoutError:
                    pass
                if not self._pause_gate.is_set():
                    continue
                await self._check_audio_health()
        except asyncio.CancelledError:
            raise

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stopped.is_set():
            source = self._cfg.audio_source
            source_generation = self._audio_source_generation
            # Accumulation buffer: collect small audio chunks until we have enough
            # for a quality ASR pass, reducing latency vs the old 3s monolithic chunks.
            accum_samples: list[np.ndarray] = []
            accum_chunk_rms: list[float] = []
            accum_duration: float = 0.0
            accum_start_time: float = 0.0
            threshold = self._cfg.asr_accumulation_seconds
            # Overlap tail: last N samples from the previous ASR window, prepended to
            # the next window so words at chunk boundaries are not truncated.
            overlap_buf: np.ndarray = np.zeros(0, dtype=np.float32)

            def make_pre_vad_segmenter() -> PreVadSegmenter:
                return PreVadSegmenter(
                    sample_rate=source.sample_rate,
                    enabled=self._cfg.pre_vad_enabled,
                    detector_threshold=self._cfg.pre_vad_threshold,
                    detector_energy_rms_threshold=self._cfg.pre_vad_energy_rms_threshold,
                    trailing_silence_seconds=self._cfg.pre_vad_trailing_silence_seconds,
                    max_segment_seconds=self._cfg.pre_vad_max_segment_seconds,
                )

            pre_vad = make_pre_vad_segmenter()
            noise_reducer = AdaptiveNoiseReducer(
                sample_rate=source.sample_rate,
                enabled=self._cfg.denoise_enabled,
                strength=self._cfg.denoise_strength,
                noise_update_rms_threshold=self._cfg.denoise_noise_update_rms_threshold,
            )
            pre_vad_conditioner = AudioConditioner(
                enabled=self._cfg.pre_vad_conditioning_enabled,
                target_rms=self._cfg.pre_vad_conditioning_target_rms,
                min_rms_for_gain=self._cfg.pre_vad_conditioning_min_rms,
                max_gain=self._cfg.pre_vad_conditioning_max_gain,
                peak_limit=self._cfg.audio_conditioning_peak_limit,
            )
            audio_conditioner = AudioConditioner(
                enabled=self._cfg.audio_conditioning_enabled,
                target_rms=self._cfg.audio_conditioning_target_rms,
                min_rms_for_gain=self._cfg.audio_conditioning_min_rms,
                max_gain=self._cfg.audio_conditioning_max_gain,
                peak_limit=self._cfg.audio_conditioning_peak_limit,
            )
            source_switched = False
            weak_rescue_samples: list[np.ndarray] = []
            weak_rescue_start_time: Optional[float] = None
            weak_rescue_duration: float = 0.0
            weak_rescue_last_attempt_end: float = -1_000_000.0
            recent_audio_chunks: deque[AudioChunk] = deque()
            recent_audio_duration = 0.0
            recent_audio_max_seconds = max(
                self._cfg.fast_preview_window_seconds + 2.0,
                self._cfg.fast_preview_window_seconds * 2.0,
            )

            def append_recent_audio(chunk: AudioChunk) -> None:
                nonlocal recent_audio_duration
                if not self._cfg.fast_preview_enabled:
                    return
                samples = chunk.samples.astype(np.float32, copy=False).copy()
                recent_audio_chunks.append(
                    AudioChunk(
                        samples=samples,
                        sample_rate=chunk.sample_rate,
                        start_time=chunk.start_time,
                    )
                )
                recent_audio_duration += len(samples) / chunk.sample_rate
                while recent_audio_chunks and recent_audio_duration > recent_audio_max_seconds:
                    old = recent_audio_chunks.popleft()
                    recent_audio_duration = max(
                        0.0,
                        recent_audio_duration - len(old.samples) / old.sample_rate,
                    )

            def snapshot_recent_audio() -> tuple[np.ndarray, int, float] | None:
                if not recent_audio_chunks:
                    return None
                sample_rate = recent_audio_chunks[-1].sample_rate
                target_frames = max(1, int(self._cfg.fast_preview_window_seconds * sample_rate))
                selected: list[AudioChunk] = []
                frame_count = 0
                for chunk in reversed(recent_audio_chunks):
                    if chunk.sample_rate != sample_rate:
                        break
                    selected.append(chunk)
                    frame_count += len(chunk.samples)
                    if frame_count >= target_frames:
                        break
                if not selected:
                    return None
                selected.reverse()
                concat_start = time.perf_counter()
                samples = np.concatenate([chunk.samples for chunk in selected]).astype(np.float32, copy=False)
                if len(samples) > target_frames:
                    samples = samples[-target_frames:]
                    total_selected_frames = sum(len(chunk.samples) for chunk in selected)
                    buf_start = selected[0].start_time + max(0, total_selected_frames - len(samples)) / sample_rate
                else:
                    buf_start = selected[0].start_time

                # Clip the front to the last formally committed Whisper position.
                # Qwen re-decoding already-committed speech wastes compute and
                # causes the preview to display the tail of the previous (already
                # committed) sentence, making it look like a "previous sentence"
                # subtitle.  No overlap: the window starts exactly at the commit
                # boundary so every preview shows only genuinely new speech.
                # The window then grows naturally as speech continues, producing
                # the "growing subtitles" effect.
                committed = self._state.last_emitted_end_time
                if committed is not None and committed > 0.0:
                    clip_start = max(buf_start, committed)
                    if clip_start > buf_start:
                        trim_frames = int((clip_start - buf_start) * sample_rate)
                        if trim_frames >= len(samples):
                            return None
                        samples = samples[trim_frames:]
                        buf_start = clip_start

                out = samples.copy()
                self._state.snapshot_concat_wall_seconds = time.perf_counter() - concat_start
                return out, sample_rate, buf_start

            def snapshot_audio_span(
                start_time: float,
                end_time: float,
                *,
                margin_seconds: float,
            ) -> tuple[np.ndarray, int, float] | None:
                if not recent_audio_chunks:
                    return None
                sample_rate = recent_audio_chunks[-1].sample_rate
                span_start = max(0.0, start_time - margin_seconds)
                span_end = end_time + margin_seconds
                if span_end <= span_start:
                    return None

                slices: list[np.ndarray] = []
                actual_start: Optional[float] = None
                for chunk in recent_audio_chunks:
                    if chunk.sample_rate != sample_rate:
                        continue
                    chunk_start = chunk.start_time
                    chunk_end = chunk_start + len(chunk.samples) / sample_rate
                    if chunk_end <= span_start or chunk_start >= span_end:
                        continue
                    start_idx = max(0, int(round((span_start - chunk_start) * sample_rate)))
                    end_idx = min(len(chunk.samples), int(round((span_end - chunk_start) * sample_rate)))
                    if end_idx <= start_idx:
                        continue
                    if actual_start is None:
                        actual_start = chunk_start + start_idx / sample_rate
                    slices.append(chunk.samples[start_idx:end_idx])

                if not slices or actual_start is None:
                    return None
                samples = np.concatenate(slices).astype(np.float32, copy=False)
                return samples.copy(), sample_rate, actual_start

            async def try_whisper_retry_for_qwen(orphan: TranscriptSegment) -> bool:
                if not self._cfg.qwen_targeted_retry_enabled:
                    self._state.qwen_targeted_retry_skipped += 1
                    self._state.qwen_targeted_retry_last_reason = "disabled"
                    return False
                if self._update_resource_pressure() != "normal":
                    self._state.qwen_targeted_retry_skipped += 1
                    self._state.qwen_targeted_retry_last_reason = self._state.resource_pressure_level
                    return False

                margin = max(0.0, self._cfg.qwen_targeted_retry_margin_seconds)
                requested_audio_seconds = (orphan.end_time - orphan.start_time) + (2.0 * margin)
                if requested_audio_seconds > self._cfg.qwen_targeted_retry_max_audio_seconds:
                    self._state.qwen_targeted_retry_skipped += 1
                    self._state.qwen_targeted_retry_last_reason = "span_too_long"
                    return False

                snap = snapshot_audio_span(orphan.start_time, orphan.end_time, margin_seconds=margin)
                if snap is None:
                    self._state.qwen_targeted_retry_skipped += 1
                    self._state.qwen_targeted_retry_last_reason = "audio_unavailable"
                    return False

                samples, sample_rate, buf_start = snap
                if samples.size == 0:
                    self._state.qwen_targeted_retry_skipped += 1
                    self._state.qwen_targeted_retry_last_reason = "empty_audio"
                    return False

                samples = audio_conditioner.process_samples(samples)
                prompt = self._build_whisper_prompt()
                effective_lang = self._cfg.forced_language or (
                    self._state.locked_language if self._cfg.language_lock_enabled else None
                )
                audio_seconds = len(samples) / sample_rate
                self._state.qwen_targeted_retry_attempts += 1
                self._state.qwen_targeted_retry_last_reason = None
                logger.info(
                    "qwen_targeted_retry attempt [%.2f-%.2f] audio=%.2fs text=%r",
                    orphan.start_time,
                    orphan.end_time,
                    audio_seconds,
                    orphan.text.strip()[:80],
                )

                try:
                    self._state.asr_inflight_start_wall_time = time.monotonic()
                    self._state.asr_inflight_audio_seconds = audio_seconds
                    self._state.asr_inflight_phase = "qwen_retry"
                    await self._emit_compute_activity_state()
                    start = time.perf_counter()
                    segments = await loop.run_in_executor(
                        self._asr_executor,
                        lambda m=samples, sr=sample_rate, p=prompt, fl=effective_lang: self._cfg.asr.transcribe(
                            m,
                            sr,
                            forced_language=fl,
                            offset_seconds=0.0,
                            initial_prompt=p,
                            quality_preset="retry",
                        ),
                    )
                    wall = time.perf_counter() - start
                    self._record_asr_timing(
                        audio_seconds=audio_seconds,
                        wall_seconds=wall,
                        phase="qwen_retry",
                    )
                except Exception as exc:
                    self._state.qwen_targeted_retry_failed += 1
                    self._state.qwen_targeted_retry_last_reason = "asr_error"
                    logger.debug("qwen_targeted_retry failed: %s", exc)
                    return False
                finally:
                    self._state.asr_inflight_start_wall_time = None
                    self._state.asr_inflight_audio_seconds = None
                    self._state.asr_inflight_phase = None
                    await self._emit_compute_activity_state()

                candidates: list[TranscriptSegment] = []
                for asr_seg in segments:
                    abs_start = buf_start + asr_seg.start_time
                    abs_end = buf_start + asr_seg.end_time
                    if abs_start >= orphan.end_time + margin or abs_end <= orphan.start_time - margin:
                        continue
                    if self._is_pure_filler(asr_seg.text, asr_seg.language):
                        self._state.filler_filtered_total += 1
                        continue
                    if self._classify_asr_segment(asr_seg) != "keep":
                        continue
                    text = asr_seg.text.strip()
                    if not text or text == self._state.last_emitted_text:
                        continue
                    candidates.append(
                        TranscriptSegment(
                            id=str(uuid.uuid4()),
                            meeting_id=self._state.meeting_id,
                            start_time=abs_start,
                            end_time=abs_end,
                            text=text,
                            original_language=asr_seg.language,
                            confidence=asr_seg.confidence,
                            quality=self._compute_quality(asr_seg),
                            created_at=datetime.now(tz=timezone.utc),
                            emitted_at_elapsed_seconds=max(self._state.elapsed_seconds, abs_end),
                        )
                    )

                if not candidates:
                    self._state.qwen_targeted_retry_failed += 1
                    self._state.qwen_targeted_retry_last_reason = "no_usable_whisper_text"
                    return False

                for seg in candidates:
                    await self._persist_and_emit_segment(seg)
                self._schedule_summary(self._maybe_emit_rolling())
                self._schedule_summary(self._maybe_emit_memory())
                self._schedule_summary(self._maybe_emit_cumulative())
                self._schedule_summary(self._maybe_emit_refined_transcript())
                self._state.qwen_targeted_retry_recovered += len(candidates)
                self._state.qwen_targeted_retry_last_reason = "recovered"
                logger.info(
                    "qwen_targeted_retry recovered %d segment(s) for [%.2f-%.2f]",
                    len(candidates),
                    orphan.start_time,
                    orphan.end_time,
                )
                await self._emit_transcript_preview()
                return True

            def clear_weak_rescue_buffer() -> None:
                nonlocal weak_rescue_start_time, weak_rescue_duration
                weak_rescue_samples.clear()
                weak_rescue_start_time = None
                weak_rescue_duration = 0.0
                self._state.weak_rescue_buffer_seconds = 0.0

            def drop_live_buffers() -> None:
                nonlocal accum_duration, accum_start_time, overlap_buf, pre_vad, recent_audio_duration
                accum_samples.clear()
                accum_chunk_rms.clear()
                accum_duration = 0.0
                accum_start_time = 0.0
                overlap_buf = np.zeros(0, dtype=np.float32)
                pre_vad = make_pre_vad_segmenter()
                clear_weak_rescue_buffer()
                recent_audio_chunks.clear()
                recent_audio_duration = 0.0
                self._state.fast_preview_segment = None
                # Clear pending Qwen drafts — they belong to the previous source
                # and Whisper will never cover them on the new source.
                self._state.qwen_committed_drafts.clear()

            async def maybe_rescue_weak_speech(
                *,
                energy_chunk: AudioChunk,
                prepared_chunk: AudioChunk,
            ) -> None:
                nonlocal weak_rescue_start_time, weak_rescue_duration, weak_rescue_last_attempt_end

                if not self._cfg.weak_speech_rescue_enabled or self._asr_safeguard_active():
                    clear_weak_rescue_buffer()
                    return
                if energy_chunk.samples.size == 0 or prepared_chunk.samples.size == 0:
                    return

                rms = float(np.sqrt(np.mean(energy_chunk.samples ** 2)))
                if (
                    rms < self._cfg.weak_speech_rescue_rms_min
                    or rms > self._cfg.weak_speech_rescue_rms_max
                ):
                    clear_weak_rescue_buffer()
                    return

                if weak_rescue_start_time is None:
                    weak_rescue_start_time = prepared_chunk.start_time
                weak_rescue_samples.append(prepared_chunk.samples.astype(np.float32, copy=False).copy())
                weak_rescue_duration += len(prepared_chunk.samples) / prepared_chunk.sample_rate
                self._state.weak_rescue_buffer_seconds = weak_rescue_duration

                rescue_window_seconds = self._cfg.weak_speech_rescue_window_seconds
                fast_window_seconds = min(
                    rescue_window_seconds,
                    max(0.1, self._cfg.weak_speech_rescue_fast_window_seconds),
                )
                if rms <= self._cfg.weak_speech_rescue_fast_rms_max:
                    rescue_window_seconds = fast_window_seconds

                if weak_rescue_duration < rescue_window_seconds:
                    return

                rescue_end = weak_rescue_start_time + weak_rescue_duration
                if rescue_end - weak_rescue_last_attempt_end < self._cfg.weak_speech_rescue_cooldown_seconds:
                    clear_weak_rescue_buffer()
                    return

                merged = np.concatenate(weak_rescue_samples).astype(np.float32, copy=False)
                rescue_chunk = AudioChunk(
                    samples=merged,
                    sample_rate=prepared_chunk.sample_rate,
                    start_time=weak_rescue_start_time,
                )
                before_count = self._active_transcript_item_count()
                self._state.weak_rescue_attempts += 1
                logger.info(
                    "weak speech rescue attempt duration=%.2fs rms=%.5f start=%.2f",
                    weak_rescue_duration,
                    rms,
                    weak_rescue_start_time,
                )
                clear_weak_rescue_buffer()
                weak_rescue_last_attempt_end = rescue_end
                await consume_asr_chunk(rescue_chunk)
                after_count = self._active_transcript_item_count()
                if after_count > before_count:
                    self._state.weak_rescue_emitted += 1

            async def consume_asr_chunk(chunk: AudioChunk) -> None:
                nonlocal accum_duration, accum_start_time, overlap_buf, accum_samples, accum_chunk_rms

                if not accum_samples:
                    accum_start_time = chunk.start_time

                chunk_rms = float(np.sqrt(np.mean(chunk.samples ** 2))) if chunk.samples.size else 0.0
                accum_samples.append(chunk.samples)
                accum_chunk_rms.append(chunk_rms)
                accum_duration += len(chunk.samples) / chunk.sample_rate

                flush_threshold = self._resolve_asr_flush_threshold(
                    threshold=threshold,
                    accum_chunk_rms=accum_chunk_rms,
                    current_chunk_rms=chunk_rms,
                )

                # Normal cadence: keep accumulating until we reach the threshold.
                if accum_duration < flush_threshold:
                    return
                hard_cap_seconds = self._asr_accumulation_hard_cap_seconds()
                self._log_accumulation_flush_reason(
                    accum_duration,
                    threshold=flush_threshold,
                    hard_cap_seconds=hard_cap_seconds,
                )

                # Flush the accumulated buffer through ASR.
                merged = np.concatenate(accum_samples)
                buf_start = accum_start_time
                sample_rate = chunk.sample_rate
                accum_samples.clear()
                accum_chunk_rms.clear()
                accum_duration = 0.0

                # Skip near-silent chunks to avoid Whisper hallucinations.
                rms = float(np.sqrt(np.mean(merged ** 2)))
                if rms < self._cfg.silence_rms_threshold:
                    logger.debug("chunk silent rms=%.5f, skipping ASR", rms)
                    committed_before_silence = len(self._state.segments)
                    self._state.elapsed_seconds = max(
                        self._state.elapsed_seconds,
                        buf_start + len(merged) / sample_rate,
                    )
                    await self._flush_silence_boundary_segments()
                    await self._emit_session_progress_if_needed()
                    if len(self._state.segments) > committed_before_silence:
                        self._schedule_summary(self._maybe_emit_rolling())
                        self._schedule_summary(self._maybe_emit_memory())
                        self._schedule_summary(self._maybe_emit_cumulative())
                        self._schedule_summary(self._maybe_emit_refined_transcript())
                    overlap_buf = np.zeros(0, dtype=np.float32)
                    return

                merged = audio_conditioner.process_samples(merged)
    
                # Prepend overlap tail from the previous window to recover words
                # that would otherwise be cut at the chunk boundary.
                overlap_frames = int(self._cfg.asr_overlap_seconds * sample_rate)
                overlap_duration = len(overlap_buf) / sample_rate
                extended = np.concatenate([overlap_buf, merged]) if len(overlap_buf) > 0 else merged
                asr_buf_start = buf_start - overlap_duration
    
                prompt = self._build_whisper_prompt()
                effective_lang = self._cfg.forced_language or (
                    self._state.locked_language if self._cfg.language_lock_enabled else None
                )
                try:
                    segments = await self._transcribe_window(
                        loop,
                        extended,
                        sample_rate,
                        prompt=prompt,
                        forced_language=effective_lang,
                    )
                except Exception as exc:
                    logger.warning("ASR failed, skipping chunk: %s", exc)
                    await self._emit("error", ErrorPayload(code="asr_error", message=str(exc)))
                    overlap_buf = (
                        merged[-overlap_frames:].copy() if overlap_frames > 0 and len(merged) >= overlap_frames
                        else (merged.copy() if overlap_frames > 0 else np.zeros(0, dtype=np.float32))
                    )
                    return

                overlap_buf = (
                    merged[-overlap_frames:].copy()
                    if overlap_frames > 0 and len(merged) >= overlap_frames
                    else (merged.copy() if overlap_frames > 0 else np.zeros(0, dtype=np.float32))
                )

                # Drop Whisper segments that fall entirely within a time region
                # already promoted by Qwen — prevents duplicate formal entries.
                if self._state.qwen_covered_until is not None:
                    covered = self._state.qwen_covered_until
                    n_before = len(segments)
                    segments = [
                        s for s in segments
                        if (asr_buf_start + s.end_time) > covered + 0.1
                    ]
                    if len(segments) < n_before:
                        logger.debug(
                            "whisper_dedup: dropped %d segment(s) already covered by Qwen (covered_until=%.2f)",
                            n_before - len(segments),
                            covered,
                        )

                dia_segments = None
                if self._cfg.diarizer is not None:
                    dia_segments = await loop.run_in_executor(
                        None,
                        lambda m=merged, sr=sample_rate, bs=buf_start: self._cfg.diarizer.diarize(
                            m, sr, offset_seconds=bs,
                        ),
                    )

                repetition_tripped = False
                candidate_segments: list[TranscriptSegment] = []
                for asr_seg in segments:
                    if asr_seg.start_time < overlap_duration:
                        continue
                    if self._is_pure_filler(asr_seg.text, asr_seg.language):
                        self._state.filler_filtered_total += 1
                        logger.debug("filler segment dropped: %r (lang=%s)", asr_seg.text, asr_seg.language)
                        continue
                    if self._classify_asr_segment(asr_seg) != "keep":
                        continue
    
                    norm_text = asr_seg.text.strip()
                    if norm_text and norm_text == self._state.last_emitted_text:
                        self._state.repetition_streak += 1
                        if self._state.repetition_streak >= 2:
                            logger.warning(
                                "ASR repetition detected (%r ×%d) — dropping and resetting state",
                                norm_text, self._state.repetition_streak + 1,
                            )
                            await self._emit(
                                "error",
                                ErrorPayload(
                                    code="asr_repetition",
                                    message=f"dropped repeated segment: {norm_text!r}",
                                ),
                            )
                            repetition_tripped = True
                            continue
                    else:
                        self._state.repetition_streak = 0
    
                    seg = TranscriptSegment(
                        id=str(uuid.uuid4()),
                        meeting_id=self._state.meeting_id,
                        start_time=asr_buf_start + asr_seg.start_time,
                        end_time=asr_buf_start + asr_seg.end_time,
                        text=asr_seg.text,
                        original_language=asr_seg.language,
                        confidence=asr_seg.confidence,
                        quality=self._compute_quality(asr_seg),
                        created_at=datetime.now(tz=timezone.utc),
                        emitted_at_elapsed_seconds=max(
                            self._state.elapsed_seconds,
                            asr_buf_start + asr_seg.end_time,
                        ),
                    )
    
                    if dia_segments:
                        speaker_label = self._match_speaker(
                            seg.start_time, seg.end_time, dia_segments,
                        )
                        if speaker_label:
                            speaker = self._ensure_speaker(speaker_label)
                            seg.speaker_id = speaker.id
                    candidate_segments.append(seg)
    
                if candidate_segments and self._state.pending_segments:
                    await self._drain_non_mergeable_pending_segments(candidate_segments[0])
    
                if candidate_segments and self._state.pending_segments:
                    candidate_segments[0] = self._consume_pending_segment(candidate_segments[0])

                candidate_segments = await self._reconcile_preview_candidate(candidate_segments)
    
                tail_segment = None
                if candidate_segments and not self._is_sentence_complete(candidate_segments[-1].text):
                    tail_segment = candidate_segments.pop()
    
                preview_complete_segment = None
                if candidate_segments and tail_segment is None:
                    preview_complete_segment = candidate_segments.pop()

                for seg in candidate_segments:
                    await self._persist_and_emit_segment(seg)
    
                if tail_segment is not None:
                    await self._queue_pending_segment(tail_segment)
                elif (
                    preview_complete_segment is not None
                    and not self._is_redundant_preview_candidate(preview_complete_segment)
                ):
                    await self._queue_preview_candidate(preview_complete_segment)
    
                if repetition_tripped:
                    overlap_buf = np.zeros(0, dtype=np.float32)
    
                if self._cfg.language_lock_enabled and self._cfg.forced_language is None and segments:
                    batch_rep = next(
                        (s for s in segments if s.start_time >= overlap_duration and s.language != "unknown"),
                        None,
                    )
                    if batch_rep:
                        batch_lang = batch_rep.language
                        batch_conf = batch_rep.confidence
                        if self._state.locked_language is None:
                            self._state.language_votes[batch_lang] = (
                                self._state.language_votes.get(batch_lang, 0) + 1
                            )
                            total_votes = sum(self._state.language_votes.values())
                            if total_votes >= 5:
                                dominant = max(self._state.language_votes, key=self._state.language_votes.get)
                                self._state.locked_language = dominant
                                logger.info("language locked to %s after %d votes", dominant, total_votes)
                        else:
                            if batch_lang != self._state.locked_language and batch_conf > 0.8:
                                self._state.language_dissent_streak += 1
                                if self._state.language_dissent_streak >= 3:
                                    logger.info(
                                        "language unlocked from %s after 3 strong dissents",
                                        self._state.locked_language,
                                    )
                                    self._state.locked_language = None
                                    self._state.language_votes.clear()
                                    self._state.language_dissent_streak = 0
                            else:
                                self._state.language_dissent_streak = 0
    
                for sp in self._pending_speaker_updates:
                    await self._emit("speaker_update", sp)
                self._pending_speaker_updates.clear()
    
                self._state.elapsed_seconds = max(
                    self._state.elapsed_seconds,
                    buf_start + len(merged) / sample_rate,
                )
                await self._emit_session_progress_if_needed()
    
                self._schedule_summary(self._maybe_emit_rolling())
                self._schedule_summary(self._maybe_emit_memory())
                self._schedule_summary(self._maybe_emit_cumulative())
                self._schedule_summary(self._maybe_emit_refined_transcript())

            queue_chunk_seconds = max(0.05, self._cfg.audio_chunk_seconds)
            audio_queue_maxsize = max(
                1,
                int(self._cfg.audio_input_queue_max_seconds / queue_chunk_seconds),
            )
            audio_queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue(maxsize=audio_queue_maxsize)
            self._state.audio_input_backlog_seconds = 0.0

            async def enqueue_end_sentinel() -> None:
                while audio_queue.full():
                    dropped = audio_queue.get_nowait()
                    if dropped is not None:
                        dropped_seconds = len(dropped.samples) / dropped.sample_rate
                        self._state.audio_input_backlog_seconds = max(
                            0.0,
                            self._state.audio_input_backlog_seconds - dropped_seconds,
                        )
                        self._state.audio_input_queue_drop_total += 1
                await audio_queue.put(None)

            async def fast_preview_loop() -> None:
                if not self._cfg.fast_preview_enabled:
                    return
                interval = max(0.2, self._cfg.fast_preview_interval_seconds)
                try:
                    while not self._stopped.is_set():
                        scheduled_wakeup = loop.time() + interval
                        try:
                            await asyncio.wait_for(self._stopped.wait(), timeout=interval)
                            return
                        except asyncio.TimeoutError:
                            self._state.fast_preview_schedule_delay_seconds = max(
                                0.0,
                                loop.time() - scheduled_wakeup,
                            )
                        if source_generation != self._audio_source_generation or source is not self._cfg.audio_source:
                            return
                        if not self._pause_gate.is_set():
                            self._state.fast_preview_segment = None
                            await self._emit_transcript_preview()
                            continue
                        formal_pending = (
                            bool(self._state.pending_segments)
                            or self._state.preview_candidate_segment is not None
                        )
                        if formal_pending and not self._can_continue_preview_during_formal():
                            continue
                        if self._state.fast_preview_inflight:
                            self._state.fast_preview_skipped += 1
                            continue
                        if not self._has_dedicated_preview_asr() and self._state.asr_inflight_start_wall_time is not None:
                            self._state.fast_preview_skipped += 1
                            continue
                        if self._state.audio_input_backlog_seconds > self._cfg.fast_preview_max_backlog_seconds:
                            self._state.fast_preview_skipped += 1
                            continue
                        if self._asr_safeguard_active():
                            self._state.fast_preview_skipped += 1
                            continue
                        # When Qwen has its own executor the formal-Whisper RTF is
                        # irrelevant to preview throughput — they run on separate
                        # thread pools.  Only gate on Whisper RTF when preview
                        # shares the same executor (no dedicated preview ASR).
                        if not self._has_dedicated_preview_asr():
                            formal_rtf = self._state.asr_realtime_factor
                            if (
                                formal_rtf is not None
                                and formal_rtf > self._cfg.fast_preview_max_asr_realtime_factor
                            ):
                                self._state.fast_preview_skipped += 1
                                continue
                        # No RTF gate for dedicated preview ASR: fast_preview_inflight
                        # already prevents concurrent runs; stale RTF from a small
                        # post-commit window would self-reinforce and block indefinitely.

                        snap = snapshot_recent_audio()
                        if snap is None:
                            continue
                        samples, sample_rate, buf_start = snap
                        rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
                        if rms < self._cfg.fast_preview_min_rms:
                            self._state.fast_preview_segment = None
                            await self._emit_transcript_preview()
                            continue
                        if formal_pending:
                            self._state.preview_continued_during_formal += 1

                        conditioned = audio_conditioner.process_samples(samples)
                        effective_lang = self._cfg.forced_language or (
                            self._state.locked_language if self._cfg.language_lock_enabled else None
                        )
                        try:
                            seg = await self._transcribe_fast_preview_window(
                                loop,
                                conditioned,
                                sample_rate,
                                buf_start=buf_start,
                                prompt=self._cfg.vocabulary_hint or None,
                                forced_language=effective_lang,
                            )
                        except Exception as exc:
                            self._state.fast_preview_skipped += 1
                            logger.debug("fast preview ASR skipped: %s", exc)
                            continue
                        if seg is not None:
                            self._state.fast_preview_segment = seg
                            await self._emit_transcript_preview(seg)

                        # Safety net: promote any Qwen previews that are too old
                        # without formal Whisper coverage ("no missed sentences").
                        await self._promote_time_stranded_previews(
                            whisper_retry=try_whisper_retry_for_qwen,
                        )
                except asyncio.CancelledError:
                    raise

            async def read_audio_source() -> None:
                try:
                    async for raw_chunk in source.stream():
                        if self._stopped.is_set():
                            break
                        if source_generation != self._audio_source_generation or source is not self._cfg.audio_source:
                            break

                        now = time.monotonic()
                        if self._state.first_chunk_wall_time is None:
                            self._state.first_chunk_wall_time = now
                        self._state.last_chunk_wall_time = now
                        chunk_rms = float(np.sqrt(np.mean(raw_chunk.samples ** 2))) if raw_chunk.samples.size else 0.0
                        if chunk_rms >= self._cfg.silence_rms_threshold:
                            self._state.last_voiced_chunk_wall_time = now
                            await self._recover_watchdog_episode(
                                "audio_all_silent",
                                "audio levels recovered above silence threshold",
                            )
                        append_recent_audio(raw_chunk)

                        chunk_seconds = len(raw_chunk.samples) / raw_chunk.sample_rate
                        while audio_queue.full():
                            dropped = audio_queue.get_nowait()
                            if dropped is None:
                                continue
                            dropped_seconds = len(dropped.samples) / dropped.sample_rate
                            self._state.audio_input_backlog_seconds = max(
                                0.0,
                                self._state.audio_input_backlog_seconds - dropped_seconds,
                            )
                            self._state.audio_input_queue_drop_total += 1
                        raw_chunk.enqueue_wall_time = time.monotonic()
                        await audio_queue.put(raw_chunk)
                        self._state.audio_input_backlog_seconds += chunk_seconds
                except Exception as exc:
                    if not self._stopped.is_set():
                        logger.exception("audio source stream crashed")
                        await self._emit_watchdog_event(
                            code="audio_source_crashed",
                            message=f"audio source raised: {exc}",
                            status="error",
                        )
                finally:
                    await enqueue_end_sentinel()

            reader_task = asyncio.create_task(read_audio_source())
            preview_task = asyncio.create_task(fast_preview_loop())

            try:
                while not self._stopped.is_set():
                    raw_chunk = await audio_queue.get()
                    if raw_chunk is None:
                        break
                    if raw_chunk.enqueue_wall_time is not None:
                        self._state.audio_queue_wait_seconds = max(
                            0.0,
                            time.monotonic() - raw_chunk.enqueue_wall_time,
                        )
                    chunk_seconds = len(raw_chunk.samples) / raw_chunk.sample_rate
                    self._state.audio_input_backlog_seconds = max(
                        0.0,
                        self._state.audio_input_backlog_seconds - chunk_seconds,
                    )
                    if source_generation != self._audio_source_generation or source is not self._cfg.audio_source:
                        source_switched = True
                        break
                    if not self._pause_gate.is_set():
                        drop_live_buffers()
                        continue

                    energy_chunk = noise_reducer.process(raw_chunk)
                    prepared_chunk = pre_vad_conditioner.process(energy_chunk)
                    pre_vad.set_trailing_silence_seconds(
                        self._resolve_pre_vad_trailing_silence_seconds()
                    )
                    vad_chunks = pre_vad.push(prepared_chunk)
                    if vad_chunks:
                        clear_weak_rescue_buffer()
                    else:
                        await maybe_rescue_weak_speech(
                            energy_chunk=energy_chunk,
                            prepared_chunk=prepared_chunk,
                        )
                    for chunk in vad_chunks:
                        await consume_asr_chunk(chunk)

                if not reader_task.done():
                    reader_task.cancel()
                    try:
                        await reader_task
                    except asyncio.CancelledError:
                        pass
                if not preview_task.done():
                    preview_task.cancel()
                    try:
                        await preview_task
                    except asyncio.CancelledError:
                        pass

                if source_generation != self._audio_source_generation or source is not self._cfg.audio_source:
                    source_switched = True

                if source_switched:
                    self._clear_preview_candidate()
                    self._state.fast_preview_segment = None
                    self._state.pending_segments.clear()
                    await self._emit_transcript_preview(None)
                    continue

                for chunk in pre_vad.finish():
                    await consume_asr_chunk(chunk)

                # Flush remaining accumulated audio.
                if accum_samples:
                    merged = np.concatenate(accum_samples)
                    sample_rate = source.sample_rate
                    final_rms = float(np.sqrt(np.mean(merged ** 2)))
                    if final_rms < self._cfg.silence_rms_threshold:
                        self._state.elapsed_seconds = max(
                            self._state.elapsed_seconds,
                            accum_start_time + len(merged) / sample_rate,
                        )
                        await self._emit_session_progress_if_needed(force=True)
                        segments = []
                        overlap_duration = 0.0
                        asr_buf_start = accum_start_time
                    else:
                        merged = audio_conditioner.process_samples(merged)
                        overlap_frames = int(self._cfg.asr_overlap_seconds * sample_rate)
                        overlap_duration = len(overlap_buf) / sample_rate
                        extended = np.concatenate([overlap_buf, merged]) if len(overlap_buf) > 0 else merged
                        asr_buf_start = accum_start_time - overlap_duration
                        prompt = self._build_whisper_prompt()
                        effective_lang = self._cfg.forced_language or (
                            self._state.locked_language if self._cfg.language_lock_enabled else None
                        )
                        try:
                            segments = await self._transcribe_window(
                                loop,
                                extended,
                                sample_rate,
                                prompt=prompt,
                                forced_language=effective_lang,
                            )
                        except Exception as exc:
                            logger.warning("ASR failed on final flush, skipping: %s", exc)
                            await self._emit("error", ErrorPayload(code="asr_error", message=str(exc)))
                            segments = []
                        if self._state.qwen_covered_until is not None:
                            covered = self._state.qwen_covered_until
                            segments = [
                                s for s in segments
                                if (asr_buf_start + s.end_time) > covered + 0.1
                            ]
                    for asr_seg in segments:
                        if asr_seg.start_time < overlap_duration:
                            continue
                        if self._is_pure_filler(asr_seg.text, asr_seg.language):
                            self._state.filler_filtered_total += 1
                            logger.debug("filler segment dropped: %r (lang=%s)", asr_seg.text, asr_seg.language)
                            continue
                        if self._classify_asr_segment(asr_seg) != "keep":
                            continue
                        norm_text = asr_seg.text.strip()
                        if norm_text and norm_text == self._state.last_emitted_text:
                            logger.warning(
                                "ASR repetition detected on final flush (%r) — dropping",
                                norm_text,
                            )
                            await self._emit(
                                "error",
                                ErrorPayload(
                                    code="asr_repetition",
                                    message=f"dropped repeated segment on final flush: {norm_text!r}",
                                ),
                            )
                            continue
                        seg = TranscriptSegment(
                            id=str(uuid.uuid4()),
                            meeting_id=self._state.meeting_id,
                            start_time=asr_buf_start + asr_seg.start_time,
                            end_time=asr_buf_start + asr_seg.end_time,
                            text=asr_seg.text,
                            original_language=asr_seg.language,
                            confidence=asr_seg.confidence,
                            quality=self._compute_quality(asr_seg),
                            created_at=datetime.now(tz=timezone.utc),
                            emitted_at_elapsed_seconds=max(
                                self._state.elapsed_seconds,
                                asr_buf_start + asr_seg.end_time,
                            ),
                        )
                        if self._state.pending_segments:
                            await self._drain_non_mergeable_pending_segments(seg)
                        seg = self._consume_pending_segment(seg)
                        await self._persist_and_emit_segment(seg)
                    self._state.elapsed_seconds = max(
                        self._state.elapsed_seconds,
                        accum_start_time + len(merged) / sample_rate,
                    )
                    await self._emit_session_progress_if_needed(force=True)

                while self._state.pending_segments:
                    await self._emit_oldest_pending_segment()

                await self._flush_preview_candidate()
                await self._drain_qwen_orphan_queue()

                if self._stopped.is_set() and not self._graceful_stop_requested:
                    logger.info(
                        "fast shutdown requested; skipping summary/memory/finalization"
                    )
                    break

                # Wait for in-flight summary tasks before final summary.
                await self._drain_background_tasks(
                    self._summary_tasks,
                    timeout=self._STOP_SUMMARY_WAIT_SECONDS,
                    label="summary",
                )
                await self._drain_background_tasks(
                    self._translation_tasks,
                    timeout=self._STOP_TRANSLATION_WAIT_SECONDS,
                    label="translation",
                )

                await self._emit_memory(force=True)
                await self._emit_final()
                break
            except Exception as exc:
                if "reader_task" in locals() and not reader_task.done():
                    reader_task.cancel()
                    try:
                        await reader_task
                    except asyncio.CancelledError:
                        pass
                if "preview_task" in locals() and not preview_task.done():
                    preview_task.cancel()
                    try:
                        await preview_task
                    except asyncio.CancelledError:
                        pass
                if source_generation != self._audio_source_generation or source is not self._cfg.audio_source:
                    self._clear_preview_candidate()
                    self._state.fast_preview_segment = None
                    self._state.pending_segments.clear()
                    await self._emit_transcript_preview(None)
                    continue
                logger.exception("session loop crashed")
                await self._emit(
                    "error",
                    ErrorPayload(code="session_loop", message=str(exc)),
                )
                break

    def _update_resource_pressure(self) -> str:
        rtf = self._state.asr_realtime_factor or 0.0
        backlog = self._state.audio_input_backlog_seconds
        pending_background = sum(1 for task in self._summary_tasks if not task.done()) + sum(
            1 for task in self._translation_task_order if not task.done()
        )
        level = "normal"
        reason = None
        if self._asr_safeguard_active():
            level = "critical"
            reason = self._state.asr_safeguard_reason or "ASR safeguard active"
        elif rtf >= self._cfg.resource_critical_rtf_threshold:
            level = "critical"
            reason = f"ASR RTF {rtf:.2f} above critical threshold"
        elif backlog >= self._cfg.resource_critical_backlog_seconds:
            level = "critical"
            reason = f"audio backlog {backlog:.1f}s above critical threshold"
        elif rtf >= self._cfg.resource_pressure_rtf_threshold:
            level = "pressure"
            reason = f"ASR RTF {rtf:.2f} above pressure threshold"
        elif backlog >= self._cfg.resource_pressure_backlog_seconds:
            level = "pressure"
            reason = f"audio backlog {backlog:.1f}s above pressure threshold"
        elif pending_background >= 4 and self._cfg.resource_governor_policy == "conservative":
            level = "pressure"
            reason = f"background queue has {pending_background} pending task(s)"

        self._state.resource_pressure_level = level
        self._state.resource_governor_reason = reason
        return level

    def _governor_allows_background(self, kind: str) -> bool:
        level = self._update_resource_pressure()
        policy = (self._cfg.resource_governor_policy or "balanced").lower()
        if level == "critical":
            return False
        if level == "pressure":
            if kind in {"memory", "cumulative", "refined_transcript", "translation_backfill"}:
                return False
            if kind == "rolling":
                # Under sustained pressure, keep CPU/GPU budget for realtime ASR.
                # Conservative policy always pauses rolling summaries; balanced and
                # quality pause only when decode is clearly above realtime.
                rtf = self._state.asr_realtime_factor or 0.0
                if policy == "conservative" or rtf >= (self._cfg.resource_pressure_rtf_threshold + 0.10):
                    return False
        return True

    def _governor_skip(self, kind: str) -> None:
        self._state.resource_governor_skips += 1
        logger.debug(
            "resource governor skipped %s level=%s reason=%s",
            kind,
            self._state.resource_pressure_level,
            self._state.resource_governor_reason,
        )

    def _allow_asr_retry(self) -> tuple[bool, Optional[str]]:
        """Return whether the expensive retry pass should run for this window.

        Retry improves quality for suspicious segments but doubles decode work.
        On lower-end hardware this can destabilize realtime throughput, so we
        disable retry adaptively whenever pressure safeguards are active.
        """
        if not self._cfg.asr_retry_enabled:
            return False, "disabled by config"
        if self._asr_safeguard_active():
            return False, "ASR safeguard active"

        level = self._update_resource_pressure()
        if level != "normal":
            reason = self._state.resource_governor_reason or f"resource level {level}"
            return False, reason
        return True, None

    def _schedule_summary(self, coro) -> None:
        """Run a summary coroutine as a background task."""
        self._update_resource_pressure()
        if self._state.resource_pressure_level == "critical":
            self._governor_skip("summary")
            coro.close()
            return
        task = asyncio.create_task(coro)
        self._summary_tasks.add(task)
        task.add_done_callback(self._summary_tasks.discard)

    def _schedule_translation(self, seg: TranscriptSegment, target_language: LanguageCode) -> None:
        if seg.original_language == target_language:
            return
        if seg.translations.get(target_language):
            return

        previous_segment_id = self._translation_requested_segment_ids.get(target_language)
        if previous_segment_id and previous_segment_id != seg.id:
            self._state.translation_backlog_trim_total += 1
        self._translation_requested_segment_ids[target_language] = seg.id

        task = self._translation_workers_by_language.get(target_language)
        if task is not None and not task.done():
            return

        task = asyncio.create_task(self._drain_live_translations(target_language))
        self._translation_workers_by_language[target_language] = task
        self._translation_tasks.add(task)
        self._translation_task_order.append(task)
        task.add_done_callback(self._translation_tasks.discard)
        task.add_done_callback(self._forget_translation_task)

    def _forget_translation_task(self, task: asyncio.Task[None]) -> None:
        try:
            self._translation_task_order.remove(task)
        except ValueError:
            pass
        for language, current in list(self._translation_workers_by_language.items()):
            if current is task:
                self._translation_workers_by_language.pop(language, None)

    def _trim_translation_backlog(self) -> None:
        max_pending = self._effective_translation_max_pending()
        pending = [task for task in self._translation_task_order if not task.done()]
        self._translation_task_order = pending
        overflow = len(pending) - max_pending + 1
        if overflow <= 0:
            return
        cancelled = 0
        for task in pending:
            if cancelled >= overflow:
                break
            if task.done():
                continue
            task.cancel()
            cancelled += 1
        if cancelled:
            self._state.translation_backlog_trim_total += cancelled
            logger.info(
                "translation backlog trimmed: cancelled %d stale task(s), pending=%d max=%d",
                cancelled,
                len(pending),
                max_pending,
            )

    def _effective_translation_max_pending(self) -> int:
        max_pending = max(1, self._cfg.live_translation_max_pending)
        if self._asr_safeguard_active():
            return max(1, min(max_pending, self._cfg.live_translation_safeguard_max_pending))
        return max_pending

    def _prune_background_task_lists(self) -> None:
        self._translation_task_order = [
            task for task in self._translation_task_order if not task.done()
        ]
        self._summary_tasks = {task for task in self._summary_tasks if not task.done()}

    def _backfill_live_translations(self) -> None:
        if not self._governor_allows_background("translation_backfill"):
            self._governor_skip("translation_backfill")
            return
        if self._cfg.live_translation_language is None:
            return
        limit = max(0, self._cfg.live_translation_backfill_limit)
        if limit == 0:
            return
        latest = next(
            (
                seg for seg in reversed(self._state.segments[-limit:])
                if seg.original_language != self._cfg.live_translation_language
                and not seg.translations.get(self._cfg.live_translation_language)
            ),
            None,
        )
        if latest is not None:
            self._schedule_translation(latest, self._cfg.live_translation_language)

    def _next_translation_segment(self, target_language: LanguageCode) -> Optional[TranscriptSegment]:
        requested_segment_id = self._translation_requested_segment_ids.get(target_language)
        if requested_segment_id is not None:
            for seg in reversed(self._state.segments):
                if seg.id != requested_segment_id:
                    continue
                if seg.original_language == target_language or seg.translations.get(target_language):
                    self._translation_requested_segment_ids.pop(target_language, None)
                    break
                return seg
            else:
                self._translation_requested_segment_ids.pop(target_language, None)

        if self._state.resource_pressure_level == "critical":
            # Under critical pressure, preserve only the newest explicitly requested
            # translation and drop all opportunistic/backfill work.
            self._governor_skip("translation")
            return None

        limit = max(0, self._cfg.live_translation_backfill_limit)
        if limit == 0:
            return None
        for seg in reversed(self._state.segments[-limit:]):
            if seg.original_language == target_language:
                continue
            if seg.translations.get(target_language):
                continue
            if (seg.id, target_language) in self._translation_in_flight:
                continue
            return seg
        return None

    async def _drain_live_translations(self, target_language: LanguageCode) -> None:
        while True:
            self._trim_translation_backlog()
            seg = self._next_translation_segment(target_language)
            if seg is None:
                return
            if self._translation_requested_segment_ids.get(target_language) == seg.id:
                self._translation_requested_segment_ids.pop(target_language, None)
            await self._translate_segment(seg, target_language)

    async def _emit_session_progress_if_needed(self, *, force: bool = False) -> None:
        current_second = int(self._state.elapsed_seconds)
        if not force and current_second <= self._state.last_progress_emit_second:
            return
        self._state.last_progress_emit_second = current_second
        await self._emit(
            "session_state",
            self._session_state_payload(state=self._active_state_name()),
        )

    def _asr_accumulation_hard_cap_seconds(self) -> float:
        hard_cap = 4 * self._cfg.asr_accumulation_seconds
        if self._cfg.pre_vad_enabled:
            # Pre-VAD intentionally groups continuous speech into utterance-sized
            # chunks, so a long uninterrupted turn can legitimately exceed the
            # realtime accumulation target. Allow one accumulation window of slack
            # beyond the configured VAD segment target before calling it overflow.
            hard_cap = max(
                hard_cap,
                self._cfg.pre_vad_max_segment_seconds + self._cfg.asr_accumulation_seconds,
            )
        return hard_cap

    def _asr_safeguard_active(self) -> bool:
        return self._cfg.asr_safeguard_enabled and self._state.asr_safeguard_cooldown_windows > 0

    def _record_asr_timing(self, *, audio_seconds: float, wall_seconds: float, phase: str) -> None:
        self._state.asr_last_audio_seconds = audio_seconds
        self._state.asr_last_wall_seconds = wall_seconds
        rtf = wall_seconds / max(audio_seconds, 1e-6)
        self._state.asr_realtime_factor = rtf

        if not self._cfg.asr_safeguard_enabled:
            return

        threshold = max(0.1, self._cfg.asr_safeguard_rtf_threshold)
        if rtf >= threshold:
            was_active = self._asr_safeguard_active()
            self._state.asr_safeguard_cooldown_windows = max(
                1,
                self._cfg.asr_safeguard_cooldown_windows,
            )
            self._state.asr_safeguard_reason = (
                f"ASR realtime factor {rtf:.2f} exceeded {threshold:.2f} during {phase}"
            )
            if not was_active:
                self._state.asr_safeguard_events += 1
                logger.warning(
                    "ASR safeguard enabled rtf=%.2f wall=%.2fs audio=%.2fs phase=%s",
                    rtf,
                    wall_seconds,
                    audio_seconds,
                    phase,
                )
                asyncio.create_task(self._emit(
                    "error",
                    ErrorPayload(
                        code="asr_safeguard",
                        message=(
                            "ASR is close to falling behind realtime; "
                            "temporarily skipping retry/old translations"
                        ),
                    ),
                ))
            return

        if self._state.asr_safeguard_cooldown_windows > 0:
            self._state.asr_safeguard_cooldown_windows -= 1
            if self._state.asr_safeguard_cooldown_windows <= 0:
                logger.info("ASR safeguard cleared rtf=%.2f", rtf)
                self._state.asr_safeguard_reason = None

    def _log_accumulation_flush_reason(
        self,
        accum_duration: float,
        *,
        threshold: float,
        hard_cap_seconds: float,
    ) -> None:
        normal_long_speech_seconds = max(
            4 * threshold,
            self._cfg.pre_vad_max_segment_seconds if self._cfg.pre_vad_enabled else 0.0,
        )
        if accum_duration > hard_cap_seconds:
            self._state.last_backpressure_elapsed_seconds = self._state.elapsed_seconds
            if self._cfg.asr_safeguard_enabled:
                self._state.asr_safeguard_cooldown_windows = max(
                    1,
                    self._cfg.asr_safeguard_cooldown_windows,
                )
                self._state.asr_safeguard_reason = (
                    f"audio accumulation exceeded {hard_cap_seconds:.1f}s"
                )
                self._state.asr_safeguard_events += 1
            logger.warning(
                "real_backpressure_or_slow_asr accumulation=%.1fs threshold=%.1fs hard_cap=%.1fs — forcing flush",
                accum_duration,
                threshold,
                hard_cap_seconds,
            )
            asyncio.create_task(self._emit(
                "error",
                ErrorPayload(
                    code="asr_buffer_overflow",
                    message=(
                        f"audio accumulation buffer exceeded {hard_cap_seconds:.0f}s "
                        "— likely backpressure or slow ASR"
                    ),
                ),
            ))
            return
        if accum_duration > normal_long_speech_seconds:
            logger.info(
                "normal_long_speech_flush accumulation=%.1fs threshold=%.1fs hard_cap=%.1fs",
                accum_duration,
                threshold,
                hard_cap_seconds,
            )

    def _retry_text_signature(self, segments) -> str:
        parts = [re.sub(r"\s+", " ", seg.text).strip() for seg in segments if seg.text.strip()]
        return " | ".join(parts).casefold()

    def _retry_quality_score(self, segments) -> int:
        score_map = {"keep": 2, "retry": 1, "drop": 0}
        return sum(score_map[self._classify_asr_segment(seg)] for seg in segments)

    def _log_retry_outcome(self, realtime_segments, retry_segments) -> None:
        realtime_text = self._retry_text_signature(realtime_segments)
        retry_text = self._retry_text_signature(retry_segments)
        realtime_score = self._retry_quality_score(realtime_segments)
        retry_score = self._retry_quality_score(retry_segments)
        self._state.retry_windows_total += 1

        if realtime_text == retry_text and realtime_score == retry_score:
            outcome = "unchanged"
            self._state.retry_windows_unchanged += 1
        elif retry_score > realtime_score:
            outcome = "improved"
            self._state.retry_windows_improved += 1
        else:
            outcome = "diverged"
            self._state.retry_windows_diverged += 1

        logger.info(
            "retry %s realtime_score=%d retry_score=%d realtime_text=%r retry_text=%r",
            outcome,
            realtime_score,
            retry_score,
            realtime_text[:160],
            retry_text[:160],
        )

    async def _transcribe_window(
        self,
        loop: asyncio.AbstractEventLoop,
        samples: np.ndarray,
        sample_rate: int,
        *,
        prompt: Optional[str],
        forced_language: Optional[str],
    ):
        audio_seconds = len(samples) / sample_rate
        try:
            self._state.asr_inflight_start_wall_time = time.monotonic()
            self._state.asr_inflight_audio_seconds = audio_seconds
            self._state.asr_inflight_phase = "realtime"
            await self._emit_compute_activity_state()
            start = time.perf_counter()
            segments = await loop.run_in_executor(
                self._asr_executor,
                lambda m=samples, sr=sample_rate, p=prompt, fl=forced_language: self._cfg.asr.transcribe(
                    m,
                    sr,
                    forced_language=fl,
                    offset_seconds=0.0,
                    initial_prompt=p,
                    quality_preset="realtime",
                ),
            )
            realtime_wall = time.perf_counter() - start
            self._record_asr_timing(
                audio_seconds=audio_seconds,
                wall_seconds=realtime_wall,
                phase="realtime",
            )
            has_retry_candidates = any(
                self._classify_asr_segment(seg) == "retry" for seg in segments
            )
            retry_allowed, retry_reason = self._allow_asr_retry()
            if (
                has_retry_candidates
                and retry_allowed
            ):
                logger.info("retrying ASR window with higher-quality preset")
                self._state.asr_inflight_start_wall_time = time.monotonic()
                self._state.asr_inflight_phase = "retry"
                await self._emit_compute_activity_state()
                retry_start = time.perf_counter()
                retry_segments = await loop.run_in_executor(
                    self._asr_executor,
                    lambda m=samples, sr=sample_rate, p=prompt, fl=forced_language: self._cfg.asr.transcribe(
                        m,
                        sr,
                        forced_language=fl,
                        offset_seconds=0.0,
                        initial_prompt=p,
                        quality_preset="retry",
                    ),
                )
                retry_wall = time.perf_counter() - retry_start
                self._record_asr_timing(
                    audio_seconds=audio_seconds,
                    wall_seconds=realtime_wall + retry_wall,
                    phase="retry",
                )
                self._log_retry_outcome(segments, retry_segments)
                return retry_segments
            if has_retry_candidates and retry_reason:
                logger.info("adaptive load shedding: skipping retry (%s)", retry_reason)
            return segments
        finally:
            self._state.asr_inflight_start_wall_time = None
            self._state.asr_inflight_audio_seconds = None
            self._state.asr_inflight_phase = None
            await self._emit_compute_activity_state()

    async def _transcribe_fast_preview_window(
        self,
        loop: asyncio.AbstractEventLoop,
        samples: np.ndarray,
        sample_rate: int,
        *,
        buf_start: float,
        prompt: Optional[str],
        forced_language: Optional[str],
    ) -> Optional[TranscriptSegment]:
        audio_seconds = len(samples) / sample_rate
        if audio_seconds <= 0.0:
            return None

        self._state.fast_preview_attempts += 1
        self._state.fast_preview_inflight = True
        await self._emit_compute_activity_state()
        start = time.perf_counter()
        try:
            segments = await self._run_fast_preview_transcribe(
                loop,
                samples,
                sample_rate,
                prompt=prompt,
                forced_language=forced_language,
            )
        finally:
            wall = time.perf_counter() - start
            self._state.fast_preview_last_audio_seconds = audio_seconds
            self._state.fast_preview_last_wall_seconds = wall
            self._state.fast_preview_realtime_factor = wall / max(audio_seconds, 1e-6)
            self._state.fast_preview_inflight = False
            await self._emit_compute_activity_state()

        last_committed_rel = None
        if self._state.last_emitted_end_time is not None:
            last_committed_rel = max(0.0, self._state.last_emitted_end_time - buf_start)

        usable: list[ASRSegment] = []
        for asr_seg in segments:
            if last_committed_rel is not None and asr_seg.end_time <= last_committed_rel + 0.1:
                continue
            if not asr_seg.text.strip():
                continue
            if self._is_pure_filler(asr_seg.text, asr_seg.language):
                continue
            if self._classify_asr_segment(asr_seg) == "drop":
                continue
            usable.append(asr_seg)

        if not usable:
            self._state.fast_preview_skipped += 1
            return None

        first = usable[0]
        last = usable[-1]
        language = last.language
        text = ""
        for asr_seg in usable:
            text = self._join_segment_text(text, asr_seg.text.strip(), asr_seg.language)
        if not text.strip():
            self._state.fast_preview_skipped += 1
            return None

        quality = "ok"
        for asr_seg in usable:
            quality = self._merge_quality(quality, self._compute_quality(asr_seg))

        seg = TranscriptSegment(
            id=f"preview-{uuid.uuid4()}",
            meeting_id=self._state.meeting_id,
            start_time=buf_start + first.start_time,
            end_time=buf_start + last.end_time,
            text=text,
            original_language=language,
            confidence=min(s.confidence for s in usable),
            quality=quality,
            created_at=datetime.now(tz=timezone.utc),
            emitted_at_elapsed_seconds=max(
                self._state.elapsed_seconds,
                buf_start + last.end_time,
            ),
        )
        self._state.fast_preview_emitted += 1
        return seg

    async def _run_fast_preview_transcribe(
        self,
        loop: asyncio.AbstractEventLoop,
        samples: np.ndarray,
        sample_rate: int,
        *,
        prompt: Optional[str],
        forced_language: Optional[str],
    ) -> list[ASRSegment]:
        if (
            self._cfg.preview_asr is not None
            and self._preview_asr_disabled
            and not self._cfg.preview_asr_fallback_on_error
        ):
            return []

        executor = self._asr_executor
        adapter = self._cfg.asr
        if self._has_dedicated_preview_asr():
            executor = self._preview_asr_executor or self._asr_executor
            adapter = self._cfg.preview_asr or self._cfg.asr
        try:
            return await loop.run_in_executor(
                executor,
                lambda m=samples, sr=sample_rate, p=prompt, fl=forced_language, asr=adapter: asr.transcribe(
                    m,
                    sr,
                    forced_language=fl,
                    offset_seconds=0.0,
                    initial_prompt=p,
                    quality_preset="realtime",
                ),
            )
        except Exception as exc:
            if adapter is self._cfg.preview_asr and self._cfg.preview_asr is not None:
                self._disable_dedicated_preview_asr(exc)
                if not self._cfg.preview_asr_fallback_on_error:
                    return []
                return await loop.run_in_executor(
                    self._asr_executor,
                    lambda m=samples, sr=sample_rate, p=prompt, fl=forced_language: self._cfg.asr.transcribe(
                        m,
                        sr,
                        forced_language=fl,
                        offset_seconds=0.0,
                        initial_prompt=p,
                        quality_preset="realtime",
                    ),
                )
            raise

    def _classify_asr_segment(self, seg) -> str:
        text = seg.text.strip()
        if not text:
            return "drop"

        duration = max(0.0, seg.end_time - seg.start_time)
        signal_count = 0

        if (
            seg.no_speech_prob is not None
            and seg.no_speech_prob >= self._cfg.suspicious_segment_no_speech_prob
        ):
            signal_count += 1
        if (
            seg.avg_logprob is not None
            and seg.avg_logprob <= self._cfg.suspicious_segment_avg_logprob
        ):
            signal_count += 1
        if (
            seg.compression_ratio is not None
            and seg.compression_ratio >= self._cfg.suspicious_segment_compression_ratio
        ):
            signal_count += 1
        # Structural duration/density heuristics intentionally removed.
        # Whisper's own three signals (no_speech_prob, avg_logprob, compression_ratio)
        # are the model's built-in hallucination detectors and are authoritative for
        # all languages.  Duration/density rules misfired on:
        #   - Short English/German words ("yesterday", "Bundestag") at fast speech rates
        #   - Any segment where Whisper under-reports timestamp duration (common)
        #   - Normal CJK speech (one character ≈ one syllable, density looks high)
        # Removing them reduces false drops across EN / DE / ZH equally.

        if signal_count < 2:
            return "keep"

        decision = "drop" if signal_count >= 3 else "retry"

        logger.info(
            "%s suspicious ASR segment text=%r duration=%.2f conf=%.2f logprob=%s no_speech=%s compression=%s signals=%d",
            decision,
            text,
            duration,
            seg.confidence,
            seg.avg_logprob,
            seg.no_speech_prob,
            seg.compression_ratio,
            signal_count,
        )
        return decision

    def _is_suspicious_asr_segment(self, seg) -> bool:
        return self._classify_asr_segment(seg) == "drop"

    def _resolve_asr_flush_threshold(
        self,
        *,
        threshold: float,
        accum_chunk_rms: list[float],
        current_chunk_rms: float,
    ) -> float:
        if not self._cfg.asr_early_flush_enabled:
            return threshold

        early_threshold = min(threshold, max(0.1, self._cfg.asr_early_flush_min_seconds))
        if len(accum_chunk_rms) == 1:
            return early_threshold

        if (
            current_chunk_rms < self._cfg.silence_rms_threshold
            and any(rms >= self._cfg.silence_rms_threshold for rms in accum_chunk_rms[:-1])
        ):
            return early_threshold

        return threshold

    def _resolve_pre_vad_trailing_silence_seconds(self) -> float:
        base = max(0.0, self._cfg.pre_vad_trailing_silence_seconds)
        if not self._cfg.pre_vad_adaptive_trailing_silence_enabled:
            return base
        if self._asr_safeguard_active():
            return base
        if self._state.pending_segments or self._state.preview_candidate_segment is not None:
            return base

        rtf = self._state.asr_realtime_factor
        if rtf is not None and rtf > self._cfg.pre_vad_adaptive_max_realtime_factor:
            return base

        fast = max(0.0, self._cfg.pre_vad_adaptive_fast_trailing_silence_seconds)
        return min(base, fast)

    def _is_sentence_complete(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.endswith((".", "!", "?", ",", ";", ":", "…", "。", "！", "？", "，", "；", "：")):
            return True
        if stripped.endswith(("-", "(", "[", "{")):
            return False
        tokens = re.findall(r"[A-Za-z']+|[\u4e00-\u9fff]+", stripped)
        if not tokens:
            return False
        last_token = tokens[-1].lower()
        if last_token in {
            "and", "or", "but", "so", "if", "when", "because", "that", "which", "who",
            "with", "for", "of", "to", "in", "on", "at", "from", "the", "a", "an",
            "is", "are", "was", "were", "be", "been", "being",
            "的", "了", "和", "或", "但", "所以", "如果", "因为", "就是", "然后", "这", "這",
            "这个", "這個", "那个", "那個", "让", "讓", "把", "跟", "在", "是", "要",
        }:
            return False
        if any("\u4e00" <= ch <= "\u9fff" for ch in stripped):
            return len(stripped) >= 10
        return len(tokens) >= 4

    def _latest_pending_segment(self) -> Optional[TranscriptSegment]:
        if not self._state.pending_segments:
            return None
        return self._state.pending_segments[-1]

    def _latest_preview_segment(self) -> Optional[TranscriptSegment]:
        pending = self._latest_pending_segment()
        if pending is not None:
            return pending
        if self._state.preview_candidate_segment is not None:
            return self._state.preview_candidate_segment
        return self._state.fast_preview_segment

    def _can_merge_pending_segment(self, seg: TranscriptSegment) -> bool:
        pending = self._latest_pending_segment()
        if pending is None:
            return False
        same_language = pending.original_language == seg.original_language
        close_in_time = (seg.start_time - pending.end_time) <= 2.0
        return same_language and close_in_time

    def _join_segment_text(self, left: str, right: str, language: str) -> str:
        left = left.rstrip()
        right = right.lstrip()
        if not left:
            return right
        if not right:
            return left
        if language in {"zh", "unknown"}:
            return left + right
        return f"{left} {right}"

    def _consume_pending_segment(self, seg: TranscriptSegment) -> TranscriptSegment:
        pending = self._latest_pending_segment()
        if pending is None:
            return seg
        if not self._can_merge_pending_segment(seg):
            return seg
        self._state.pending_segments.pop()
        return TranscriptSegment(
            id=pending.id,
            meeting_id=pending.meeting_id,
            start_time=pending.start_time,
            end_time=seg.end_time,
            text=self._join_segment_text(pending.text, seg.text, seg.original_language),
            original_language=seg.original_language,
            speaker_id=pending.speaker_id or seg.speaker_id,
            confidence=min(pending.confidence, seg.confidence),
            translations=dict(pending.translations),
            created_at=seg.created_at,
            emitted_at_elapsed_seconds=seg.emitted_at_elapsed_seconds,
        )

    async def _whisper_replaces_qwen_drafts(self, whisper_seg: TranscriptSegment) -> None:
        """Remove Qwen-draft segments that overlap with an incoming Whisper segment.

        Qwen fast-promotes segments as quality="low" placeholders so the user
        sees text quickly.  When Whisper commits the same time region its output
        is preferred: the draft is deleted from the DB and a
        ``transcript_segment_removed`` event lets the frontend swap it out.
        """
        if not self._state.qwen_committed_drafts:
            return
        to_remove = [
            seg_id
            for seg_id, (s, e) in self._state.qwen_committed_drafts.items()
            if s < whisper_seg.end_time and e > whisper_seg.start_time
        ]
        if not to_remove:
            return
        for seg_id in to_remove:
            del self._state.qwen_committed_drafts[seg_id]
            self._state.segments = [s for s in self._state.segments if s.id != seg_id]
            self._cfg.storage.delete_segment(seg_id)
            await self._emit(
                "transcript_segment_removed",
                TranscriptSegmentRemovedPayload(segment_id=seg_id),
            )
            logger.info(
                "qwen_draft_replaced [%.2f-%.2f] removed_draft=%s",
                whisper_seg.start_time,
                whisper_seg.end_time,
                seg_id[:8],
            )

    async def _persist_and_emit_segment(
        self, seg: TranscriptSegment, *, is_qwen_draft: bool = False
    ) -> None:
        if not is_qwen_draft:
            await self._whisper_replaces_qwen_drafts(seg)
        self._state.segments.append(seg)
        self._cfg.storage.insert_segment(seg)
        self._state.last_emitted_text = seg.text.strip()
        # Only the formal Whisper commit advances the clip boundary used by
        # snapshot_recent_audio().  Qwen-draft promotions must NOT move it:
        # that would cause the next preview window to start at the draft's
        # end_time, making short post-commit windows and stale-suppression false
        # positives that produce the "previous sentence still showing" bug.
        if not is_qwen_draft:
            self._state.last_emitted_end_time = seg.end_time
        # Clear the fast-preview overlay whenever Whisper formally commits any
        # segment whose time range overlaps the current preview.  The previous
        # condition (preview.end_time <= commit.end_time + 0.1) was too narrow:
        # it left previews showing already-committed content when the preview's
        # end_time extended past the commit boundary by more than 0.1 s.
        if (
            not is_qwen_draft
            and self._state.fast_preview_segment is not None
            and self._state.fast_preview_segment.start_time < seg.end_time
        ):
            self._state.fast_preview_segment = None
        # Alignment diagnostics: compare overlapping recent preview text with formal text.
        self._run_alignment_diagnostics(seg)
        await self._emit("transcript_segment", seg)
        if self._cfg.live_translation_language is not None:
            self._schedule_translation(seg, self._cfg.live_translation_language)

    def _run_alignment_diagnostics(self, formal: TranscriptSegment) -> None:
        """Compare formally committed segment against overlapping recent preview segments."""
        overlapping = [
            p for p in self._state.recent_preview_segments
            if p.start_time < formal.end_time and p.end_time > formal.start_time
        ]
        formal_text = formal.text.strip()
        for preview in overlapping:
            preview_text = preview.text.strip()
            if not preview_text or not formal_text:
                continue
            ratio = difflib.SequenceMatcher(None, preview_text, formal_text).ratio()
            self._state.preview_alignment_compared += 1
            self._state.preview_alignment_similarity_sum += ratio
            self._state.preview_alignment_similarity_last = ratio
            logger.debug(
                "preview alignment: preview=%r formal=%r similarity=%.3f",
                preview_text[:60],
                formal_text[:60],
                ratio,
            )
        # Prune matched segments to avoid unbounded growth.
        if overlapping:
            self._state.recent_preview_segments = [
                p for p in self._state.recent_preview_segments
                if p not in overlapping
            ]
        self._mark_unconfirmed_previews_after_formal(formal)

    def _mark_unconfirmed_previews_after_formal(self, formal: TranscriptSegment) -> None:
        """Count previews that formal ASR has advanced past without overlapping.

        This is diagnostic-only: it never persists Qwen preview text. A high count
        suggests that either Qwen hallucinated temporary text or the formal lane
        missed/filtered speech that preview heard.
        """
        if not self._state.recent_preview_segments:
            return
        grace = max(0.3, self._cfg.preview_stale_tolerance_seconds)
        confirmed_until = formal.end_time - grace
        if confirmed_until <= 0:
            return

        remaining = []
        for preview in self._state.recent_preview_segments:
            if preview.end_time <= confirmed_until:
                text = preview.text.strip()
                if text:
                    self._state.preview_unconfirmed_after_formal += 1
                    self._state.preview_unconfirmed_last_text = text[:160]
                    logger.debug(
                        "preview unconfirmed after formal: preview=[%.2f-%.2f] formal=[%.2f-%.2f] text=%r",
                        preview.start_time,
                        preview.end_time,
                        formal.start_time,
                        formal.end_time,
                        text[:80],
                    )
                    # Queue for promotion when dedicated preview ASR (Qwen) is active.
                    if self._has_dedicated_preview_asr():
                        self._state.qwen_orphan_queue.append(preview)
                continue
            remaining.append(preview)
        self._state.recent_preview_segments = remaining

    async def _translate_segment(self, seg: TranscriptSegment, target_language: LanguageCode) -> None:
        if seg.original_language == target_language:
            return
        if seg.translations.get(target_language):
            return
        task_key = (seg.id, target_language)
        if task_key in self._translation_in_flight:
            return
        self._translation_in_flight.add(task_key)
        try:
            if self._stopped.is_set():
                return
            loop = asyncio.get_running_loop()
            translated = await loop.run_in_executor(
                self._translation_executor,
                lambda: self._cfg.translator.translate(
                    seg.text,
                    source_language=seg.original_language,
                    target_language=target_language,
                ),
            )
            translated = translated.strip()
            await self._maybe_emit_translation_error()
            if not translated or translated == seg.text:
                return
            seg.translations[target_language] = translated
            self._cfg.storage.update_segment_translations(seg.id, seg.translations)
            await self._emit(
                "transcript_translation",
                TranscriptTranslationPayload(
                    segment_id=seg.id,
                    language=target_language,
                    text=translated,
                ),
            )
        except Exception as exc:
            if "cannot schedule new futures after shutdown" in str(exc):
                logger.debug("translation skipped (session shutting down): %s -> %s", seg.id, target_language)
                return
            logger.warning("live transcript translation failed for %s -> %s: %s", seg.id, target_language, exc)
        finally:
            self._translation_in_flight.discard(task_key)

    async def _emit_transcript_preview(self, segment: Optional[TranscriptSegment] = None) -> None:
        backend = self._cfg.preview_asr_backend_name
        is_experimental = backend == "qwen3"
        quality_note = "experimental_fast_preview" if is_experimental else None
        # Resolve which segment to show (explicit argument beats state lookup).
        resolved = segment if segment is not None else self._latest_preview_segment()
        if self._cfg.runtime_profile == "summary_only":
            resolved = None
        # Stale-preview suppression: if the segment's time range is already fully
        # covered by the committed formal transcript, suppress it and emit null.
        if resolved is not None and self._state.last_emitted_end_time is not None:
            if resolved.end_time <= self._state.last_emitted_end_time + self._cfg.preview_stale_tolerance_seconds:
                self._state.preview_stale_suppressed += 1
                logger.debug(
                    "preview stale-suppressed: seg.end=%.3f formal_end=%.3f tol=%.3f",
                    resolved.end_time,
                    self._state.last_emitted_end_time,
                    self._cfg.preview_stale_tolerance_seconds,
                )
                resolved = None
        # Track non-null previews in a ring buffer for alignment diagnostics.
        if resolved is not None:
            self._state.recent_preview_segments.append(resolved)
            # Keep at most 20 recent preview segments to bound memory.
            if len(self._state.recent_preview_segments) > 20:
                self._state.recent_preview_segments.pop(0)
        await self._emit(
            "transcript_preview",
            TranscriptPreviewPayload(
                segment=resolved,
                preview_backend=backend,
                preview_is_experimental=is_experimental,
                preview_quality_note=quality_note,
            ),
        )

    async def _emit_oldest_pending_segment(self) -> None:
        if not self._state.pending_segments:
            return
        oldest = self._state.pending_segments.pop(0)
        await self._persist_and_emit_segment(oldest)
        await self._emit_transcript_preview()

    async def _drain_non_mergeable_pending_segments(self, seg: TranscriptSegment) -> None:
        while len(self._state.pending_segments) > 1:
            await self._emit_oldest_pending_segment()
        if self._state.pending_segments and not self._can_merge_pending_segment(seg):
            await self._emit_oldest_pending_segment()

    async def _queue_pending_segment(self, seg: TranscriptSegment) -> None:
        while len(self._state.pending_segments) >= 2:
            await self._emit_oldest_pending_segment()
        self._state.pending_segments.append(seg)
        await self._emit_transcript_preview()

    def _match_speaker(
        self, seg_start: float, seg_end: float, dia_segments
    ) -> Optional[str]:
        """Find the diarization segment with the best time overlap."""
        best_label = None
        best_overlap = 0.0
        for ds in dia_segments:
            overlap_start = max(seg_start, ds.start_time)
            overlap_end = min(seg_end, ds.end_time)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = ds.speaker_label
        return best_label

    def _ensure_speaker(self, label: str) -> Speaker:
        """Get or create a Speaker for the given label, persisting to DB."""
        if label in self._state.speakers:
            return self._state.speakers[label]
        speaker = Speaker(
            id=str(uuid.uuid4()),
            meeting_id=self._state.meeting_id,
            inferred_label=label,
            confidence=0.5,
        )
        self._state.speakers[label] = speaker
        self._cfg.storage.upsert_speaker(speaker)
        self._pending_speaker_updates.append(speaker)
        return speaker

    def _segments_in_window(self, window_seconds: float) -> list[TranscriptSegment]:
        cutoff = max(0.0, self._state.elapsed_seconds - window_seconds)
        return [s for s in self._state.segments if s.end_time >= cutoff]

    def _summary_context_segments(self) -> list[TranscriptSegment]:
        """Recent raw transcript kept beside the compressed meeting memory.

        Long meetings should not send the full transcript to the LLM for every
        cumulative/final summary. The compressed memory carries durable state;
        this tail preserves fresh wording and unsummarized details.
        """

        tail = self._segments_in_window(self._cfg.summary_tail_seconds)
        if tail:
            return tail
        return list(self._state.segments[-10:])

    def _refinement_context_segments(self) -> list[TranscriptSegment]:
        """Formal transcript span used for a non-authoritative clean conversation."""

        if self._cfg.refinement_window_seconds <= 0:
            return list(self._state.segments)
        tail = self._segments_in_window(self._cfg.refinement_window_seconds)
        if tail:
            return tail
        return list(self._state.segments[-12:])

    def _refinement_hint_context(self, segments: list[TranscriptSegment]) -> str:
        """Compact Qwen/preview hints for transcript refinement.

        This context is intentionally diagnostic-only. The raw formal transcript
        remains the source of truth; the LLM may use these hints to flag likely
        gaps or obvious ASR corrections, but the refined snapshot is never fed
        back into persistence or later ASR.
        """

        if not segments:
            return ""
        start = min(s.start_time for s in segments)
        end = max(s.end_time for s in segments)
        previews = [
            p for p in self._state.recent_preview_segments
            if p.start_time < end and p.end_time > start and getattr(p, "text", "").strip()
        ]
        previews = previews[-8:]
        lines = [
            "Fusion context for producing a clean conversation record.",
            "Compare Whisper/formal transcript and Qwen/preview hints case by case.",
            "Do not assume either engine is globally better; choose the locally more plausible wording from timing, language, overlap, and context.",
            "If neither source is reliable, keep the safer wording and mark the conflict as low confidence.",
            f"Preview alignment compared: {self._state.preview_alignment_compared}",
            f"Last preview/formal similarity: {self._state.preview_alignment_similarity_last if self._state.preview_alignment_similarity_last is not None else 'unknown'}",
            f"Preview unconfirmed after formal: {self._state.preview_unconfirmed_after_formal}",
        ]
        if self._state.preview_unconfirmed_last_text:
            lines.append(f"Last unconfirmed preview text: {self._state.preview_unconfirmed_last_text}")
        if previews:
            lines.append("Recent temporary preview hints in this window:")
            for preview in previews:
                text = preview.text.strip().replace("\n", " ")
                lines.append(f"- [{preview.start_time:.1f}-{preview.end_time:.1f}] {text[:220]}")
        else:
            lines.append("Recent temporary preview hints in this window: none")
        return "\n".join(lines)

    async def _maybe_emit_llm_error(self) -> None:
        """Surface a silent LLM fallback to the UI, once per distinct error."""
        last_error = getattr(self._cfg.summarizer, "last_error", None)
        if last_error == self._last_summary_error_emitted:
            return
        self._last_summary_error_emitted = last_error
        if last_error:
            await self._emit(
                "error",
                ErrorPayload(
                    code="llm_unavailable",
                    message=(
                        "AI summaries fell back to offline mode "
                        f"(LLM error: {last_error}). Open Settings to check your API key."
                    ),
                ),
            )

    async def _maybe_emit_translation_error(self) -> None:
        """Surface a silent live-translation fallback to the UI, once per distinct error."""
        last_error = getattr(self._cfg.translator, "last_error", None)
        if last_error == self._last_translation_error_emitted:
            return
        self._last_translation_error_emitted = last_error
        if last_error:
            await self._emit(
                "error",
                ErrorPayload(
                    code="translation_unavailable",
                    message=(
                        "Live translation fell back to the original text "
                        f"(LLM error: {last_error}). Open Settings to check your API key."
                    ),
                ),
            )

    async def _build_and_emit_snapshot(
        self,
        *,
        summary_type: SummaryType,
        segments: list[TranscriptSegment],
        time_start: float,
        time_end: float,
        previous_summary: Optional[str] = None,
    ) -> Optional[SummarySnapshot]:
        if not segments:
            return None
        kind = summary_type if summary_type in {
            "meeting_memory",
            "rolling_summary",
            "cumulative_meeting_summary",
            "final_summary",
            "refined_transcript",
        } else "rolling_summary"
        loop = asyncio.get_running_loop()
        content = await loop.run_in_executor(
            self._summary_executor,
            lambda: self._cfg.summarizer.summarize(
                segments,
                kind=kind,  # type: ignore[arg-type]
                language=self._cfg.summary_language,
                previous_summary=previous_summary,
                vocabulary=self._cfg.vocabulary_hint,
            ),
        )
        await self._maybe_emit_llm_error()
        if not content:
            return None
        snap = SummarySnapshot(
            id=str(uuid.uuid4()),
            meeting_id=self._state.meeting_id,
            summary_type=summary_type,
            time_start=time_start,
            time_end=time_end,
            language=self._cfg.summary_language,
            content=content,
            source_segment_ids=[s.id for s in segments],
            is_latest=True,
            created_at=datetime.now(tz=timezone.utc),
        )
        self._cfg.storage.insert_snapshot(snap)
        await self._emit("summary_snapshot", snap)
        return snap

    async def _maybe_emit_rolling(self) -> None:
        now = self._state.elapsed_seconds
        overdue = (now - self._state.last_rolling_at) >= (self._cfg.rolling_interval_seconds * 2.0)
        if not self._governor_allows_background("rolling"):
            # Keep summary-only mode responsive: if rolling has been stale for too
            # long, allow one catch-up refresh even under pressure.
            if overdue:
                await self._emit_rolling(force=True)
                return
            self._governor_skip("rolling")
            return
        await self._emit_rolling(force=False)

    async def _emit_rolling(self, *, force: bool) -> None:
        if self._state.rolling_in_flight:
            return
        now = self._state.elapsed_seconds
        due = (now - self._state.last_rolling_at) >= self._cfg.rolling_interval_seconds
        if not due and not force:
            return
        window_segments = self._segments_in_window(self._cfg.rolling_window_seconds)
        if len(window_segments) < self._cfg.min_segments_for_rolling:
            return
        start = min(s.start_time for s in window_segments)
        end = max(s.end_time for s in window_segments)
        # Only feed the LLM segments that are genuinely new since the last rolling
        # summary. The previous_summary already covers everything before
        # last_rolling_end_time, so re-sending those segments just causes the LLM
        # to repeat already-summarized content.
        if self._state.last_rolling_end_time > 0.0:
            new_segments = [s for s in window_segments if s.end_time > self._state.last_rolling_end_time]
        else:
            new_segments = window_segments
        if not new_segments:
            return
        self._state.rolling_in_flight = True
        try:
            snap = await self._build_and_emit_snapshot(
                summary_type="rolling_summary",
                segments=new_segments,
                time_start=min(s.start_time for s in new_segments),
                time_end=end,
                previous_summary=self._state.latest_rolling_text,
            )
            if snap is not None:
                self._state.last_rolling_at = now
                self._state.last_rolling_end_time = end
                self._state.latest_rolling_text = snap.content
        finally:
            self._state.rolling_in_flight = False

    async def _maybe_emit_memory(self) -> None:
        if not self._governor_allows_background("memory"):
            self._governor_skip("memory")
            return
        now = self._state.elapsed_seconds
        due = (now - self._state.last_memory_at) >= self._cfg.memory_interval_seconds
        if not due:
            return
        await self._emit_memory(force=False)

    async def _emit_memory(self, *, force: bool) -> Optional[SummarySnapshot]:
        if self._state.memory_in_flight:
            return None
        new_segments = self._state.segments[self._state.last_memory_segment_index :]
        if len(new_segments) < self._cfg.min_segments_for_memory and not force:
            return None
        if not new_segments:
            return None

        self._state.memory_in_flight = True
        end_index = len(self._state.segments)
        start = min(s.start_time for s in new_segments)
        end = max(s.end_time for s in new_segments)
        try:
            snap = await self._build_and_emit_snapshot(
                summary_type="meeting_memory",
                segments=list(new_segments),
                time_start=start,
                time_end=end,
                previous_summary=self._state.latest_meeting_memory,
            )
            if snap is not None:
                self._state.latest_meeting_memory = snap.content
                self._state.last_memory_segment_index = end_index
                self._state.last_memory_at = self._state.elapsed_seconds
            return snap
        finally:
            self._state.memory_in_flight = False

    async def _maybe_emit_cumulative(self) -> None:
        if not self._governor_allows_background("cumulative"):
            self._governor_skip("cumulative")
            return
        await self._emit_cumulative(force=False)

    async def _emit_cumulative(self, *, force: bool) -> None:
        if self._state.cumulative_in_flight:
            return
        now = self._state.elapsed_seconds
        due = (now - self._state.last_cumulative_at) >= self._cfg.cumulative_interval_seconds
        if not due and not force:
            return
        context_segments = self._summary_context_segments()
        if len(context_segments) < self._cfg.min_segments_for_cumulative:
            return
        end = max(s.end_time for s in self._state.segments)
        compressed_context = self._state.latest_meeting_memory or self._state.latest_cumulative_text
        self._state.cumulative_in_flight = True
        try:
            snap = await self._build_and_emit_snapshot(
                summary_type="cumulative_meeting_summary",
                segments=context_segments,
                time_start=0.0,
                time_end=end,
                previous_summary=compressed_context,
            )
            if snap is not None:
                self._state.last_cumulative_at = now
                self._state.latest_cumulative_text = snap.content
        finally:
            self._state.cumulative_in_flight = False

    async def _maybe_emit_refined_transcript(self) -> None:
        if not self._governor_allows_background("refined_transcript"):
            self._governor_skip("refined_transcript")
            return
        await self._emit_refined_transcript(force=False)

    async def _emit_refined_transcript(self, *, force: bool) -> None:
        if self._state.refinement_in_flight:
            return
        now = self._state.elapsed_seconds
        due = (now - self._state.last_refinement_at) >= self._cfg.refinement_interval_seconds
        if not due and not force:
            return
        context_segments = self._refinement_context_segments()
        if len(context_segments) < self._cfg.min_segments_for_refinement and not force:
            return
        if not context_segments:
            return
        start = min(s.start_time for s in context_segments)
        end = max(s.end_time for s in context_segments)
        hint_context = self._refinement_hint_context(context_segments)
        self._state.refinement_in_flight = True
        try:
            snap = await self._build_and_emit_snapshot(
                summary_type="refined_transcript",
                segments=context_segments,
                time_start=start,
                time_end=end,
                previous_summary=hint_context or None,
            )
            if snap is not None:
                self._state.last_refinement_at = now
        finally:
            self._state.refinement_in_flight = False

    async def _emit_final(self) -> None:
        if self._state.retry_windows_total > 0:
            logger.info(
                "retry summary total=%d improved=%d unchanged=%d diverged=%d",
                self._state.retry_windows_total,
                self._state.retry_windows_improved,
                self._state.retry_windows_unchanged,
                self._state.retry_windows_diverged,
            )
        if not self._state.segments:
            return
        end = max(s.end_time for s in self._state.segments)
        compressed_context = self._state.latest_meeting_memory or self._state.latest_cumulative_text
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self._summary_executor,
                lambda: self._cfg.summarizer.finalize_meeting(
                    self._summary_context_segments(),
                    language=self._cfg.summary_language,
                    vocabulary=self._cfg.vocabulary_hint,
                    meeting_memory=self._state.latest_meeting_memory,
                ),
            )
            if not isinstance(result, dict):
                raise ValueError("finalize_meeting returned non-dict payload")
        except Exception as exc:
            logger.warning("finalize_meeting failed, falling back to legacy final_summary: %s", exc)
            await self._emit("error", ErrorPayload(code="finalize_failed", message=str(exc)))
            await self._build_and_emit_snapshot(
                summary_type="final_summary",
                segments=self._summary_context_segments(),
                time_start=0.0,
                time_end=end,
                previous_summary=compressed_context,
            )
            return
        chapters = []
        for item in result.get("chapters") or []:
            if not isinstance(item, dict):
                continue
            try:
                start = max(0.0, min(float(item.get("time_start", 0.0) or 0.0), end))
                finish = max(start, min(float(item.get("time_end", end) or end), end))
            except (TypeError, ValueError):
                start, finish = 0.0, end
            chapters.append({
                "title": str(item.get("title") or "Untitled chapter").strip(),
                "time_start": start,
                "time_end": finish,
                "summary": str(item.get("summary") or "").strip(),
            })
        action_items = []
        for item in result.get("action_items") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            assignee = item.get("assignee")
            due = item.get("due")
            action_items.append({
                "text": text,
                "assignee": None if assignee in (None, "") else str(assignee),
                "due": None if due in (None, "") else str(due),
            })
        source_segment_ids = [s.id for s in self._state.segments]
        for kind, content in (
            ("final_summary", str(result.get("final_summary") or "").strip()),
            ("chapter_list", json.dumps(chapters, ensure_ascii=False)),
            ("action_item_list", json.dumps(action_items, ensure_ascii=False)),
        ):
            if kind == "final_summary" and not content:
                continue
            snap = SummarySnapshot(
                id=str(uuid.uuid4()),
                meeting_id=self._state.meeting_id,
                summary_type=kind,  # type: ignore[arg-type]
                time_start=0.0,
                time_end=end,
                language=self._cfg.summary_language,
                content=content,
                source_segment_ids=source_segment_ids,
                is_latest=True,
                created_at=datetime.now(tz=timezone.utc),
            )
            self._cfg.storage.insert_snapshot(snap)
            await self._emit("summary_snapshot", snap)
