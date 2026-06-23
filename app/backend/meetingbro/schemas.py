from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


LanguageCode = Literal["zh", "en", "de"]
OriginalLanguage = Literal["zh", "en", "de", "unknown"]

SummaryType = Literal[
    "meeting_memory",
    "rolling_summary",
    "cumulative_meeting_summary",
    "time_window_summary",
    "speaker_summary",
    "final_summary",
    "refined_transcript",
    "chapter_list",
    "action_item_list",
]

SessionState = Literal["starting", "running", "paused", "ended"]


class TranscriptSegment(BaseModel):
    id: str
    meeting_id: str
    start_time: float
    end_time: float
    text: str
    original_language: OriginalLanguage
    speaker_id: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    quality: Literal["ok", "uncertain", "low"] = "ok"
    translations: dict[LanguageCode, str] = Field(default_factory=dict)
    created_at: datetime
    emitted_at_elapsed_seconds: Optional[float] = None


class SummarySnapshot(BaseModel):
    id: str
    meeting_id: str
    summary_type: SummaryType
    time_start: float
    time_end: float
    language: LanguageCode
    content: str
    source_segment_ids: list[str] = Field(default_factory=list)
    is_latest: bool = False
    translations: dict[LanguageCode, str] = Field(default_factory=dict)
    created_at: datetime


class Speaker(BaseModel):
    id: str
    meeting_id: str
    display_name: Optional[str] = None
    inferred_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_local_user: bool = False


class Note(BaseModel):
    id: str
    meeting_id: str
    content: str
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    time_seconds: Optional[float] = None
    created_at: datetime


class MeetingListItem(BaseModel):
    """Lightweight row for the meeting-history list (one entry per past meeting)."""

    id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    preferred_summary_language: str = "en"
    segment_count: int = 0
    title: Optional[str] = None


class SessionStatePayload(BaseModel):
    state: SessionState
    meeting_id: str
    elapsed_seconds: float = 0.0
    source: str = "mic"
    runtime_profile: str = "balanced"
    hardware_profile: Optional[str] = None
    hardware_summary: Optional[str] = None
    compute_cpu_active: bool = False
    compute_cpu_available: bool = True
    compute_cpu_configured: bool = False
    compute_gpu_active: bool = False
    compute_gpu_available: bool = False
    compute_gpu_configured: bool = False
    compute_activity_label: Optional[str] = None
    audio_chunk_seconds: Optional[float] = None
    asr_accumulation_seconds: Optional[float] = None
    language_lock_enabled: bool = False
    live_translation_language: Optional[LanguageCode] = None
    retry_windows_total: int = 0
    retry_windows_improved: int = 0
    retry_windows_unchanged: int = 0
    retry_windows_diverged: int = 0
    last_backpressure_elapsed_seconds: Optional[float] = None
    asr_last_audio_seconds: Optional[float] = None
    asr_last_wall_seconds: Optional[float] = None
    asr_realtime_factor: Optional[float] = None
    asr_safeguard_active: bool = False
    asr_safeguard_reason: Optional[str] = None
    asr_safeguard_events: int = 0
    resource_pressure_level: str = "normal"
    resource_governor_policy: str = "balanced"
    resource_governor_reason: Optional[str] = None
    resource_governor_skips: int = 0
    weak_rescue_attempts: int = 0
    weak_rescue_emitted: int = 0
    weak_rescue_buffer_seconds: float = 0.0
    summary_pending_count: int = 0
    translation_pending_count: int = 0
    translation_backlog_trim_total: int = 0
    audio_drop_total: int = 0
    audio_input_backlog_seconds: float = 0.0
    audio_input_queue_drop_total: int = 0
    audio_queue_wait_seconds: Optional[float] = None
    event_queue_depth: int = 0
    fast_preview_enabled: bool = False
    fast_preview_attempts: int = 0
    fast_preview_emitted: int = 0
    fast_preview_skipped: int = 0
    fast_preview_last_audio_seconds: Optional[float] = None
    fast_preview_last_wall_seconds: Optional[float] = None
    fast_preview_realtime_factor: Optional[float] = None
    fast_preview_schedule_delay_seconds: Optional[float] = None
    snapshot_concat_wall_seconds: Optional[float] = None
    preview_continued_during_formal: int = 0
    preview_stale_suppressed: int = 0
    preview_alignment_compared: int = 0
    preview_alignment_similarity_avg: Optional[float] = None
    preview_alignment_similarity_last: Optional[float] = None
    preview_unconfirmed_after_formal: int = 0
    preview_unconfirmed_last_text: Optional[str] = None
    qwen_targeted_retry_attempts: int = 0
    qwen_targeted_retry_recovered: int = 0
    qwen_targeted_retry_failed: int = 0
    qwen_targeted_retry_skipped: int = 0
    qwen_targeted_retry_last_reason: Optional[str] = None
    mixed_microphone_gain: Optional[float] = None
    mixed_system_gain: Optional[float] = None
    mixed_effective_microphone_gain: Optional[float] = None
    mixed_auto_balance_enabled: Optional[bool] = None


class TranscriptSegmentRemovedPayload(BaseModel):
    segment_id: str


class TranscriptPreviewPayload(BaseModel):
    segment: Optional[TranscriptSegment] = None
    preview_backend: str = "unknown"
    preview_is_experimental: bool = False
    preview_quality_note: Optional[str] = None


class TranscriptTranslationPayload(BaseModel):
    segment_id: str
    language: LanguageCode
    text: str


class ErrorPayload(BaseModel):
    code: str
    message: str


class CreateNoteRequest(BaseModel):
    meeting_id: str
    content: str
    source_type: Optional[str] = None
    source_id: Optional[str] = None


class ExportMeetingRequest(BaseModel):
    source: Optional[str] = None
    runtime_profile: Optional[str] = None
    summary_language: Optional[LanguageCode] = None
    subtitle_language: Optional[str] = None
    export_root: Optional[str] = None
    export_dir: Optional[str] = None


class ExportMeetingResponse(BaseModel):
    meeting_id: str
    export_dir: str
    files: list[str]
