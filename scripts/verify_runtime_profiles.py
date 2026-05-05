from __future__ import annotations

import sys
import asyncio
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, "app/backend")

from meetingbro.audio.capture import AudioChunk, AudioSource  # noqa: E402
from meetingbro.main import _chunk_seconds_for_profile, _runtime_settings_from_profile  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.session.profiles import normalize_runtime_profile, runtime_profile_defaults  # noqa: E402


class _SilentSource(AudioSource):
    @property
    def sample_rate(self) -> int:
        return 16_000

    async def stream(self):
        yield AudioChunk(samples=np.zeros(1600, dtype=np.float32), sample_rate=16_000, start_time=0.0)

    async def aclose(self) -> None:
        pass


class _NoopASR:
    def transcribe(self, *args, **kwargs):
        return []


class _NoopSummarizer:
    def summarize(self, *args, **kwargs):
        return ""


class _NoopTranslator:
    def translate(self, *args, **kwargs):
        return ""


def _storage() -> MagicMock:
    storage = MagicMock()
    storage.insert_segment = MagicMock()
    storage.insert_snapshot = MagicMock()
    storage.create_meeting = MagicMock()
    storage.end_meeting = MagicMock()
    storage.update_meeting_summary_language = MagicMock()
    return storage


async def main() -> int:
    # Legacy profile aliases should normalize to the current three-profile model.
    assert normalize_runtime_profile("low-latency") == "balanced"
    assert normalize_runtime_profile("low_latency") == "balanced"
    assert normalize_runtime_profile("robust") == "performance"
    assert normalize_runtime_profile("single_language") == "balanced"
    assert normalize_runtime_profile("does-not-exist") == "balanced"

    balanced = runtime_profile_defaults("balanced")
    performance = runtime_profile_defaults("performance")
    summary_only = runtime_profile_defaults("summary_only")
    balanced_settings = _runtime_settings_from_profile("balanced")
    performance_settings = _runtime_settings_from_profile("performance")
    summary_only_settings = _runtime_settings_from_profile("summary_only")

    # Performance favors lower latency than balanced; summary-only favors stability.
    assert performance["asr_accumulation_seconds"] < balanced["asr_accumulation_seconds"]
    assert summary_only["asr_accumulation_seconds"] > balanced["asr_accumulation_seconds"]
    assert balanced["language_lock_enabled"] is False
    assert _chunk_seconds_for_profile("performance") == performance["chunk_seconds"]
    assert _chunk_seconds_for_profile("summary_only") == summary_only["chunk_seconds"]
    assert performance_settings["asr_accumulation_seconds"] < summary_only_settings["asr_accumulation_seconds"]
    assert balanced_settings["language_lock_enabled"] is False

    manager = SessionManager(
        SessionConfig(
            audio_source=_SilentSource(),
            audio_chunk_seconds=float(balanced["chunk_seconds"]),
            runtime_profile="balanced",
            asr=_NoopASR(),
            summarizer=_NoopSummarizer(),
            translator=_NoopTranslator(),
            storage=_storage(),
        )
    )
    manager.update_runtime_settings(
        forced_language=None,
        runtime_profile="summary_only",
        runtime_settings={
            "audio_chunk_seconds": float(summary_only["chunk_seconds"]),
            "asr_accumulation_seconds": float(summary_only["asr_accumulation_seconds"]),
            "language_lock_enabled": bool(summary_only["language_lock_enabled"]),
        },
    )

    payload = manager._session_state_payload(state="running")
    print(f"profile: {payload.runtime_profile}")
    print(f"chunk:   {payload.audio_chunk_seconds}")
    print(f"accum:   {payload.asr_accumulation_seconds}")
    print(f"lock:    {payload.language_lock_enabled}")

    ok = (
        payload.runtime_profile == "summary_only"
        and payload.language_lock_enabled is False
        and payload.asr_accumulation_seconds == summary_only["asr_accumulation_seconds"]
    )
    if ok:
        print("\nOK: runtime profiles normalize, map to expected settings, and update session state")
        return 0
    print("\nFAIL: runtime profile behavior regressed")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
