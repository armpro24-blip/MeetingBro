"""Streaming preview ASR adapter: shared formal Whisper model + LocalAgreement.

Plugs into the existing ``preview_asr`` slot. The session manager's
``fast_preview_loop`` calls ``transcribe`` repeatedly on a window that starts at
the last formal-commit boundary and grows; this adapter decodes that window with
word timestamps, runs LocalAgreement-2 to stabilise the committed prefix, and
returns a single growing caption segment. Best-effort: never authoritative.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..schemas import OriginalLanguage
from .base import ASRAdapter, ASRSegment
from .streaming import StreamingTranscriber, Word

_PREVIEW_CONFIDENCE = 0.6


class StreamingWhisperAdapter(ASRAdapter):
    def __init__(self, formal, *, reset_shrink_seconds: float = 0.25) -> None:
        self._formal = formal
        self._committer = StreamingTranscriber()
        self._reset_shrink_seconds = reset_shrink_seconds
        self._last_window_duration: Optional[float] = None
        self._language: OriginalLanguage = "unknown"

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        forced_language: Optional[str] = None,
        offset_seconds: float = 0.0,
        initial_prompt: Optional[str] = None,
        quality_preset: str = "realtime",
    ) -> list[ASRSegment]:
        # A significant shrink in window duration means the formal lane committed
        # and clipped the front of the buffer — the utterance restarted, so reset
        # LocalAgreement state.  Window grows tick-to-tick within an utterance
        # (no reset) and only shrinks at a formal commit boundary (reset).
        current_duration = len(samples) / sample_rate
        if (
            self._last_window_duration is not None
            and current_duration < self._last_window_duration - self._reset_shrink_seconds
        ):
            self._committer.reset()
        self._last_window_duration = current_duration

        words: list[Word] = self._formal.transcribe_words(
            samples,
            sample_rate,
            forced_language=forced_language,
            offset_seconds=0.0,
            initial_prompt=initial_prompt,
        )
        if forced_language in ("zh", "en", "de"):
            self._language = forced_language  # type: ignore[assignment]

        newly, pending = self._committer.step(words)
        caption_words = self._committer.committed_words() + list(pending)
        text = " ".join(w.text for w in caption_words).strip()
        if not text:
            return []
        return [
            ASRSegment(
                start_time=caption_words[0].start,
                end_time=caption_words[-1].end,
                text=text,
                language=self._language,
                confidence=_PREVIEW_CONFIDENCE,
            )
        ]

    def flush(self) -> None:
        self._committer.flush()
        self._last_window_duration = None
