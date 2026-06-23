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
    def __init__(self, formal, *, reset_gap_seconds: float = 0.25) -> None:
        self._formal = formal
        self._committer = StreamingTranscriber()
        self._reset_gap_seconds = reset_gap_seconds
        self._last_offset: Optional[float] = None
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
        # A forward jump in the window start means the formal lane committed and
        # a new utterance window began — reset LocalAgreement state.
        if self._last_offset is None or offset_seconds > self._last_offset + self._reset_gap_seconds:
            self._committer.reset()
        self._last_offset = offset_seconds

        words: list[Word] = self._formal.transcribe_words(
            samples,
            sample_rate,
            forced_language=forced_language,
            offset_seconds=offset_seconds,
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
        self._last_offset = None
