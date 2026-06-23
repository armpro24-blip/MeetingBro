"""Pure LocalAgreement-2 stable-prefix committer for streaming ASR.

No audio or model dependency, so it is unit-testable in isolation. Given a
sequence of hypotheses over a growing buffer (each a list of timestamped
``Word``), it commits the longest leading run of words that agree across two
consecutive hypotheses, and keeps the rest as pending. Committed words are
monotonic — once committed they never change.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_STRIP = " \t\n.,!?;:。，！？；：、"


@dataclass
class Word:
    text: str
    start: float
    end: float


def _norm(text: str) -> str:
    return text.strip().lower().strip(_STRIP)


@dataclass
class StreamingTranscriber:
    _prev: list[Word] = field(default_factory=list)
    _committed: list[Word] = field(default_factory=list)

    def step(self, hypothesis: list[Word]) -> tuple[list[Word], list[Word]]:
        """Feed the current hypothesis over the growing buffer.

        Returns ``(newly_committed, pending)`` for this step.
        """
        n = len(self._committed)
        prev_tail = self._prev[n:]
        hyp_tail = hypothesis[n:]
        agreed = 0
        for a, b in zip(prev_tail, hyp_tail):
            if _norm(a.text) == _norm(b.text) and _norm(a.text):
                agreed += 1
            else:
                break
        newly = hyp_tail[:agreed]
        self._committed.extend(newly)
        self._prev = list(hypothesis)
        pending = hypothesis[len(self._committed):]
        return newly, pending

    def flush(self) -> list[Word]:
        """Commit the remainder of the last hypothesis, then reset."""
        remaining = self._prev[len(self._committed):]
        self._committed.extend(remaining)
        result = list(self._committed)
        self.reset()
        return result

    def committed_words(self) -> list[Word]:
        return list(self._committed)

    def committed_until(self) -> float:
        return self._committed[-1].end if self._committed else 0.0

    def reset(self) -> None:
        self._prev = []
        self._committed = []
