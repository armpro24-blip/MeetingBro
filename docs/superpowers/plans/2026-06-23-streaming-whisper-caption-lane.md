# Streaming Whisper Caption Lane (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Whisper-only majority accurate, low-latency live captions by adding an additive streaming-decode caption lane backed by the already-loaded formal Whisper model.

**Architecture:** A pure LocalAgreement-2 committer (`StreamingTranscriber`) decides which words are stable across consecutive decodes of a growing audio window. A `StreamingWhisperAdapter` (an `ASRAdapter`) wraps the shared formal Whisper model + the committer and plugs into the existing `preview_asr` slot — the manager's `fast_preview_loop` already drives a growing window from the last commit boundary, so no new UI or decode loop is needed. The formal lane stays the authoritative transcript.

**Tech Stack:** Python 3.12, faster-whisper (ctranslate2), numpy, pytest 9.x. Backend lives in `app/backend/meetingbro/`; tests in repo-root `tests/` (CI runs `pytest tests/`).

## Global Constraints

- Python **3.12+** (backend requirement; CI uses 3.12).
- **CPU-first:** the streaming lane must run on CPU with the existing `small/int8` model; no GPU requirement, no new model download.
- **Single model:** the streaming lane shares the formal `FasterWhisperAdapter`'s loaded model — no second `WhisperModel` instance.
- **Anti-hallucination:** all decode calls keep `condition_on_previous_text=False`.
- **Formal lane untouched:** stored transcript, summaries, translation come only from the formal lane in Phase 1. The streaming lane is best-effort and may be dropped under load without affecting the formal lane.
- **16 kHz mono float32** audio throughout (existing pipeline invariant).
- Run tests with the project's conda env: `conda activate MeetingBro` (Python 3.12, pytest 9.x installed). Commands below assume that env is active.

---

### Task 1: `StreamingTranscriber` — pure LocalAgreement-2 committer

**Files:**
- Create: `app/backend/meetingbro/asr/streaming.py`
- Test: `tests/test_streaming_transcriber.py`

**Interfaces:**
- Produces: `Word(text: str, start: float, end: float)` dataclass; `StreamingTranscriber` with `step(hypothesis: list[Word]) -> tuple[list[Word], list[Word]]` (returns `(newly_committed, pending)`), `flush() -> list[Word]` (commits remainder of last hypothesis, then resets), `reset() -> None`, `committed_words() -> list[Word]`, `committed_until() -> float` (end time of last committed word, or `0.0`).
- Consumes: nothing (stdlib only — keeps the unit pure and test-only-import).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_streaming_transcriber.py
from meetingbro.asr.streaming import StreamingTranscriber, Word


def w(text, start, end):
    return Word(text=text, start=start, end=end)


def test_commits_prefix_that_agrees_across_two_hypotheses():
    st = StreamingTranscriber()
    # First hypothesis: nothing agrees yet (no prior), so nothing commits.
    newly, pending = st.step([w("the", 0.0, 0.2), w("cat", 0.2, 0.5)])
    assert [x.text for x in newly] == []
    assert [x.text for x in pending] == ["the", "cat"]
    # Second hypothesis agrees on "the cat", extends with "sat".
    newly, pending = st.step([w("the", 0.0, 0.2), w("cat", 0.2, 0.5), w("sat", 0.5, 0.8)])
    assert [x.text for x in newly] == ["the", "cat"]
    assert [x.text for x in pending] == ["sat"]


def test_disagreement_holds_back_uncommitted_tail():
    st = StreamingTranscriber()
    st.step([w("the", 0.0, 0.2), w("kat", 0.2, 0.5)])
    # "the" agrees, "mat" != "kat" -> only "the" commits.
    newly, pending = st.step([w("the", 0.0, 0.2), w("mat", 0.2, 0.5)])
    assert [x.text for x in newly] == ["the"]
    assert [x.text for x in pending] == ["mat"]


def test_normalization_ignores_case_and_trailing_punctuation():
    st = StreamingTranscriber()
    st.step([w("Hello", 0.0, 0.3)])
    newly, _ = st.step([w("hello,", 0.0, 0.3), w("world", 0.3, 0.6)])
    assert [x.text for x in newly] == ["hello,"]


def test_flush_commits_remainder_and_resets():
    st = StreamingTranscriber()
    st.step([w("the", 0.0, 0.2), w("cat", 0.2, 0.5)])
    st.step([w("the", 0.0, 0.2), w("cat", 0.2, 0.5)])  # commits "the cat"
    flushed = st.flush()
    assert [x.text for x in flushed] == ["the", "cat"]
    # after flush, state is clear
    assert st.committed_words() == []
    assert st.committed_until() == 0.0


def test_committed_until_tracks_last_committed_word_end():
    st = StreamingTranscriber()
    st.step([w("a", 0.0, 0.4), w("b", 0.4, 0.9)])
    st.step([w("a", 0.0, 0.4), w("b", 0.4, 0.9)])
    assert st.committed_until() == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_streaming_transcriber.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'meetingbro.asr.streaming'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/backend/meetingbro/asr/streaming.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_streaming_transcriber.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app/backend/meetingbro/asr/streaming.py tests/test_streaming_transcriber.py
git commit -m "feat(asr): pure LocalAgreement-2 streaming committer + unit tests"
```

---

### Task 2: Word-timestamp decode on `FasterWhisperAdapter` (shared model)

**Files:**
- Modify: `app/backend/meetingbro/asr/faster_whisper_adapter.py`
- Test: `tests/test_faster_whisper_words.py`

**Interfaces:**
- Consumes: `Word` from `meetingbro.asr.streaming` (Task 1).
- Produces: `FasterWhisperAdapter.transcribe_words(samples, sample_rate, *, forced_language=None, offset_seconds=0.0, initial_prompt=None) -> list[Word]` — runs the **same shared model** with `word_timestamps=True`, `condition_on_previous_text=False`, returning flat word list with absolute times (`offset_seconds + word time`).

- [ ] **Step 1: Write the failing test** (uses a fake model so it runs without downloading Whisper)

```python
# tests/test_faster_whisper_words.py
import numpy as np
from meetingbro.asr.faster_whisper_adapter import FasterWhisperAdapter
from meetingbro.asr.streaming import Word


class _FakeWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class _FakeSeg:
    def __init__(self, words):
        self.words = words


class _FakeModel:
    def transcribe(self, samples, **kwargs):
        assert kwargs["word_timestamps"] is True
        assert kwargs["condition_on_previous_text"] is False
        seg = _FakeSeg([_FakeWord(" the", 0.0, 0.2), _FakeWord(" cat", 0.2, 0.5)])

        class _Info:
            language = "en"

        return iter([seg]), _Info()


def test_transcribe_words_returns_flat_absolute_timed_words():
    a = FasterWhisperAdapter(model_size="tiny", device="cpu")
    a._model = _FakeModel()  # inject fake; bypass real load
    words = a.transcribe_words(np.zeros(1600, dtype=np.float32), 16_000, offset_seconds=10.0)
    assert [w.text for w in words] == ["the", "cat"]
    assert isinstance(words[0], Word)
    assert words[0].start == 10.0 and round(words[1].end, 3) == 10.5


def test_transcribe_words_empty_audio_returns_empty():
    a = FasterWhisperAdapter(model_size="tiny", device="cpu")
    assert a.transcribe_words(np.zeros(0, dtype=np.float32), 16_000) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_faster_whisper_words.py -v`
Expected: FAIL — `AttributeError: 'FasterWhisperAdapter' object has no attribute 'transcribe_words'`

- [ ] **Step 3: Write minimal implementation** — add this method to `FasterWhisperAdapter` (after `transcribe`), and add the import at the top of the file.

Add import near the existing imports:

```python
from .streaming import Word
```

Add the method:

```python
    def transcribe_words(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        forced_language: Optional[str] = None,
        offset_seconds: float = 0.0,
        initial_prompt: Optional[str] = None,
    ) -> list[Word]:
        """Decode with word-level timestamps for the streaming lane.

        Uses the SAME loaded model as ``transcribe`` (no second WhisperModel).
        Returns a flat list of ``Word`` with absolute times.
        """
        if samples.size == 0:
            return []
        if sample_rate != 16_000:
            raise ValueError(f"FasterWhisperAdapter expects 16 kHz input, got {sample_rate}")
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        model = self._ensure_model()
        segments_iter, _info = model.transcribe(
            samples,
            language=forced_language,
            initial_prompt=initial_prompt or None,
            vad_filter=True,
            vad_parameters=dict(
                threshold=self._vad_threshold,
                min_speech_duration_ms=self._vad_min_speech_ms,
                min_silence_duration_ms=self._vad_min_silence_ms,
                speech_pad_ms=self._vad_speech_pad_ms,
            ),
            temperature=0.0,
            condition_on_previous_text=False,
            no_repeat_ngram_size=3,
            word_timestamps=True,
            beam_size=self._beam_size,
            best_of=1,
            multilingual=self._multilingual,
            language_detection_threshold=self._language_detection_threshold,
            language_detection_segments=self._language_detection_segments,
        )
        out: list[Word] = []
        for seg in segments_iter:
            for wd in getattr(seg, "words", None) or []:
                text = (wd.word or "").strip()
                if not text:
                    continue
                out.append(Word(text=text, start=offset_seconds + float(wd.start), end=offset_seconds + float(wd.end)))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_faster_whisper_words.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/backend/meetingbro/asr/faster_whisper_adapter.py tests/test_faster_whisper_words.py
git commit -m "feat(asr): word-timestamp decode on FasterWhisperAdapter (shared model)"
```

---

### Task 3: `StreamingWhisperAdapter` — preview adapter wrapping shared model + committer

**Files:**
- Create: `app/backend/meetingbro/asr/streaming_whisper_adapter.py`
- Test: `tests/test_streaming_whisper_adapter.py`

**Interfaces:**
- Consumes: `FasterWhisperAdapter.transcribe_words` (Task 2), `StreamingTranscriber`/`Word` (Task 1), `ASRAdapter`/`ASRSegment` from `meetingbro.asr.base`.
- Produces: `StreamingWhisperAdapter(formal: FasterWhisperAdapter, *, reset_gap_seconds: float = 0.25)` implementing `ASRAdapter.transcribe(...) -> list[ASRSegment]`. Each call decodes the given (growing) window via the shared model, runs LocalAgreement, and returns **at most one** `ASRSegment` whose text is `committed + pending` (the live caption). It resets the committer when `offset_seconds` jumps forward (new utterance / formal commit advanced).

**Behaviour notes:**
- The manager's `fast_preview_loop` calls `preview_asr.transcribe(window, sr, offset_seconds=buf_start, ...)` repeatedly on a window that starts at the last formal-commit boundary and grows. When `buf_start` advances (formal lane committed), the utterance boundary moved → reset the committer.
- Returns a single segment spanning the window so it flows through the existing preview rendering/reconciliation unchanged. Confidence is a fixed mid value (preview is advisory).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_streaming_whisper_adapter.py
import numpy as np
from meetingbro.asr.streaming import Word
from meetingbro.asr.streaming_whisper_adapter import StreamingWhisperAdapter


class _ScriptedFormal:
    """Returns a preset word list per call, ignoring audio."""
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self._i = 0

    def transcribe_words(self, samples, sample_rate, *, forced_language=None, offset_seconds=0.0, initial_prompt=None):
        words = self._scripts[self._i]
        self._i += 1
        return [Word(w[0], w[1], w[2]) for w in words]


def _samples(n=1600):
    return np.zeros(n, dtype=np.float32)


def test_emits_growing_caption_and_stabilizes_committed_prefix():
    formal = _ScriptedFormal([
        [("the", 0.0, 0.2), ("cat", 0.2, 0.5)],
        [("the", 0.0, 0.2), ("cat", 0.2, 0.5), ("sat", 0.5, 0.8)],
    ])
    a = StreamingWhisperAdapter(formal)
    s1 = a.transcribe(_samples(), 16_000, offset_seconds=0.0)
    assert s1 and s1[0].text == "the cat"          # committed("") + pending
    s2 = a.transcribe(_samples(), 16_000, offset_seconds=0.0)
    assert s2[0].text == "the cat sat"             # committed("the cat") + pending("sat")


def test_offset_advance_resets_committer_for_new_utterance():
    formal = _ScriptedFormal([
        [("the", 0.0, 0.2), ("cat", 0.2, 0.5)],
        [("dog", 5.0, 5.3)],   # new utterance, buffer start jumped to 5.0
    ])
    a = StreamingWhisperAdapter(formal)
    a.transcribe(_samples(), 16_000, offset_seconds=0.0)
    s = a.transcribe(_samples(), 16_000, offset_seconds=5.0)
    assert s[0].text == "dog"                        # not "the cat dog"


def test_empty_decode_returns_no_segment():
    formal = _ScriptedFormal([[]])
    a = StreamingWhisperAdapter(formal)
    assert a.transcribe(_samples(), 16_000, offset_seconds=0.0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_streaming_whisper_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'meetingbro.asr.streaming_whisper_adapter'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/backend/meetingbro/asr/streaming_whisper_adapter.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_streaming_whisper_adapter.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/backend/meetingbro/asr/streaming_whisper_adapter.py tests/test_streaming_whisper_adapter.py
git commit -m "feat(asr): StreamingWhisperAdapter — preview lane via shared model + LocalAgreement"
```

---

### Task 4: Wire `whisper-streaming` as a selectable preview backend

**Files:**
- Modify: `app/backend/meetingbro/main.py` (the `_build_preview_asr` builder and preview-backend selection)
- Modify: `app/backend/meetingbro/session/manager.py` (confirm `fast_preview_loop` passes the window start as `offset_seconds` to `preview_asr.transcribe`; thread it if absent)
- Test: `tests/test_preview_backend_selection.py`

**Interfaces:**
- Consumes: `StreamingWhisperAdapter` (Task 3), the formal adapter built in `main.py`.
- Produces: when the configured preview backend is `"whisper-streaming"` (or defaulted to it), `_build_preview_asr` returns a `StreamingWhisperAdapter` wrapping the formal adapter; `preview_asr_backend_name` is set to `"whisper-streaming"`.

- [ ] **Step 1: Investigate and document the call site (no code change yet)**

Read `app/backend/meetingbro/session/manager.py` `fast_preview_loop`, specifically the block that calls `self._cfg.preview_asr.transcribe(...)` with the snapshot from `snapshot_recent_audio()` (which returns `(samples, sample_rate, buf_start)`). Confirm whether `buf_start` is passed as `offset_seconds`. Write the finding as a one-line comment in the PR/commit message for this task. If `offset_seconds` is NOT already `buf_start`, change that single call to pass `offset_seconds=buf_start`. (The StreamingWhisperAdapter relies on it for utterance reset; Qwen ignores it, so the change is safe for both.)

- [ ] **Step 2: Write the failing test**

```python
# tests/test_preview_backend_selection.py
from meetingbro.asr.faster_whisper_adapter import FasterWhisperAdapter
from meetingbro.asr.streaming_whisper_adapter import StreamingWhisperAdapter
from meetingbro.main import _build_streaming_preview_adapter


def test_build_streaming_preview_wraps_formal_adapter():
    formal = FasterWhisperAdapter(model_size="tiny", device="cpu")
    preview = _build_streaming_preview_adapter(formal)
    assert isinstance(preview, StreamingWhisperAdapter)
    assert preview._formal is formal  # shares the formal model, no second load
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_preview_backend_selection.py -v`
Expected: FAIL — `ImportError: cannot import name '_build_streaming_preview_adapter'`

- [ ] **Step 4: Write minimal implementation**

In `app/backend/meetingbro/main.py`, add the factory near `_build_preview_asr`:

```python
def _build_streaming_preview_adapter(formal_asr):
    """Build the whisper-streaming preview lane, sharing the formal model."""
    from .asr.streaming_whisper_adapter import StreamingWhisperAdapter

    return StreamingWhisperAdapter(formal_asr)
```

Then, in the preview-backend resolution in `_build_preview_asr` (and/or the startup wiring that sets `app.state.preview_asr` / `preview_asr_backend_name`), add a branch: when the resolved backend name is `"whisper-streaming"`, return `_build_streaming_preview_adapter(app.state.asr)` and set `preview_asr_backend_name = "whisper-streaming"`. Set the default: when `fast_preview` is enabled, the env/profile backend is unset or `"auto"`, and the Qwen model dir does not exist, resolve to `"whisper-streaming"` (so the Whisper-only majority gets it); when the Qwen dir exists, keep the current Qwen default. Add `"whisper-streaming"` to the accepted values of the `MEETINGBRO_PREVIEW_ASR_BACKEND` env handling.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_preview_backend_selection.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Verify backend still imports and boots**

Run: `python -c "import meetingbro.main; print('ok')"`
Expected: prints `ok`

- [ ] **Step 7: Commit**

```bash
git add app/backend/meetingbro/main.py app/backend/meetingbro/session/manager.py tests/test_preview_backend_selection.py
git commit -m "feat(asr): select whisper-streaming preview backend (default when no Qwen)"
```

---

### Task 5: Measure the streaming lane on AMI (success criteria gate)

**Files:**
- Modify: `scripts/benchmark_latency_accuracy_frontier.py` (add a `--streaming-preview` mode that enables the whisper-streaming lane in the latency pass and reports committed-caption latency + WER of the committed stream)
- Modify: `scripts/benchmark_accuracy.py` (`_run_latency` already accepts a `preview_asr`; add an `extra_overrides` path is already present — reuse it to enable `fast_preview_enabled=True` with a `StreamingWhisperAdapter`)
- Doc: `docs/benchmarks/streaming-whisper-2026-06.md` (generated result + interpretation)

**Interfaces:**
- Consumes: `StreamingWhisperAdapter` (Task 3), the existing `_run_latency`/`_run_accuracy` runners.
- Produces: a report comparing the streaming preview lane against the formal lane on `ami_en2002a`: committed-caption p50/p95, committed-stream WER, ASR RTF.

- [ ] **Step 1: Add the measurement mode**

In `scripts/benchmark_latency_accuracy_frontier.py`, add a `--streaming-preview` flag. When set, for the baseline knob point build a `StreamingWhisperAdapter(adapter)` and call `ba._run_latency(adapter, wav, preview_asr=streaming_adapter)` (the preview path already records `preview_p50/p95/segments`), and also score the committed preview text's WER by collecting `transcript_preview` committed text over an offline pass. Report a small table: lane (formal vs whisper-streaming), p50, p95, WER, rtf.

- [ ] **Step 2: Run it**

Run (MeetingBro env):
```bash
python scripts/benchmark_latency_accuracy_frontier.py --streaming-preview --pre-vad-max 4 --accumulation 2.0 --out docs/benchmarks/streaming-whisper-2026-06.md
```
Expected: completes, writes the report. Record committed-caption p50/p95, WER, and RTF.

- [ ] **Step 3: Check against success criteria**

Confirm in the report: committed-caption p50 well under the ~3.8 s formal median (target ~1–1.5 s); committed-stream WER not materially worse than the formal lane's (~0.58 on AMI); ASR RTF < 1 (no sustained backpressure). If RTF ≥ 1, note it and reduce cadence / widen the gate in a follow-up (do not block the commit; record the finding).

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark_latency_accuracy_frontier.py scripts/benchmark_accuracy.py docs/benchmarks/streaming-whisper-2026-06.md
git commit -m "bench(asr): measure whisper-streaming preview lane vs formal on AMI"
```

---

## Self-Review

**Spec coverage:** StreamingTranscriber (Task 1), shared-model word decode (Task 2), preview-adapter integration reusing existing plumbing (Task 3), backend selection + default-when-no-Qwen + formal-untouched (Task 4), frontier-harness success-criteria measurement (Task 5). Non-goals (authoritative commits, retiring Qwen, UI changes) are excluded. CPU gating/governor backpressure are runtime config (existing mechanisms) — Task 4 sets the default-on conditions; deeper hardware-gating tuning is a follow-up noted in Task 5 Step 3, not a Phase-1 blocker.

**Placeholder scan:** No TBD/TODO; every code step has complete code. Task 4 Step 1 is an explicit investigation step (read a specific function, make one conditional one-line change) — not a placeholder.

**Type consistency:** `Word(text,start,end)` and `StreamingTranscriber.step/flush/reset/committed_words/committed_until` are defined in Task 1 and used identically in Tasks 2–3. `transcribe_words(...) -> list[Word]` defined in Task 2, consumed in Task 3. `StreamingWhisperAdapter(formal)` defined in Task 3, built in Task 4. `_build_streaming_preview_adapter` defined and tested in Task 4.
