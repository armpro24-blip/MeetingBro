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


def test_window_duration_drop_resets_committer_for_new_utterance():
    # Simulate: first call uses a long buffer (5 s window), then the formal lane
    # commits and the window restarts with a short buffer (0.1 s) — that drop
    # exceeds the 0.25 s threshold and must reset LocalAgreement state.
    formal = _ScriptedFormal([
        [("the", 0.0, 0.2), ("cat", 0.2, 0.5)],
        [("dog", 0.0, 0.3)],   # new utterance after formal commit, window-relative times
    ])
    a = StreamingWhisperAdapter(formal)
    long_samples = np.zeros(5 * 16_000, dtype=np.float32)   # 5 s
    short_samples = np.zeros(int(0.1 * 16_000), dtype=np.float32)  # 0.1 s  (drop > 0.25 s)
    a.transcribe(long_samples, 16_000, offset_seconds=0.0)
    s = a.transcribe(short_samples, 16_000, offset_seconds=0.0)
    assert s[0].text == "dog"                        # not "the cat dog"


def test_empty_decode_returns_no_segment():
    formal = _ScriptedFormal([[]])
    a = StreamingWhisperAdapter(formal)
    assert a.transcribe(_samples(), 16_000, offset_seconds=0.0) == []
