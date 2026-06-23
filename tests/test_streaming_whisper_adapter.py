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
