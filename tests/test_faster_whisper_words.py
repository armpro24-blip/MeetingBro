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
