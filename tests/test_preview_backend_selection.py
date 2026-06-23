from meetingbro.asr.faster_whisper_adapter import FasterWhisperAdapter
from meetingbro.asr.streaming_whisper_adapter import StreamingWhisperAdapter
from meetingbro.main import _build_streaming_preview_adapter


def test_build_streaming_preview_wraps_formal_adapter():
    formal = FasterWhisperAdapter(model_size="tiny", device="cpu")
    preview = _build_streaming_preview_adapter(formal)
    assert isinstance(preview, StreamingWhisperAdapter)
    assert preview._formal is formal  # shares the formal model, no second load
