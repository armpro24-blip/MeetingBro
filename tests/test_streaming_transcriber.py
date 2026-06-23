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
