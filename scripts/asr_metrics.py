"""Shared ASR accuracy metrics for MeetingBro benchmarks.

WER (word error rate) for space-delimited / Latin-script languages and CER
(character error rate) for CJK, both built on a small O(m·n)/O(n)-space
Levenshtein. Text is normalised the same way for reference and hypothesis:
lower-cased, punctuation stripped, whitespace collapsed.

This consolidates the inline implementation that previously lived in
``benchmark_qwen3_offline_capability.py`` so every benchmark reports comparable
numbers. Run this file directly to execute the self-test:

    python scripts/asr_metrics.py
"""
from __future__ import annotations

import re

# Unicode \w keeps CJK ideographs, so this strips punctuation while preserving
# both Latin words and CJK characters.
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

_CJK_RE = re.compile(
    r"[一-鿿㐀-䶿豈-﫿"  # Han
    r"぀-ゟ゠-ヿ]"               # Hiragana / Katakana
)


def has_cjk(text: str) -> bool:
    """True if the text contains any CJK / kana character."""
    return bool(_CJK_RE.search(text))


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two sequences. O(m·n) time, O(n) space."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def _words(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.lower()).split()


def _chars(text: str) -> list[str]:
    return [c for c in _PUNCT.sub("", text.lower()) if not c.isspace()]


def wer(ref: str, hyp: str) -> float:
    """Word error rate. Empty ref → 0.0 if hyp also empty else 1.0."""
    ref_w = _words(ref)
    hyp_w = _words(hyp)
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return edit_distance(ref_w, hyp_w) / len(ref_w)


def cer(ref: str, hyp: str) -> float:
    """Character error rate (whitespace ignored). Empty ref → 0.0/1.0."""
    ref_c = _chars(ref)
    hyp_c = _chars(hyp)
    if not ref_c:
        return 0.0 if not hyp_c else 1.0
    return edit_distance(ref_c, hyp_c) / len(ref_c)


_CJK_LANGS = {"zh", "ja", "yue", "zh-cn", "zh-tw"}


def metric_for(language: str | None, reference: str = "") -> str:
    """Pick "cer" for CJK content else "wer".

    When ``language`` is given it is authoritative (CJK languages → cer, any
    other language → wer); the manifest is the source of truth. When it is
    ``None`` the script of the reference text decides.
    """
    if language:
        return "cer" if language.lower() in _CJK_LANGS else "wer"
    return "cer" if has_cjk(reference) else "wer"


def score(ref: str, hyp: str, *, language: str | None = None) -> tuple[str, float]:
    """Return ``(metric_name, value)`` using the appropriate metric."""
    name = metric_for(language, ref)
    value = cer(ref, hyp) if name == "cer" else wer(ref, hyp)
    return name, value


def _selftest() -> None:
    # Exact match → 0.
    assert wer("the cat sat", "the cat sat") == 0.0
    assert cer("你好世界", "你好世界") == 0.0

    # One substitution out of three words → 1/3.
    assert abs(wer("the cat sat", "the dog sat") - 1 / 3) < 1e-9

    # One deletion out of three words → 1/3.
    assert abs(wer("the cat sat", "the cat") - 1 / 3) < 1e-9

    # One insertion out of three words → 1/3.
    assert abs(wer("the cat sat", "the cat sat down") - 1 / 3) < 1e-9

    # Punctuation and case are normalised away.
    assert wer("Hello, world!", "hello world") == 0.0

    # One wrong character out of four → 1/4.
    assert abs(cer("你好世界", "你好世节") - 1 / 4) < 1e-9

    # Empty reference handling.
    assert wer("", "") == 0.0
    assert wer("", "extra") == 1.0
    assert cer("", "") == 0.0

    # Script / language routing.
    assert metric_for(None, "你好") == "cer"
    assert metric_for(None, "hello") == "wer"
    assert metric_for("zh", "hello") == "cer"  # explicit language wins
    assert metric_for("en", "你好") == "wer"

    name, value = score("你好世界", "你好世节", language="zh")
    assert name == "cer" and abs(value - 0.25) < 1e-9

    print("asr_metrics self-test: OK")


if __name__ == "__main__":
    _selftest()
