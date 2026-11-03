"""Streaming enforcement.

A post-hoc scan of a streamed response is a notification, not a control: the
secret already reached the user. These tests pin the two properties that make
streaming enforcement real -- nothing is released before it has been scanned, and
a pattern straddling a chunk boundary is still caught.
"""
from __future__ import annotations

from charterguard import GuardrailPipeline
from charterguard.streaming import StreamGuard
from charterguard.types import Action, Layer, Verdict

SECRET = "sk_live_4eC39HqLyjWDarjtT1zdp7dc"


class BlockingPipeline:
    """Duck-typed stand-in: StreamGuard only needs check_output."""

    def check_output(self, text: str, **_) -> Verdict:
        return Verdict(layer=Layer.OUTPUT, action=Action.BLOCK, score=1.0)


def _tokens(text: str, size: int = 7):
    return [text[i : i + size] for i in range(0, len(text), size)]


def test_pii_split_across_chunks_is_redacted_not_emitted():
    """Per-chunk scanning misses a value that straddles a boundary; the
    hold-back window rescans the buffer, so the redaction still lands."""
    guard = StreamGuard(GuardrailPipeline(), holdback_chars=64)
    text = "please contact a.b@corp.example.com about the invoice"
    released = "".join(c.text for c in guard.feed_all(_tokens(text)))
    assert "a.b@corp.example.com" not in released
    assert "[REDACTED" in released


def test_critical_secret_split_across_chunks_terminates_the_stream():
    """A live credential at the output boundary is BLOCK, not redact -- so the
    stream stops and the user never receives the partial key either."""
    guard = StreamGuard(GuardrailPipeline(), holdback_chars=64)
    chunks = list(guard.feed_all(_tokens(f"key is {SECRET} ok")))
    released = "".join(c.text for c in chunks)
    assert SECRET not in released
    assert guard.blocked


def test_nothing_is_released_before_it_is_scanned():
    guard = StreamGuard(GuardrailPipeline(), holdback_chars=32)
    emitted = ""
    for token in _tokens("The credential is " + SECRET):
        emitted += guard.feed(token).text
        assert SECRET not in emitted
    emitted += guard.finish().text
    assert SECRET not in emitted


def test_clean_text_is_released_in_full():
    guard = StreamGuard(GuardrailPipeline(), holdback_chars=16)
    text = "The quarterly revenue grew by four percent across all regions."
    out = "".join(c.text for c in guard.feed_all(_tokens(text)))
    assert out == text


def test_holdback_delays_release_until_finish():
    guard = StreamGuard(GuardrailPipeline(), holdback_chars=1_000)
    text = "a short clean answer"
    mid = "".join(guard.feed(t).text for t in _tokens(text))
    assert mid == ""  # everything still inside the window
    assert guard.finish().text == text


def test_block_terminates_the_stream():
    guard = StreamGuard(BlockingPipeline(), holdback_chars=8)
    chunks = list(guard.feed_all(_tokens("anything at all here")))
    assert chunks and chunks[-1].blocked
    assert guard.blocked
    assert all(c.text == "" for c in chunks)


def test_feeding_after_a_block_stays_blocked():
    guard = StreamGuard(BlockingPipeline())
    guard.feed("x")
    assert guard.feed("more").blocked
    assert guard.finish().blocked


def test_no_double_release_of_the_same_span():
    guard = StreamGuard(GuardrailPipeline(), holdback_chars=4)
    text = "one two three four five six seven"
    out = "".join(c.text for c in guard.feed_all(_tokens(text, 3)))
    assert out == text


def test_empty_stream_is_safe():
    guard = StreamGuard(GuardrailPipeline())
    assert list(guard.feed_all([])) == []
