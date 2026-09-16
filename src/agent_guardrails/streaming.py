"""Streaming output guard.

Token streaming breaks a whole-response scanner in two ways. A secret is emitted
to the user before the response is complete, so a post-hoc verdict is a
notification rather than a control; and a pattern that straddles two chunks is
invisible to per-chunk scanning.

`StreamGuard` solves both with a hold-back window. Text is buffered and only the
prefix older than `holdback_chars` is released, so a rule spanning a chunk
boundary is still matched intact before anything is emitted. Redaction rewrites
the buffered text; a BLOCK stops the stream and reports the terminal verdict.

The cost is deliberate and explicit: the client sees output `holdback_chars`
behind generation. That is the price of enforcing rather than observing, and it
is the reason the window is configurable per call site instead of hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .config import GuardrailConfig
from .types import Action, Layer, Verdict

__all__ = ["StreamGuard", "StreamChunk"]

_DEFAULT_HOLDBACK = 256


@dataclass(slots=True)
class StreamChunk:
    """One releasable unit. `text` is post-redaction and safe to emit."""

    text: str
    verdict: Verdict | None = None
    blocked: bool = False


class StreamGuard:
    """Incremental output enforcement over a token stream.

    Usage::

        guard = StreamGuard(pipeline)
        for chunk in guard.feed_all(token_iter):
            if chunk.blocked:
                emit_refusal(chunk.verdict)
                break
            emit(chunk.text)
    """

    __slots__ = ("_pipeline", "_holdback", "_buffer", "_released", "_blocked", "_last")

    def __init__(self, pipeline, *, holdback_chars: int = _DEFAULT_HOLDBACK) -> None:
        self._pipeline = pipeline
        self._holdback = max(0, holdback_chars)
        self._buffer = ""
        self._released = 0
        self._blocked = False
        self._last: Verdict | None = None

    @property
    def verdict(self) -> Verdict | None:
        """Verdict from the most recent scan, or None before the first chunk."""
        return self._last

    @property
    def blocked(self) -> bool:
        return self._blocked

    def feed(self, token: str) -> StreamChunk:
        """Buffer a token and release whatever is now safe to emit."""
        if self._blocked:
            return StreamChunk("", self._last, blocked=True)
        self._buffer += token
        return self._drain(final=False)

    def finish(self) -> StreamChunk:
        """Flush the hold-back window after generation ends."""
        if self._blocked:
            return StreamChunk("", self._last, blocked=True)
        return self._drain(final=True)

    def feed_all(self, tokens: Iterable[str]) -> Iterator[StreamChunk]:
        for token in tokens:
            chunk = self.feed(token)
            if chunk.text or chunk.blocked:
                yield chunk
            if chunk.blocked:
                return
        tail = self.finish()
        if tail.text or tail.blocked:
            yield tail

    # ---------------------------------------------------------------- internal
    def _drain(self, *, final: bool) -> StreamChunk:
        # Rescan the whole buffer, not just the new token: that is what catches a
        # pattern straddling a chunk boundary. Buffers are bounded by the
        # response length, and the pipeline result cache absorbs repeat work.
        verdict = self._pipeline.check_output(self._buffer)
        self._last = verdict
        text = verdict.sanitized if verdict.sanitized is not None else self._buffer

        if verdict.action is Action.BLOCK:
            self._blocked = True
            return StreamChunk("", verdict, blocked=True)

        cut = len(text) if final else max(0, len(text) - self._holdback)
        if cut <= self._released:
            return StreamChunk("", verdict)
        out = text[self._released : cut]
        self._released = cut
        return StreamChunk(out, verdict)
