"""Streaming output guard, sync and async.

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

Scan cost is the one number to watch: every scan re-reads the whole buffer, so
scanning per token makes per-token cost grow with response length. Set
`coalesce_chars` to rescan only once per N new characters; nothing is released
before it is scanned, so this trades release granularity, never safety.

`AsyncStreamGuard` is the same state machine awaiting `acheck_output`. It exists
because every framework adapter is async: bridging a sync guard into a running
event loop would either block the loop for the duration of each scan or deadlock
outright. The hold-back logic is shared rather than reimplemented, so the two
cannot drift -- a test asserts they release identical text for identical input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterable, AsyncIterator, Iterable, Iterator, Sequence

from .types import Action, Verdict

__all__ = ["StreamGuard", "AsyncStreamGuard", "StreamChunk"]

_DEFAULT_HOLDBACK = 256


@dataclass(slots=True)
class StreamChunk:
    """One releasable unit. `text` is post-redaction and safe to emit."""

    text: str
    verdict: Verdict | None = None
    blocked: bool = False


class _Window:
    """The hold-back state machine, independent of how a verdict is obtained.

    Sync and async guards differ only in whether they `call` or `await` the
    pipeline, so everything that decides *what to release* lives here.
    """

    __slots__ = ("holdback", "coalesce", "buffer", "released", "unscanned", "blocked", "last")

    def __init__(self, holdback: int, coalesce: int = 0) -> None:
        self.holdback = max(0, holdback)
        self.coalesce = max(0, coalesce)
        self.buffer = ""
        self.released = 0
        self.unscanned = 0
        self.blocked = False
        self.last: Verdict | None = None

    def should_scan(self, *, final: bool) -> bool:
        """Whether enough new text has arrived to justify another full rescan.

        Every scan re-reads the whole buffer -- that is what catches a pattern
        straddling a chunk boundary -- so scanning once per token makes per-token
        cost grow with response length. Coalescing amortises that over `coalesce`
        characters. It cannot leak: nothing is released until it has been
        scanned, so the only effect of waiting is that release is coarser.
        """
        if final or self.coalesce == 0:
            return True
        return self.unscanned >= self.coalesce

    def terminal(self) -> StreamChunk:
        return StreamChunk("", self.last, blocked=True)

    def apply(self, verdict: Verdict, *, final: bool) -> StreamChunk:
        self.last = verdict
        self.unscanned = 0
        text = verdict.sanitized if verdict.sanitized is not None else self.buffer

        if verdict.action is Action.BLOCK:
            self.blocked = True
            return StreamChunk("", verdict, blocked=True)

        cut = len(text) if final else max(0, len(text) - self.holdback)
        if cut <= self.released:
            return StreamChunk("", verdict)
        out = text[self.released : cut]
        self.released = cut
        return StreamChunk(out, verdict)


class StreamGuard:
    """Incremental output enforcement over a token stream.

    Usage::

        guard = StreamGuard(pipeline, session_id="u-1")
        for chunk in guard.feed_all(token_iter):
            if chunk.blocked:
                emit_refusal(chunk.verdict)
                break
            emit(chunk.text)
    """

    __slots__ = ("_pipeline", "_window", "_kwargs")

    def __init__(
        self,
        pipeline,
        *,
        holdback_chars: int = _DEFAULT_HOLDBACK,
        coalesce_chars: int = 0,
        session_id: str | None = None,
        allowed_citations: Sequence[str] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._window = _Window(holdback_chars, coalesce_chars)
        # Passed through on every scan: a stream that skips the session charges
        # no volume to the caller, so per-session output ceilings never fire on
        # exactly the responses that are largest.
        self._kwargs: dict[str, object] = {}
        if session_id is not None:
            self._kwargs["session_id"] = session_id
        if allowed_citations is not None:
            self._kwargs["allowed_citations"] = tuple(allowed_citations)

    @property
    def verdict(self) -> Verdict | None:
        """Verdict from the most recent scan, or None before the first chunk."""
        return self._window.last

    @property
    def blocked(self) -> bool:
        return self._window.blocked

    def feed(self, token: str) -> StreamChunk:
        """Buffer a token and release whatever is now safe to emit."""
        if self._window.blocked:
            return self._window.terminal()
        self._window.buffer += token
        self._window.unscanned += len(token)
        if not self._window.should_scan(final=False):
            return StreamChunk("", self._window.last)
        return self._drain(final=False)

    def finish(self) -> StreamChunk:
        """Flush the hold-back window after generation ends."""
        if self._window.blocked:
            return self._window.terminal()
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
        verdict = self._pipeline.check_output(self._window.buffer, **self._kwargs)
        return self._window.apply(verdict, final=final)


class AsyncStreamGuard:
    """`StreamGuard` for async token streams; identical hold-back semantics.

    Usage::

        guard = AsyncStreamGuard(pipeline, session_id="u-1")
        async for chunk in guard.feed_all(atoken_iter):
            if chunk.blocked:
                await emit_refusal(chunk.verdict)
                break
            await emit(chunk.text)
    """

    __slots__ = ("_pipeline", "_window", "_kwargs")

    def __init__(
        self,
        pipeline,
        *,
        holdback_chars: int = _DEFAULT_HOLDBACK,
        coalesce_chars: int = 0,
        session_id: str | None = None,
        allowed_citations: Sequence[str] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._window = _Window(holdback_chars, coalesce_chars)
        self._kwargs: dict[str, object] = {}
        if session_id is not None:
            self._kwargs["session_id"] = session_id
        if allowed_citations is not None:
            self._kwargs["allowed_citations"] = tuple(allowed_citations)

    @property
    def verdict(self) -> Verdict | None:
        return self._window.last

    @property
    def blocked(self) -> bool:
        return self._window.blocked

    async def feed(self, token: str) -> StreamChunk:
        if self._window.blocked:
            return self._window.terminal()
        self._window.buffer += token
        self._window.unscanned += len(token)
        if not self._window.should_scan(final=False):
            return StreamChunk("", self._window.last)
        return await self._drain(final=False)

    async def finish(self) -> StreamChunk:
        if self._window.blocked:
            return self._window.terminal()
        return await self._drain(final=True)

    async def feed_all(
        self, tokens: AsyncIterable[str] | Iterable[str]
    ) -> AsyncIterator[StreamChunk]:
        """Guard an async (or plain) iterable of tokens.

        Accepts both so a caller can pass a model's async stream or a list in a
        test without a second code path.
        """
        if isinstance(tokens, AsyncIterable):
            iterator = tokens.__aiter__()
            try:
                async for token in iterator:
                    chunk = await self.feed(token)
                    if chunk.text or chunk.blocked:
                        yield chunk
                    if chunk.blocked:
                        return
            finally:
                # A blocked stream abandons the upstream iterator mid-flight;
                # closing it releases the generator (and whatever it holds) now
                # rather than at GC time on some later event loop.
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
        else:
            for token in tokens:
                chunk = await self.feed(token)
                if chunk.text or chunk.blocked:
                    yield chunk
                if chunk.blocked:
                    return
        tail = await self.finish()
        if tail.text or tail.blocked:
            yield tail

    # ---------------------------------------------------------------- internal
    async def _drain(self, *, final: bool) -> StreamChunk:
        verdict = await self._pipeline.acheck_output(self._window.buffer, **self._kwargs)
        return self._window.apply(verdict, final=final)
