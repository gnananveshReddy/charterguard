"""Guard logic shared by the LangChain and LangGraph adapters.

Private module, deliberately free of third-party imports. The LangChain adapter
is a thin typed shell over these functions, which means the interesting
behaviour -- what gets blocked, what gets redacted, what the envelope looks like
-- is unit-testable on plain dicts, with no optional package installed.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, AsyncIterable, AsyncIterator, Iterable, Mapping, Sequence

from ..exceptions import GuardrailBlocked
from ..pipeline import GuardrailPipeline
from ..streaming import AsyncStreamGuard, StreamChunk
from ..types import Action, Layer, Verdict
from ._lc_common import (
    document_source,
    document_text,
    last_message,
    message_text,
    replace_document_text,
    replace_message_text,
)

RAISE = "raise"
REFUSE = "refuse"
_TEXT_KEYS = ("input", "question", "query", "text", "message")


def merge_scope(verdict: Verdict, scope: Verdict) -> Verdict:
    """Fold a scope verdict into an input verdict, keeping both sets of findings."""
    if scope.action is Action.ALLOW:
        return verdict
    return replace(
        verdict,
        action=scope.action if scope.action is Action.BLOCK else verdict.action,
        findings=verdict.findings + scope.findings,
        score=max(verdict.score, scope.score),
        policy_id=scope.policy_id,
    )


def _is_message_mapping(payload: Mapping[str, Any]) -> bool:
    """True for a message-shaped dict, as distinct from a dict of chain variables.

    Both arrive as mappings, and the difference matters: a message carries its
    text under `content`, while chain variables carry it under `input` or
    `question`. Guessing wrong means scanning an empty string.
    """
    return "content" in payload and ("role" in payload or "type" in payload)


def payload_text(payload: Any, *, roles: tuple[str, ...] = ("human",)) -> str:
    """Pull scannable text out of whatever a LangChain chain passes along.

    Handles the four shapes that actually flow through a chain: a bare string, a
    single message, a message list, and a dict of chain variables. `PromptValue`
    is covered by duck-typing `to_messages`, so no import is needed.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    to_messages = getattr(payload, "to_messages", None)
    if callable(to_messages):
        return message_text(last_message(to_messages(), roles=roles)) or ""
    if isinstance(payload, Mapping):
        if _is_message_mapping(payload):
            return message_text(payload)
        for key in _TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                return value
            if value is not None and not isinstance(value, (int, float, bool)):
                return payload_text(value, roles=roles)
        return ""
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray)):
        return message_text(last_message(payload, roles=roles)) or ""
    return message_text(payload)


def rewrite_payload(payload: Any, text: str, *, roles: tuple[str, ...] = ("human",)) -> Any:
    """Return `payload` with its guarded text substituted, shape preserved."""
    if payload is None or isinstance(payload, str):
        return text
    if isinstance(payload, Mapping):
        if _is_message_mapping(payload):
            return replace_message_text(payload, text)
        for key in _TEXT_KEYS:
            if key in payload:
                clone = dict(payload)
                value = payload[key]
                clone[key] = (
                    text if isinstance(value, str) else rewrite_payload(value, text, roles=roles)
                )
                return clone
        return payload
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray)):
        target = last_message(payload, roles=roles)
        if target is None:
            return payload
        out = list(payload)
        for idx in range(len(out) - 1, -1, -1):
            if out[idx] is target:
                out[idx] = replace_message_text(target, text)
                break
        return out
    return replace_message_text(payload, text)


def _enforce(verdict: Verdict, pipeline: GuardrailPipeline, on_block: str) -> str | None:
    """Raise or return refusal text. `None` means the caller should continue."""
    if not verdict.blocked:
        return None
    if on_block == RAISE:
        raise GuardrailBlocked(verdict, pipeline.refusal(verdict))
    return pipeline.refusal(verdict)


async def guard_input_payload(
    pipeline: GuardrailPipeline,
    payload: Any,
    *,
    session_id: str = "default",
    enforce_scope: bool = False,
    on_block: str = RAISE,
) -> tuple[Any, Verdict, str | None]:
    """Scan a chain's inbound payload. Returns (payload, verdict, refusal)."""
    text = payload_text(payload)
    verdict = await pipeline.acheck_input(text, session_id=session_id)
    if enforce_scope and not verdict.blocked:
        verdict = merge_scope(verdict, await pipeline.acheck_scope(text))
    refusal = _enforce(verdict, pipeline, on_block)
    if refusal is not None:
        return payload, verdict, refusal
    if verdict.sanitized is not None and verdict.sanitized != text:
        payload = rewrite_payload(payload, verdict.sanitized)
    return payload, verdict, None


async def guard_output_payload(
    pipeline: GuardrailPipeline,
    payload: Any,
    *,
    session_id: str = "default",
    allowed_citations: Sequence[str] | None = None,
    on_block: str = REFUSE,
) -> tuple[Any, Verdict, str | None]:
    """Final gate on a chain's model output.

    Defaults to `on_block="refuse"` rather than raising: by this point the user
    is waiting on a response, and a rendered refusal is a better answer than a
    500 from an uncaught exception.
    """
    text = payload_text(payload, roles=("ai",))
    verdict = await pipeline.acheck_output(
        text,
        session_id=session_id,
        allowed_citations=None if allowed_citations is None else tuple(allowed_citations),
    )
    refusal = _enforce(verdict, pipeline, on_block)
    if refusal is not None:
        return rewrite_payload(payload, refusal, roles=("ai",)), verdict, refusal
    if verdict.sanitized is not None and verdict.sanitized != text:
        payload = rewrite_payload(payload, verdict.sanitized, roles=("ai",))
    return payload, verdict, None


async def guard_tool_arguments(
    pipeline: GuardrailPipeline,
    tool_name: str,
    arguments: Any,
    *,
    session_id: str = "default",
    on_block: str = RAISE,
) -> tuple[Verdict, str | None]:
    """Charter + argument check before a tool runs."""
    verdict = pipeline.check_tool_call(tool_name, arguments, session_id=session_id)
    return verdict, _enforce(verdict, pipeline, on_block)


async def guard_tool_result(
    pipeline: GuardrailPipeline,
    output: Any,
    *,
    tool_name: str,
    session_id: str = "default",
    redact_secrets: bool = True,
) -> tuple[str, Verdict, bool]:
    """Envelope-wrap a tool's return value. Returns (text, verdict, quarantined)."""
    result = await pipeline.ainspect_tool_output(
        output if isinstance(output, str) else str(output),
        tool_name=tool_name,
        session_id=session_id,
        redact_secrets=redact_secrets,
    )
    return result.context_block(), result.verdict, result.quarantined


async def guard_documents(
    pipeline: GuardrailPipeline,
    docs: Sequence[Any],
    *,
    session_id: str = "default",
    drop_quarantined: bool = True,
    redact_secrets: bool = True,
) -> tuple[list[Any], Verdict, dict[str, int]]:
    """Envelope-wrap retrieved documents. Returns (kept, worst verdict, counts).

    Quarantined documents are dropped by default: unlike a tool result, a
    dropped document breaks no provider contract, and keeping a chunk that
    carries instructions is the whole risk being guarded against.
    """
    kept: list[Any] = []
    worst: Verdict | None = None
    dropped = 0
    for doc in docs:
        result = await pipeline.ainspect_untrusted(
            document_text(doc),
            kind="retrieval",
            source=document_source(doc),
            session_id=session_id,
            trust="none",
            redact_sensitive=redact_secrets,
        )
        if worst is None and result.verdict.action is not Action.ALLOW:
            worst = result.verdict
        if result.quarantined and drop_quarantined:
            dropped += 1
            continue
        kept.append(replace_document_text(doc, result.context_block()))
    verdict = worst if worst is not None else Verdict(layer=Layer.UNTRUSTED, action=Action.ALLOW)
    counts = {
        "documents_in": len(docs),
        "documents_kept": len(kept),
        "documents_dropped": dropped,
    }
    return kept, verdict, counts


async def guard_text_stream(
    pipeline: GuardrailPipeline,
    chunks: AsyncIterable[Any] | Iterable[Any],
    *,
    session_id: str = "default",
    holdback_chars: int = 256,
    coalesce_chars: int = 32,
    allowed_citations: Sequence[str] | None = None,
    on_block: str = REFUSE,
) -> AsyncIterator[StreamChunk]:
    """Guard a stream of text (or message) chunks incrementally.

    Yields `StreamChunk`s whose `text` is already redacted and safe to emit. The
    hold-back window means output trails generation by `holdback_chars`, which is
    what buys the ability to catch a secret spanning two chunks *before* it is
    emitted rather than after.

    On a BLOCK this yields one terminal chunk carrying the refusal (or raises
    under `on_block="raise"`) and stops consuming the upstream stream.

    Note the limit honestly: text already released cannot be recalled. The window
    bounds how much unscanned text can exist, not how much already-sent text can
    be retracted, so a larger window trades latency for a tighter bound.

    `coalesce_chars` defaults to 32 here rather than 0, because every scan
    re-reads the whole buffer: scanning per token made per-token cost grow with
    response length (measured >100x worse on a 512-token response than scanning
    per 32 characters). Nothing is released before it is scanned, so this costs
    release granularity, not safety.
    """
    guard = AsyncStreamGuard(
        pipeline,
        holdback_chars=holdback_chars,
        coalesce_chars=coalesce_chars,
        session_id=session_id,
        allowed_citations=allowed_citations,
    )

    async def _iter() -> AsyncIterator[str]:
        if isinstance(chunks, AsyncIterable):
            async for chunk in chunks:
                yield _chunk_text(chunk)
        else:
            for chunk in chunks:
                yield _chunk_text(chunk)

    # Close both generators deterministically. Left to the garbage collector,
    # an async generator abandoned mid-iteration (a BLOCK, a raise, or a consumer
    # that breaks early) is finalised later on an arbitrary loop, which logs
    # "Task was destroyed but it is pending" and can outlive the request it
    # belonged to.
    source = _iter()
    stream = guard.feed_all(source)
    try:
        async for out in stream:
            if out.blocked:
                verdict = out.verdict
                if verdict is not None and on_block == RAISE:
                    raise GuardrailBlocked(verdict, pipeline.refusal(verdict))
                refusal = pipeline.refusal(verdict) if verdict is not None else ""
                yield StreamChunk(refusal, verdict, blocked=True)
                return
            yield out
    finally:
        await stream.aclose()
        await source.aclose()


def _chunk_text(chunk: Any) -> str:
    """Text of a streamed chunk, whether it is a str, a dict, or a message chunk."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, tuple) and chunk:
        # LangGraph's stream_mode="messages" yields (chunk, metadata).
        return _chunk_text(chunk[0])
    return message_text(chunk)
