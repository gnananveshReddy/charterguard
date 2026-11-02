"""LangGraph wiring: guard nodes, a router, and thread-scoped sessions.

This module imports no third-party package, and that is a design choice rather
than an accident. A LangGraph node is any callable over the state dict, an edge
condition is any callable returning a node name, and messages coerce from
`{"role": ..., "content": ...}` dicts. Nothing here needs a `langgraph` symbol,
so these adapters work across langgraph versions and can be unit-tested without
building a graph. Where `langchain-core` happens to be installed, real
`AIMessage`/`ToolMessage` objects are emitted instead of dicts.

Wiring, smallest useful graph::

    from charterguard.integrations.langgraph import (
        guard_input_node, guard_output_node, guardrail_router, refusal_node,
    )

    g = StateGraph(State)
    g.add_node("guard_in", guard_input_node(pipeline))
    g.add_node("model", call_model)
    g.add_node("guard_out", guard_output_node(pipeline))
    g.add_node("refuse", refusal_node())
    g.set_entry_point("guard_in")
    g.add_conditional_edges("guard_in", guardrail_router(ok="model", blocked="refuse"))
    g.add_edge("model", "guard_out")

Your state must carry the key the guards write (`guardrail` by default) and, for
the output guard, the key they publish safe text to (`safe_output`)::

    class State(MessagesState, GuardrailState):
        pass
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping, Sequence, TypedDict

from ..pipeline import GuardrailPipeline
from ..types import Action, Verdict
from ._lc_guards import merge_scope
from ._lc_common import (
    last_message,
    message_id,
    message_text,
    replace_message_text,
    tool_call_id,
    tool_calls,
    tool_message,
    tool_name,
    trailing_messages,
)

__all__ = [
    "GUARDRAIL_KEY",
    "GuardrailState",
    "approval_node",
    "guard_input_node",
    "guard_output_node",
    "guard_message_stream",
    "guard_retrieval_node",
    "guard_tool_calls_node",
    "guard_tool_results_node",
    "guardrail_record",
    "guardrail_router",
    "refusal_node",
    "thread_session_id",
]

GUARDRAIL_KEY = "guardrail"
SAFE_OUTPUT_KEY = "safe_output"
_DEFAULT_SESSION = "default"


class GuardrailState(TypedDict, total=False):
    """Mix into your state so the guard nodes have somewhere to write.

    `guardrail` holds the record from the most recent guard to run; the router
    reads it. `safe_output` holds the text the output guard cleared for the
    user, which is deliberately not the message history -- see
    `guard_output_node`.
    """

    guardrail: dict[str, Any]
    safe_output: str


def thread_session_id(config: Any, default: str = _DEFAULT_SESSION) -> str:
    """Map a LangGraph thread to a guardrail session.

    Worth wiring: the ledger enforces per-session volume ceilings (tool calls,
    cumulative output, injection attempts). Left unset, every thread on the
    process shares the `"default"` session, so ceilings become global -- one
    busy conversation trips the limit for everyone, and a per-user abuse signal
    is averaged away across users. `thread_id` is the identifier LangGraph
    already persists checkpoints under, which makes it the correct join key.
    """
    if not isinstance(config, Mapping):
        return default
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        return default
    for key in ("session_id", "thread_id"):
        value = configurable.get(key)
        if isinstance(value, str) and value:
            return value
    return default


def guardrail_record(
    verdict: Verdict, stage: str, pipeline: GuardrailPipeline
) -> dict[str, Any]:
    """The dict a guard node writes to state.

    Carries the rendered refusal so the refusal node needs no pipeline handle,
    and the audit row so a checkpointed thread keeps its evidence. Serialisable
    on purpose: LangGraph checkpointers persist state, and a `Verdict` in state
    would force every backend to handle a custom type.
    """
    return {
        "stage": stage,
        "layer": verdict.layer.value,
        "action": verdict.action.value,
        "blocked": verdict.blocked,
        "flagged": verdict.action is Action.FLAG,
        "policy_id": verdict.policy_id,
        "score": verdict.score,
        "categories": [c.value for c in verdict.categories],
        "refusal": pipeline.refusal(verdict) if verdict.blocked else None,
        "audit": verdict.to_audit(),
    }


def guard_input_node(
    pipeline: GuardrailPipeline,
    *,
    messages_key: str = "messages",
    state_key: str = GUARDRAIL_KEY,
    enforce_scope: bool = False,
    session_id: str | None = None,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Scan the newest human turn. Writes a record; blocks nothing by itself.

    Enforcement is the router's job, so that a phased rollout can log FLAG and
    BLOCK outcomes for a release before any traffic is refused.
    """

    async def _node(state: Mapping[str, Any], config: Any = None) -> dict[str, Any]:
        messages = state.get(messages_key) or []
        text = message_text(last_message(messages, roles=("human",)))
        sid = session_id or thread_session_id(config)
        verdict = await pipeline.acheck_input(text, session_id=sid)
        if enforce_scope and not verdict.blocked:
            verdict = merge_scope(verdict, await pipeline.acheck_scope(text))
        return {state_key: guardrail_record(verdict, "input", pipeline)}

    return _node


def guard_output_node(
    pipeline: GuardrailPipeline,
    *,
    messages_key: str = "messages",
    state_key: str = GUARDRAIL_KEY,
    output_key: str = SAFE_OUTPUT_KEY,
    rewrite: bool = False,
    allowed_citations: Sequence[str] | None = None,
    session_id: str | None = None,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Final gate on the newest assistant turn.

    Publishes the text that is safe to send to `output_key` -- redacted where
    the scan redacted, the refusal where the verdict blocks -- and does *not*
    touch message history by default. That split is deliberate: what the user
    may see and what the model should remember are different questions, and
    rewriting history silently changes the transcript a later turn reasons over.
    Pass `rewrite=True` when you do want history to hold only the safe text.
    """

    async def _node(state: Mapping[str, Any], config: Any = None) -> dict[str, Any]:
        messages = state.get(messages_key) or []
        msg = last_message(messages, roles=("ai",))
        text = message_text(msg)
        sid = session_id or thread_session_id(config)
        verdict = await pipeline.acheck_output(
            text,
            session_id=sid,
            allowed_citations=None if allowed_citations is None else tuple(allowed_citations),
        )
        safe = (
            pipeline.refusal(verdict)
            if verdict.blocked
            else (verdict.sanitized if verdict.sanitized is not None else text)
        )
        update: dict[str, Any] = {
            state_key: guardrail_record(verdict, "output", pipeline),
            output_key: safe,
        }
        if rewrite and msg is not None and safe != text:
            update[messages_key] = [_rewritten(msg, safe)]
        return update

    return _node


def _rewritten(msg: Any, text: str) -> Any:
    """Replacement message for an in-place state update.

    `add_messages` merges by id: same id replaces, missing id appends. Returning
    a replacement without an id would leave the unredacted original in history
    *and* append the safe copy, which is the opposite of the intent -- so this
    refuses rather than guessing.
    """
    if message_id(msg) is None:
        raise ValueError(
            "cannot rewrite a message with no id: add_messages replaces by id, so a "
            "replacement without one would append and leave the original in history. "
            "Build state with the add_messages reducer (it assigns ids), or use the "
            "default rewrite=False and send state['safe_output'] to the user."
        )
    return replace_message_text(msg, text)


def guard_tool_calls_node(
    pipeline: GuardrailPipeline,
    *,
    messages_key: str = "messages",
    state_key: str = GUARDRAIL_KEY,
    session_id: str | None = None,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Charter + argument check on the tool calls the model just requested.

    Place between the model and `ToolNode`. When every call passes, this returns
    only a record and the graph proceeds to the tools.

    When any call is refused, *all* calls in the message are short-circuited
    with refusal `ToolMessage`s. That is not over-reach: the chat completions
    contract requires exactly one tool result per `tool_call_id`, and `ToolNode`
    executes every call on the message rather than a subset -- so a partial
    block would either leave an id unanswered (provider 400 on the next turn) or
    execute the refused call anyway. Answering all of them keeps the transcript
    valid and lets the model recover on its own, instead of ending the run.
    """

    async def _node(state: Mapping[str, Any], config: Any = None) -> dict[str, Any]:
        messages = state.get(messages_key) or []
        msg = last_message(messages, roles=("ai",))
        calls = tool_calls(msg)
        sid = session_id or thread_session_id(config)
        if not calls:
            return {}

        worst: Verdict | None = None
        refused: list[str] = []
        for call in calls:
            verdict = pipeline.check_tool_call(call["name"], call["args"], session_id=sid)
            if verdict.blocked:
                refused.append(call["name"])
                if worst is None:
                    worst = verdict
            elif worst is None and verdict.action is Action.FLAG:
                worst = verdict
        if worst is None:
            return {}

        record = guardrail_record(worst, "tool_call", pipeline)
        record["refused_tools"] = refused
        if not refused:
            return {state_key: record}

        note = record["refusal"] or "This tool call was refused by policy."
        return {
            state_key: record,
            messages_key: [
                tool_message(note, call["id"], call["name"]) for call in calls
            ],
        }

    return _node


def guard_tool_results_node(
    pipeline: GuardrailPipeline,
    *,
    messages_key: str = "messages",
    state_key: str = GUARDRAIL_KEY,
    redact_secrets: bool = True,
    session_id: str | None = None,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Treat tool results as untrusted input before they reach the context.

    Place directly after `ToolNode`. Each result from the latest step is scanned,
    secrets are redacted, and the content is replaced by the provenance envelope
    so instructions embedded in a tool result read as data rather than as turns.

    Quarantined results are replaced by the envelope's note, never dropped: a
    missing `tool_call_id` breaks the one-result-per-call contract, so removal
    would trade a prompt-injection risk for a hard provider error.
    """

    async def _node(state: Mapping[str, Any], config: Any = None) -> dict[str, Any]:
        messages = state.get(messages_key) or []
        pending = trailing_messages(messages, roles=("tool",))
        if not pending:
            return {}
        sid = session_id or thread_session_id(config)

        updates: list[Any] = []
        worst: Verdict | None = None
        quarantined: list[str] = []
        for msg in pending:
            result = await pipeline.ainspect_tool_output(
                message_text(msg),
                tool_name=tool_name(msg),
                session_id=sid,
                redact_secrets=redact_secrets,
            )
            if result.quarantined:
                quarantined.append(tool_name(msg))
            if worst is None and result.verdict.action is not Action.ALLOW:
                worst = result.verdict
            body = result.context_block()
            if body != message_text(msg):
                updates.append(_tool_replacement(msg, body))

        record = guardrail_record(
            worst if worst is not None else _clean_verdict(pending, pipeline),
            "tool_result",
            pipeline,
        )
        record["quarantined_tools"] = quarantined
        # A quarantined tool result is not a refusal to the user: the turn
        # continues with the content neutralised, so `blocked` stays false and
        # the router does not divert to a refusal.
        record["blocked"] = False
        record["refusal"] = None
        out: dict[str, Any] = {state_key: record}
        if updates:
            out[messages_key] = updates
        return out

    return _node


def _clean_verdict(pending: list[Any], pipeline: GuardrailPipeline) -> Verdict:
    """An explicit ALLOW so a clean step still leaves an auditable record."""
    from ..types import Layer

    return Verdict(layer=Layer.TOOL_OUTPUT, action=Action.ALLOW, meta={"results": len(pending)})


def _tool_replacement(msg: Any, body: str) -> Any:
    """Replace a tool result's body, keeping the id and the tool_call_id linkage."""
    if message_id(msg) is None:
        raise ValueError(
            "tool result has no id, so it cannot be replaced in place (add_messages "
            "merges by id). Build state with the add_messages reducer, which assigns "
            "ids to messages as they enter state."
        )
    return replace_message_text(msg, body)


def guard_retrieval_node(
    pipeline: GuardrailPipeline,
    *,
    docs_key: str = "documents",
    state_key: str = GUARDRAIL_KEY,
    drop_quarantined: bool = True,
    redact_secrets: bool = True,
    session_id: str | None = None,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Inspect retrieved documents before they are formatted into a prompt.

    This is the layer that most needs guarding in a RAG graph: a document is
    attacker-reachable through whatever writes to the index, and nothing about
    the user's turn signals that a retrieved chunk carries instructions.

    Each surviving document's text is replaced by the provenance envelope.
    Quarantined documents are dropped by default -- unlike tool results, a
    dropped document breaks no contract, and there is no id the provider will
    later demand an answer for.
    """

    async def _node(state: Mapping[str, Any], config: Any = None) -> dict[str, Any]:
        docs = state.get(docs_key) or []
        if not docs:
            return {}
        sid = session_id or thread_session_id(config)
        kept, record = await guard_documents(
            pipeline,
            docs,
            session_id=sid,
            drop_quarantined=drop_quarantined,
            redact_secrets=redact_secrets,
        )
        return {docs_key: kept, state_key: record}

    return _node


async def guard_documents(
    pipeline: GuardrailPipeline,
    docs: Sequence[Any],
    *,
    session_id: str = _DEFAULT_SESSION,
    drop_quarantined: bool = True,
    redact_secrets: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """Envelope-wrap a document list. Returns (kept, record).

    Delegates to the shared implementation, so this and the LangChain retriever
    guard quarantine on the same rules and emit the same audit shape.
    """
    from ._lc_guards import guard_documents as _guard_documents

    kept, verdict, counts = await _guard_documents(
        pipeline,
        docs,
        session_id=session_id,
        drop_quarantined=drop_quarantined,
        redact_secrets=redact_secrets,
    )
    record = guardrail_record(verdict, "retrieval", pipeline)
    record.update(counts)
    # Dropping a poisoned chunk is not a refusal to the user.
    record["blocked"] = False
    record["refusal"] = None
    return kept, record


async def guard_message_stream(
    pipeline: GuardrailPipeline,
    stream: Any,
    *,
    session_id: str = _DEFAULT_SESSION,
    holdback_chars: int = 256,
    coalesce_chars: int = 32,
    allowed_citations: Sequence[str] | None = None,
) -> Any:
    """Guard a token stream from `graph.astream(..., stream_mode="messages")`.

    `guard_output_node` runs once per turn, which is the right place for a
    non-streaming graph but useless when tokens are already on their way to the
    user by the time the node executes. Wrap the stream instead::

        stream = graph.astream(state, config, stream_mode="messages")
        async for chunk in guard_message_stream(pipeline, stream, session_id=sid):
            if chunk.blocked:
                await send(chunk.text)   # the refusal
                break
            await send(chunk.text)

    Accepts the `(message_chunk, metadata)` tuples that mode yields, as well as
    bare message chunks or strings. Yields `StreamChunk`s whose text is redacted
    and safe to emit; a BLOCK ends the stream with the refusal as its last chunk.
    """
    from ._lc_guards import guard_text_stream

    async for chunk in guard_text_stream(
        pipeline,
        stream,
        session_id=session_id,
        holdback_chars=holdback_chars,
        coalesce_chars=coalesce_chars,
        allowed_citations=allowed_citations,
    ):
        yield chunk


def approval_node(
    *,
    state_key: str = GUARDRAIL_KEY,
    on_flagged: bool = False,
    interrupt_fn: Callable[[Any], Any] | None = None,
    reason: str = "guardrail_block",
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Turn a guardrail block into a human decision instead of a dead end.

    Detection is not containment: the docs are explicit that least-privilege
    tools and human approval for irreversible actions are what bound the damage
    when injected instructions *are* followed. LangGraph is the one supported
    framework with native durable interrupts, so this is where that argument can
    actually be implemented rather than recommended.

    Place between a guard node and the router::

        g.add_node("guard_calls", guard_tool_calls_node(pipeline))
        g.add_node("approve", approval_node())
        g.add_edge("guard_calls", "approve")
        g.add_conditional_edges("approve", guardrail_router(ok="tools", blocked="refuse"))

    On a block it calls `interrupt(payload)`, which suspends the graph, persists
    the payload to the checkpoint, and surfaces it to whatever is driving the run.
    Resume with `Command(resume=True)` to approve, or `Command(resume=False)` --
    or nothing recognisable -- to refuse. Approval clears `blocked`, so the
    existing router sends the run onward with no extra wiring.

    **Fails closed.** An unrecognised resume value is treated as a rejection,
    because the alternative is that a malformed or replayed resume silently
    authorises the action the guardrail stopped.

    **This node deliberately does nothing but read state and interrupt.**
    `interrupt()` suspends by raising, and on resume LangGraph re-executes the
    node from the top -- so any scan, ledger write, or audit emission placed here
    would run twice and be double-counted. Guarding belongs in the guard node
    that ran before this one; the decision recorded here is durable because the
    checkpointer persists state.
    """

    def _node(state: Mapping[str, Any]) -> dict[str, Any]:
        record = state.get(state_key)
        if not isinstance(record, Mapping):
            return {}
        needs_review = bool(record.get("blocked")) or (
            on_flagged and bool(record.get("flagged"))
        )
        if not needs_review:
            return {}

        interrupt = interrupt_fn or _langgraph_interrupt()
        payload = {
            "reason": reason,
            "stage": record.get("stage"),
            "policy_id": record.get("policy_id"),
            "action": record.get("action"),
            "categories": record.get("categories"),
            "refused_tools": record.get("refused_tools"),
            "refusal": record.get("refusal"),
        }
        decision = interrupt(payload)
        approved, meta = _normalise_decision(decision)

        updated = dict(record)
        updated["approval"] = {"approved": approved, **meta}
        if approved:
            # The human is the higher authority: clear the block so the existing
            # router proceeds, but keep the original verdict fields for audit.
            updated["blocked"] = False
            updated["refusal"] = None
        return {state_key: updated}

    return _node


def _langgraph_interrupt() -> Callable[[Any], Any]:
    """Resolve `langgraph.types.interrupt` lazily, so importing this module is free."""
    try:
        from langgraph.types import interrupt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "approval_node() needs langgraph's interrupt(): "
            "pip install 'charterguard[langgraph]', or pass interrupt_fn=... "
            "to supply your own approval mechanism."
        ) from exc
    return interrupt


def _normalise_decision(decision: Any) -> tuple[bool, dict[str, Any]]:
    """Interpret a resume value. Anything unrecognised is a rejection."""
    if decision is True:
        return True, {}
    if isinstance(decision, str):
        return decision.strip().lower() in ("approve", "approved", "yes", "y", "true"), {}
    if isinstance(decision, Mapping):
        raw = decision.get("approved", decision.get("approve", decision.get("decision")))
        approved, _ = _normalise_decision(raw if raw is not None else False)
        meta = {
            k: v
            for k, v in decision.items()
            if k in ("reviewer", "note", "reason", "ticket") and isinstance(v, (str, int))
        }
        return approved, meta
    return False, {}


def guardrail_router(
    *,
    ok: str = "model",
    blocked: str = "refuse",
    flagged: str | None = None,
    state_key: str = GUARDRAIL_KEY,
) -> Callable[[Mapping[str, Any]], str]:
    """Conditional edge that reads the record a guard node wrote."""

    def _route(state: Mapping[str, Any]) -> str:
        record = state.get(state_key) or {}
        if not isinstance(record, Mapping):
            return ok
        if record.get("blocked"):
            return blocked
        if flagged is not None and record.get("flagged"):
            return flagged
        return ok

    return _route


def refusal_node(
    *,
    messages_key: str = "messages",
    state_key: str = GUARDRAIL_KEY,
    output_key: str = SAFE_OUTPUT_KEY,
    fallback: str = "I can't help with that request.",
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Terminal node that emits the refusal the blocking verdict rendered."""

    def _node(state: Mapping[str, Any]) -> dict[str, Any]:
        record = state.get(state_key) or {}
        text = fallback
        if isinstance(record, Mapping) and isinstance(record.get("refusal"), str):
            text = record["refusal"] or fallback
        from ._lc_common import ai_message

        return {messages_key: [ai_message(text)], output_key: text}

    return _node
