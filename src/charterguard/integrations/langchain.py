"""LangChain wiring: guard Runnables, a guarded retriever, and guarded tools.

Kept out of the core package so `charterguard` stays dependency-free:
importing this module is the only thing that requires langchain-core.

The guard decisions live in `_lc_guards`, which imports nothing third-party.
This module is the typed shell that makes them composable with `|`, so the
behaviour is identical to the LangGraph adapter's and is tested without
langchain-core installed.

Wiring a chain::

    from charterguard.integrations.langchain import guard_input, guard_output

    chain = guard_input(pipeline) | prompt | model | guard_output(pipeline)
    answer = await chain.ainvoke("hello", config={"configurable": {"session_id": "u-1"}})
"""
from __future__ import annotations

from typing import Any
from collections.abc import Sequence

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.messages import AIMessageChunk
    from langchain_core.runnables import (
        Runnable,
        RunnableConfig,
        RunnableGenerator,
        RunnableLambda,
    )
    from langchain_core.tools import BaseTool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "charterguard.integrations.langchain requires langchain-core: "
        "pip install 'charterguard[langchain]'"
    ) from exc

from ..pipeline import GuardrailPipeline
from . import _lc_guards as _g

__all__ = [
    "GuardedRetriever",
    "GuardrailBlocked",
    "config_session_id",
    "guard_input",
    "guard_output",
    "guard_output_stream",
    "guard_retrieval",
    "guard_tool",
    "guarded_tools",
]

from ..exceptions import GuardrailBlocked  # noqa: E402  (re-exported for callers)

_DEFAULT_SESSION = "default"


def config_session_id(config: Any, default: str = _DEFAULT_SESSION) -> str:
    """Read the guardrail session from a RunnableConfig.

    Worth wiring: the ledger's per-session ceilings (tool calls, cumulative
    output, injection attempts) are only meaningful when sessions map to real
    users. Unset, every request on the process shares `"default"`, so one busy
    caller trips the limit for all of them.
    """
    if not isinstance(config, dict):
        return default
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return default
    for key in ("session_id", "thread_id", "user_id"):
        value = configurable.get(key)
        if isinstance(value, str) and value:
            return value
    return default


def guard_input(
    pipeline: GuardrailPipeline,
    *,
    enforce_scope: bool = False,
    on_block: str = _g.RAISE,
    session_id: str | None = None,
) -> Runnable:
    """Runnable that scans the inbound turn and passes it through.

    Accepts whatever the chain already carries -- a string, a message, a message
    list, a `PromptValue`, or a dict of chain variables -- and returns the same
    shape, with redactions applied when the scan sanitised the text.

    Defaults to raising `GuardrailBlocked` so a blocked prompt cannot reach the
    model through a chain that forgot to check a return value. Use
    `on_block="refuse"` to have the refusal text flow onward instead.
    """

    async def _guard(payload: Any, config: RunnableConfig | None = None) -> Any:
        sid = session_id or config_session_id(config)
        guarded, _verdict, refusal = await _g.guard_input_payload(
            pipeline,
            payload,
            session_id=sid,
            enforce_scope=enforce_scope,
            on_block=on_block,
        )
        return refusal if refusal is not None else guarded

    return RunnableLambda(_guard, name="guardrail_input")


def guard_output(
    pipeline: GuardrailPipeline,
    *,
    allowed_citations: Sequence[str] | None = None,
    on_block: str = _g.REFUSE,
    session_id: str | None = None,
) -> Runnable:
    """Runnable that gates the model's response as the last link in a chain.

    Defaults to substituting the refusal rather than raising: a user is waiting
    on this call, and a rendered refusal is a better outcome than an exception
    surfacing as a 500. Set `on_block="raise"` when an upstream handler owns the
    error path.
    """

    async def _guard(payload: Any, config: RunnableConfig | None = None) -> Any:
        sid = session_id or config_session_id(config)
        guarded, _verdict, _refusal = await _g.guard_output_payload(
            pipeline,
            payload,
            session_id=sid,
            allowed_citations=allowed_citations,
            on_block=on_block,
        )
        return guarded

    return RunnableLambda(_guard, name="guardrail_output")


def guard_output_stream(
    pipeline: GuardrailPipeline,
    *,
    holdback_chars: int = 256,
    coalesce_chars: int = 32,
    allowed_citations: Sequence[str] | None = None,
    on_block: str = _g.REFUSE,
    session_id: str | None = None,
) -> Runnable:
    """Streaming output gate. Use this instead of `guard_output` when you stream.

    `guard_output` is a `RunnableLambda`, and a lambda has to see its whole input
    before it can return: dropping one into a streaming chain silently buffers
    the entire response, so the user waits for the full generation and gets no
    incremental enforcement. This is a `RunnableGenerator`, so chunks flow
    through as they arrive.

    Enforcement is incremental via a hold-back window: text is released only once
    `holdback_chars` of newer text has been scanned behind it, which is what
    catches a secret spanning two chunks *before* it reaches the user. The cost is
    explicit -- output trails generation by that many characters. On a BLOCK the
    stream stops and a final chunk carries the refusal.

    The honest limit: already-released text cannot be recalled. The window bounds
    how much text can be emitted unscanned, not how much can be retracted.

    `coalesce_chars` bounds how often the buffer is rescanned. It defaults to 32
    because a scan re-reads the whole buffer, so per-token scanning makes cost
    grow with response length; set it to 0 to scan on every chunk.

    Async only, deliberately: `.stream()` would need a blocking bridge inside a
    running loop. Use `.astream()`.
    """

    async def _atransform(chunks: Any, config: Any = None, **_kwargs: Any) -> Any:
        sid = session_id or config_session_id(config)
        # Remember the upstream chunk type so the guarded stream yields the same
        # shape a downstream parser expects.
        sample: dict[str, Any] = {}

        async def _tap() -> Any:
            async for chunk in chunks:
                sample["chunk"] = chunk
                yield chunk

        async for out in _g.guard_text_stream(
            pipeline,
            _tap(),
            session_id=sid,
            holdback_chars=holdback_chars,
            coalesce_chars=coalesce_chars,
            allowed_citations=allowed_citations,
            on_block=on_block,
        ):
            yield _as_stream_chunk(sample.get("chunk"), out.text)

    return RunnableGenerator(_atransform, name="guardrail_output_stream")


def _as_stream_chunk(sample: Any, text: str) -> Any:
    """Match the upstream chunk type: str in, str out; message chunk otherwise."""
    if sample is None or isinstance(sample, str):
        return text
    return AIMessageChunk(content=text)


def guard_retrieval(
    pipeline: GuardrailPipeline,
    *,
    drop_quarantined: bool = True,
    redact_secrets: bool = True,
    session_id: str | None = None,
) -> Runnable:
    """Runnable that envelope-wraps a list of documents mid-chain.

    Use when documents reach the prompt through a `RunnableParallel` branch
    rather than through a retriever you control; otherwise prefer
    `GuardedRetriever`, which cannot be bypassed.
    """

    async def _guard(docs: Sequence[Any], config: RunnableConfig | None = None) -> list[Any]:
        sid = session_id or config_session_id(config)
        kept, _verdict, _counts = await _g.guard_documents(
            pipeline,
            list(docs or []),
            session_id=sid,
            drop_quarantined=drop_quarantined,
            redact_secrets=redact_secrets,
        )
        return kept

    return RunnableLambda(_guard, name="guardrail_retrieval")


class GuardedRetriever(BaseRetriever):
    """Wraps a retriever so every document is inspected before it is used.

    This is the layer that most needs guarding in a RAG chain: a document is
    attacker-reachable through whatever writes to the index, and nothing in the
    user's turn signals that a retrieved chunk carries instructions. Wrapping
    the retriever rather than the prompt means a new chain built on the same
    retriever inherits the guard instead of forgetting it.

    Retrieval is I/O, so the async path is the real implementation;
    `_get_relevant_documents` exists because `BaseRetriever` requires it and
    bridges through the pipeline's sync entry points.
    """

    retriever: BaseRetriever
    pipeline: GuardrailPipeline
    drop_quarantined: bool = True
    redact_secrets: bool = True
    session_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None, **kwargs: Any
    ) -> list[Document]:
        docs = await self.retriever.ainvoke(query)
        kept, _verdict, _counts = await _g.guard_documents(
            self.pipeline,
            docs,
            session_id=self.session_id or _DEFAULT_SESSION,
            drop_quarantined=self.drop_quarantined,
            redact_secrets=self.redact_secrets,
        )
        return kept

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None, **kwargs: Any
    ) -> list[Document]:
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            # Bridging into a live loop would deadlock rather than fail, so say
            # exactly what to call instead.
            raise RuntimeError(
                "GuardedRetriever.invoke() was called from a running event loop; "
                "use `await retriever.ainvoke(query)` instead."
            )
        docs = self.retriever.invoke(query)
        return asyncio.run(
            _guarded_docs_sync(
                self.pipeline,
                docs,
                self.session_id or _DEFAULT_SESSION,
                self.drop_quarantined,
                self.redact_secrets,
            )
        )


async def _guarded_docs_sync(
    pipeline: GuardrailPipeline,
    docs: Sequence[Any],
    session_id: str,
    drop_quarantined: bool,
    redact_secrets: bool,
) -> list[Any]:
    kept, _verdict, _counts = await _g.guard_documents(
        pipeline,
        docs,
        session_id=session_id,
        drop_quarantined=drop_quarantined,
        redact_secrets=redact_secrets,
    )
    return kept


def guard_tool(
    tool: BaseTool,
    pipeline: GuardrailPipeline,
    *,
    session_id: str | None = None,
    redact_secrets: bool = True,
    on_block: str = _g.REFUSE,
) -> BaseTool:
    """Wrap a tool so its arguments are authorised and its output is enveloped.

    Both halves matter and they guard different things. The pre-check enforces
    the charter and argument policy before any side effect happens. The
    post-check treats the result as untrusted content, redacting secrets and
    replacing the body with a provenance envelope so text returned by a tool
    reads as data rather than as instructions.

    Blocked calls default to returning the refusal string rather than raising:
    an agent receives it as the tool's result and can adapt, whereas an
    exception inside a tool node usually ends the run.

    The wrapper keeps the original name, description, and argument schema, so
    the model's view of the tool is unchanged and no re-binding is needed.
    """
    guarded_session = session_id or _DEFAULT_SESSION

    async def _arun(*args: Any, **kwargs: Any) -> Any:
        payload = kwargs if kwargs else (args[0] if args else {})
        _verdict, refusal = await _g.guard_tool_arguments(
            pipeline, tool.name, payload, session_id=guarded_session, on_block=on_block
        )
        if refusal is not None:
            return refusal
        result = await tool.ainvoke(kwargs if kwargs else payload)
        body, _v, _q = await _g.guard_tool_result(
            pipeline,
            result,
            tool_name=tool.name,
            session_id=guarded_session,
            redact_secrets=redact_secrets,
        )
        return body

    from langchain_core.tools import StructuredTool

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        # Newer langchain-core requires a schema; fall back to the inferred one.
        args_schema=tool.args_schema if tool.args_schema is not None else tool.get_input_schema(),
        coroutine=_arun,
        # No sync `func`: guarding is async end to end, and a blocking bridge
        # inside a running event loop is a deadlock waiting to happen. Call the
        # async path (`ainvoke`), which is what agent executors use anyway.
        return_direct=getattr(tool, "return_direct", False),
    )


def guarded_tools(
    tools: Sequence[BaseTool],
    pipeline: GuardrailPipeline,
    *,
    session_id: str | None = None,
    redact_secrets: bool = True,
    on_block: str = _g.REFUSE,
) -> list[BaseTool]:
    """`guard_tool` over a list -- convenient right before `bind_tools`."""
    return [
        guard_tool(
            tool,
            pipeline,
            session_id=session_id,
            redact_secrets=redact_secrets,
            on_block=on_block,
        )
        for tool in tools
    ]
