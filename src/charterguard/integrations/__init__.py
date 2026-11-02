"""Optional integrations. Nothing here is imported by the core package.

Every name below is resolved lazily (PEP 562), so `import charterguard` and
even `import charterguard.integrations` stay dependency-free and cold. An
optional third-party package is imported only when you touch the specific
adapter that needs it, and a missing one raises an ImportError that names the
extra to install.

Dependency tiers, so you can see what a name costs before you use it:

    stdlib only  -- JsonlAuditSink, LoggingMetricsSink, ConsoleMetricsSink,
                    StatsdMetricsSink, SqliteCounterStore, SqliteAuditSink,
                    HttpModelScorer, OpenAIModerationScorer,
                    HttpIntentClassifier, HttpEntityRecognizer,
                    KeywordIntentClassifier, GazetteerEntityRecognizer,
                    McpServer, serve_stdio, guard_mcp_result
    [fastapi]    -- guardrail_middleware
    [langgraph]  -- (nodes are plain callables and need no third-party import;
                    only approval_node() needs langgraph, and only at call time
                    unless you pass your own interrupt_fn)
    [langchain]  -- guard_input, guard_output, guard_output_stream,
                    guard_retrieval, guard_tool, guarded_tools, GuardedRetriever
    [redis]      -- RedisCounterStore
    [prometheus] -- PrometheusMetricsSink
    [otel]       -- OtelMetricsSink
    [spacy]      -- SpacyEntityRecognizer
    [presidio]   -- PresidioEntityRecognizer
    [onnx]       -- OnnxModelScorer, OnnxEmbedder, OnnxTextClassifier
    [openai-agents]    -- integrations.openai_agents: input_guardrail,
                          output_guardrail, tool_input_guardrail, tool_output_guardrail
    [claude-agent-sdk] -- claude_agent_hooks
"""
from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY: dict[str, str] = {
    # audit (stdlib)
    "AuditSink": "audit",
    "NullAuditSink": "audit",
    "JsonlAuditSink": "audit",
    "verify_chain": "audit",
    "event_payload": "audit",
    "guardrail_rows": "audit",
    "GENESIS_HASH": "audit",
    # metrics (stdlib)
    "LoggingMetricsSink": "logging_metrics",
    "ConsoleMetricsSink": "logging_metrics",
    "StatsdMetricsSink": "logging_metrics",
    # ledger + audit persistence (stdlib)
    "SqliteCounterStore": "sqlite_backends",
    "SqliteAuditSink": "sqlite_backends",
    # model / intent / NER over HTTP (stdlib)
    "HttpError": "http_adapters",
    "HttpModelScorer": "http_adapters",
    "OpenAIModerationScorer": "http_adapters",
    "HttpIntentClassifier": "http_adapters",
    "HttpEntityRecognizer": "http_adapters",
    # dependency-free placeholders for the model seams
    "KeywordIntentClassifier": "heuristics",
    "GazetteerEntityRecognizer": "heuristics",
    # LangGraph nodes: stdlib only, despite the framework name
    "GUARDRAIL_KEY": "langgraph",
    "approval_node": "langgraph",
    "guard_message_stream": "langgraph",
    "GuardrailState": "langgraph",
    "guard_input_node": "langgraph",
    "guard_output_node": "langgraph",
    "guard_retrieval_node": "langgraph",
    "guard_tool_calls_node": "langgraph",
    "guard_tool_results_node": "langgraph",
    "guardrail_record": "langgraph",
    "guardrail_router": "langgraph",
    "refusal_node": "langgraph",
    "thread_session_id": "langgraph",
    # Model Context Protocol server + result guard (stdlib)
    "McpServer": "mcp",
    "serve_stdio": "mcp",
    "guard_mcp_result": "mcp",
    # agent SDKs: the SDK is imported only when a factory is called. The
    # OpenAI Agents guardrails live in .openai_agents (input_guardrail, ...).
    "claude_agent_hooks": "claude_agent_sdk",
    # optional third-party
    "guardrail_middleware": "fastapi",
    "GuardedRetriever": "langchain",
    "config_session_id": "langchain",
    "guard_input": "langchain",
    "guard_output": "langchain",
    "guard_output_stream": "langchain",
    "guard_retrieval": "langchain",
    "guard_tool": "langchain",
    "guarded_tools": "langchain",
    "RedisCounterStore": "redis_backends",
    "FallbackCounterStore": "redis_backends",
    "PrometheusMetricsSink": "prometheus_metrics",
    "OtelMetricsSink": "otel_metrics",
    "SpacyEntityRecognizer": "spacy_ner",
    "SPACY_LABEL_MAP": "spacy_ner",
    "PresidioEntityRecognizer": "presidio_ner",
    "PRESIDIO_LABEL_MAP": "presidio_ner",
    "FileSpec": "onnx_model",
    "OnnxModelSpec": "onnx_model",
    "ModelIntegrityError": "onnx_model",
    "ensure_local": "onnx_model",
    "OnnxTextClassifier": "onnx_model",
    "OnnxModelScorer": "onnx_model",
    "OnnxEmbedder": "onnx_model",
}

__all__ = sorted(_LAZY)

if TYPE_CHECKING:  # give type checkers the real symbols without a runtime cost
    from .audit import (  # noqa: F401
        GENESIS_HASH,
        AuditSink,
        JsonlAuditSink,
        NullAuditSink,
        event_payload,
        guardrail_rows,
        verify_chain,
    )
    from .heuristics import GazetteerEntityRecognizer, KeywordIntentClassifier  # noqa: F401
    from .claude_agent_sdk import claude_agent_hooks  # noqa: F401
    from .mcp import McpServer, guard_mcp_result, serve_stdio  # noqa: F401
    from .langgraph import (  # noqa: F401
        GUARDRAIL_KEY,
        GuardrailState,
        approval_node,
        guard_message_stream,
        guard_input_node,
        guard_output_node,
        guard_retrieval_node,
        guard_tool_calls_node,
        guard_tool_results_node,
        guardrail_record,
        guardrail_router,
        refusal_node,
        thread_session_id,
    )
    from .http_adapters import (  # noqa: F401
        HttpEntityRecognizer,
        HttpError,
        HttpIntentClassifier,
        HttpModelScorer,
        OpenAIModerationScorer,
    )
    from .logging_metrics import (  # noqa: F401
        ConsoleMetricsSink,
        LoggingMetricsSink,
        StatsdMetricsSink,
    )
    from .sqlite_backends import SqliteAuditSink, SqliteCounterStore  # noqa: F401


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value  # pay the lazy path once
    return value


def __dir__() -> list[str]:
    return list(__all__)
