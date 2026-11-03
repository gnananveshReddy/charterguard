"""OpenAI Agents SDK: charterguard as input, output and tool guardrails.

    from agents import Agent, function_tool
    from charterguard.integrations.openai_agents import (
        input_guardrail, output_guardrail, tool_input_guardrail, tool_output_guardrail,
    )

    agent = Agent(
        name="support",
        input_guardrails=[input_guardrail(pipeline)],
        output_guardrails=[output_guardrail(pipeline)],
    )

    @function_tool(tool_input_guardrails=[tool_input_guardrail(pipeline)],
                   tool_output_guardrails=[tool_output_guardrail(pipeline)])
    def search(query: str) -> str: ...

A blocked verdict trips the SDK's tripwire (`InputGuardrailTripwireTriggered` /
`OutputGuardrailTripwireTriggered`); the verdict summary -- never the matched
text -- is the guardrail's ``output_info``.

Two behaviours differ from the SDK defaults, on purpose:

* The input guardrail runs **blocking** (``run_in_parallel=False``). In the
  default parallel mode the agent starts work, tool calls included, while the
  guardrail is still deciding.
* The tool *output* guardrail replaces the output the model sees whenever the
  content is quarantined **or** anything was redacted (the SDK cannot edit a
  tool result, so "replace" is `reject_content` with the enveloped text).

Requires the ``openai-agents`` package (imported when a factory is called).
The guardrail functions themselves are plain callables, testable without it.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..pipeline import GuardrailPipeline
from ..types import Verdict

__all__ = [
    "input_guardrail",
    "output_guardrail",
    "tool_input_guardrail",
    "tool_output_guardrail",
    "input_text",
]

_DEFAULT_SESSION = "default"


def _agents() -> Any:
    try:
        import agents
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("charterguard.integrations.openai_agents needs `pip install openai-agents`") from exc
    return agents


def _summary(verdict: Verdict) -> dict[str, Any]:
    return {
        "action": verdict.action.value,
        "score": round(verdict.score, 3),
        "policy_id": verdict.policy_id,
        "rules": [f.rule_id for f in verdict.findings],
    }


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        texts = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "\n".join(texts)
    return ""


_TRUSTED_ROLES = frozenset({"system", "developer"})


# Bookkeeping fields the SDK sets itself; everything else in an untrusted item
# is scanned. A denylist, not an allowlist: a new item type with a new field
# name (results, error, action, queries, ...) is covered by default.
_METADATA_KEYS = frozenset({"id", "call_id", "status", "type", "role"})


def _leaves(value: object, out: list[str], depth: int = 0) -> None:
    """Collect every string (and key) in ``value`` as plain, unescaped text."""
    if depth > 64:
        out.append(str(value))
    elif isinstance(value, str):
        out.append(value)
    elif isinstance(value, Mapping):
        for k, v in value.items():
            out.append(str(k))
            _leaves(v, out, depth + 1)
    elif isinstance(value, list | tuple | set | frozenset):
        for v in value:
            _leaves(v, out, depth + 1)
    elif value is not None and not isinstance(value, bool | int | float):
        out.append(str(value))


def _item_text(item: Mapping[str, Any]) -> str:
    """Every model-readable string an untrusted input item carries.

    Each string leaf is scanned as plain text. A JSON dump alone would hide a
    pattern behind escaping (a newline becomes the two characters ``\\n``, a
    quote becomes ``\\"``), so the dump is only appended for structure.
    """
    payload = {k: v for k, v in item.items() if k not in _METADATA_KEYS}
    if not payload:
        return ""
    texts: list[str] = []
    _leaves(payload, texts)
    rest = json.dumps(payload, ensure_ascii=False, default=str)
    return "\n".join([*(t for t in texts if t), rest])


def input_text(user_input: str | Sequence[Any]) -> tuple[str, list[str]]:
    """Split an Agents SDK run input into (user text, untrusted texts).

    User messages are the turn checked with `check_input`. System and developer
    messages are the application's own instructions and are not scanned.
    **Everything else** -- assistant messages, tool and function call outputs,
    computer/web/file-search results, reasoning summaries, unknown item types,
    and non-dict items -- may carry injected text and is returned as untrusted.
    """
    if isinstance(user_input, str):
        return user_input, []
    user_parts: list[str] = []
    untrusted: list[str] = []
    for item in user_input:
        if not isinstance(item, Mapping):
            dump = getattr(item, "model_dump", None)
            item = dump() if callable(dump) else {"content": str(item)}
            if not isinstance(item, Mapping):
                continue
        role = item.get("role")
        if role == "user" and item.get("type") in (None, "message"):
            user_parts.append(_content_text(item.get("content")))
        elif role in _TRUSTED_ROLES and item.get("type") in (None, "message"):
            continue
        else:
            text = _item_text(item)
            if text:
                untrusted.append(text)
    return "\n".join(p for p in user_parts if p), untrusted


def _session(ctx: Any, session_id: Callable[[Any], str] | None) -> str:
    return session_id(ctx) if session_id is not None else _DEFAULT_SESSION


# ----------------------------------------------------------- agent level
def input_guardrail(
    pipeline: GuardrailPipeline,
    *,
    name: str = "charterguard_input",
    session_id: Callable[[Any], str] | None = None,
) -> Any:
    """An `agents.InputGuardrail` that trips when the turn would be blocked."""
    agents = _agents()

    def check(ctx: Any, agent: Any, user_input: str | list[Any]) -> Any:
        sid = _session(ctx, session_id)
        text, untrusted = input_text(user_input)
        verdict = pipeline.check_input(text, session_id=sid)
        info: dict[str, Any] = {"input": _summary(verdict)}
        tripped = verdict.blocked
        quarantined = [pipeline.inspect_untrusted(u, kind="tool_result", session_id=sid) for u in untrusted]
        if quarantined:
            info["tool_results"] = [_summary(r.verdict) for r in quarantined]
            tripped = tripped or any(r.quarantined for r in quarantined)
        return agents.GuardrailFunctionOutput(output_info=info, tripwire_triggered=tripped)

    return agents.InputGuardrail(guardrail_function=check, name=name, run_in_parallel=False)


def _output_text(output: object) -> str:
    if isinstance(output, str):
        return output
    dump = getattr(output, "model_dump_json", None)
    if callable(dump):
        return str(dump())
    try:
        return json.dumps(output, default=str)
    except (TypeError, ValueError):
        return str(output)


def output_guardrail(
    pipeline: GuardrailPipeline,
    *,
    name: str = "charterguard_output",
    session_id: Callable[[Any], str] | None = None,
) -> Any:
    """An `agents.OutputGuardrail` that trips when the final output would be blocked.

    Redaction cannot be applied here (the SDK does not let a guardrail edit the
    final output), so use `GuardrailPipeline.check_output(...).sanitized` on the
    result you display, or set PII to BLOCK in the policy.
    """
    agents = _agents()

    def check(ctx: Any, agent: Any, output: Any) -> Any:
        verdict = pipeline.check_output(_output_text(output), session_id=_session(ctx, session_id))
        return agents.GuardrailFunctionOutput(output_info=_summary(verdict), tripwire_triggered=verdict.blocked)

    return agents.OutputGuardrail(guardrail_function=check, name=name)


# ------------------------------------------------------------- tool level
def tool_input_guardrail(
    pipeline: GuardrailPipeline,
    *,
    name: str = "charterguard_tool_input",
    session_id: Callable[[Any], str] | None = None,
) -> Any:
    """Reject tool calls the scope charter forbids or whose arguments carry
    command-injection / exfiltration payloads. ``session_id`` receives the
    SDK's ToolContext; pass it so tool-call ceilings are per conversation."""
    agents = _agents()

    def check(data: Any) -> Any:
        ctx = data.context
        raw = getattr(ctx, "tool_arguments", None) or "{}"
        try:
            arguments: object = json.loads(raw)
        except ValueError:
            arguments = raw
        verdict = pipeline.check_tool_call(
            str(getattr(ctx, "tool_name", "tool")), arguments, session_id=_session(ctx, session_id)
        )
        if verdict.blocked:
            return agents.ToolGuardrailFunctionOutput.reject_content(
                f"Tool call blocked by policy ({verdict.policy_id}).", output_info=_summary(verdict)
            )
        return agents.ToolGuardrailFunctionOutput.allow(output_info=_summary(verdict))

    return agents.ToolInputGuardrail(guardrail_function=check, name=name)


def tool_output_guardrail(
    pipeline: GuardrailPipeline,
    *,
    name: str = "charterguard_tool_output",
    session_id: Callable[[Any], str] | None = None,
) -> Any:
    """Treat tool output as untrusted: quarantine injections, redact secrets.
    ``session_id`` receives the SDK's ToolContext."""
    agents = _agents()

    def check(data: Any) -> Any:
        output = data.output
        text = output if isinstance(output, str) else _output_text(output)
        result = pipeline.inspect_tool_output(
            text,
            tool_name=str(getattr(data.context, "tool_name", "tool")),
            session_id=_session(data.context, session_id),
        )
        info = {"quarantined": result.quarantined, **_summary(result.verdict)}
        redactions = int(result.verdict.meta.get("redaction_count", 0))
        if result.quarantined or redactions:
            return agents.ToolGuardrailFunctionOutput.reject_content(result.envelope, output_info=info)
        return agents.ToolGuardrailFunctionOutput.allow(output_info=info)

    return agents.ToolOutputGuardrail(guardrail_function=check, name=name)
