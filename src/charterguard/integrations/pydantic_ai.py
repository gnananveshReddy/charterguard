"""Pydantic AI: charterguard as one agent capability.

    from pydantic_ai import Agent
    from charterguard.integrations.pydantic_ai import guard_capability

    agent = Agent("openai:gpt-5", capabilities=[guard_capability(pipeline)])

The capability hooks every stage of a run:

* **prompt** -- `check_input` before the run starts. A blocked prompt raises
  `GuardrailBlocked` and no model request is made.
* **tool calls** -- `check_tool_call` before the tool function runs. A blocked
  call is skipped; the model receives a refusal as the tool result.
* **tool results** -- `inspect_tool_output`. A quarantined or redacted result is
  replaced by the enveloped text, so the model never sees the raw content.
* **final output** -- `check_output`. A blocked output raises
  `GuardrailBlocked`; a plain-text output with redactions is returned redacted.
  A structured output (a pydantic model, dict or list) is rebuilt from its
  redacted JSON; if that fails, the run raises rather than leak.

The session id (per-conversation ceilings) defaults to the run's
``conversation_id``, which pydantic-ai keeps stable across runs that continue
a ``message_history``. Pass ``session_id`` to derive it from your deps instead.

Message history passed in with ``message_history=`` is not rescanned: it was
checked when it was produced. Requires ``pydantic-ai-slim`` (imported when the
factory is called).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..exceptions import GuardrailBlocked
from ..pipeline import GuardrailPipeline
from ..types import Verdict

__all__ = ["guard_capability", "prompt_text"]

_DEFAULT_SESSION = "default"


def _pydantic_ai() -> tuple[Any, Any]:
    try:
        from pydantic_ai import exceptions
        from pydantic_ai.capabilities import AbstractCapability
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("charterguard.integrations.pydantic_ai needs `pip install pydantic-ai-slim`") from exc
    return AbstractCapability, exceptions


def prompt_text(prompt: object) -> str:
    """The text of a run prompt: a string, or the text parts of a content list.

    Non-text parts (images, audio, documents) carry no scannable text and are
    skipped; scan their extracted text yourself if your app produces any.
    """
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    parts: list[str] = []
    for part in prompt if isinstance(prompt, list | tuple) else [prompt]:
        if isinstance(part, str):
            parts.append(part)
        else:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                parts.append(content)
    return "\n".join(parts)


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    dump = getattr(value, "model_dump_json", None)
    if callable(dump):
        return str(dump())
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


_UNREBUILDABLE = object()


def _rebuild(output: object, sanitized_json: str) -> object:
    """The structured ``output`` again, from its redacted JSON dump.

    Returns `_UNREBUILDABLE` when the redacted text no longer parses or
    validates, or the type is not one we know how to rebuild; the caller then
    fails closed rather than returning the unredacted original.
    """
    try:
        data = json.loads(sanitized_json)
    except ValueError:
        return _UNREBUILDABLE
    validate = getattr(type(output), "model_validate", None)
    if callable(validate):
        try:
            return validate(data)
        except Exception:  # noqa: BLE001 - any validation failure means fail closed
            return _UNREBUILDABLE
    if isinstance(output, dict | list) and isinstance(data, type(output)):
        return data
    return _UNREBUILDABLE


def guard_capability(
    pipeline: GuardrailPipeline,
    *,
    session_id: Callable[[Any], str] | None = None,
    check_prompt: bool = True,
    check_output: bool = True,
) -> Any:
    """A Pydantic AI capability that runs ``pipeline`` at every stage of a run.

    ``session_id`` receives the `RunContext`. Disable ``check_prompt`` or
    ``check_output`` when another layer (an HTTP middleware, a UI) already does
    that check, so findings are not counted twice.
    """
    abstract_capability, exceptions = _pydantic_ai()

    def sid(ctx: Any) -> str:
        if session_id is not None:
            return session_id(ctx)
        # Never the run id: it is new on every run, so per-session ceilings
        # would reset each turn. Without a conversation id, share one key.
        return str(getattr(ctx, "conversation_id", None) or _DEFAULT_SESSION)

    def blocked(verdict: Verdict) -> GuardrailBlocked:
        return GuardrailBlocked(verdict, pipeline.refusal(verdict))

    class CharterGuardCapability(abstract_capability):  # type: ignore[misc, valid-type]
        """charterguard checks on prompt, tool calls, tool results and output."""

        def __init__(self) -> None:
            self.pipeline = pipeline

        async def before_run(self, ctx: Any) -> None:
            if not check_prompt:
                return
            text = prompt_text(getattr(ctx, "prompt", None))
            if text:
                verdict = await pipeline.acheck_input(text, session_id=sid(ctx))
                if verdict.blocked:
                    raise blocked(verdict)

        async def before_tool_execute(self, ctx: Any, *, call: Any, tool_def: Any, args: Any) -> Any:
            verdict = pipeline.check_tool_call(str(call.tool_name), args, session_id=sid(ctx))
            if verdict.blocked:
                raise exceptions.SkipToolExecution(f"Tool call blocked by policy ({verdict.policy_id}).")
            return args

        async def after_tool_execute(
            self, ctx: Any, *, call: Any, tool_def: Any, args: Any, result: Any
        ) -> Any:
            inspected = pipeline.inspect_tool_output(
                _text(result), tool_name=str(call.tool_name), session_id=sid(ctx)
            )
            if inspected.quarantined or int(inspected.verdict.meta.get("redaction_count", 0)):
                return inspected.envelope
            return result

        async def before_output_process(self, ctx: Any, *, output_context: Any, output: Any) -> Any:
            if not check_output:
                return output
            verdict = pipeline.check_output(_text(output), session_id=sid(ctx))
            if verdict.blocked:
                raise blocked(verdict)
            if not int(verdict.meta.get("redaction_count", 0)) or verdict.sanitized is None:
                return output
            if isinstance(output, str):
                return verdict.sanitized
            rebuilt = _rebuild(output, verdict.sanitized)
            if rebuilt is _UNREBUILDABLE:
                # Fail closed: returning the original would leak what was redacted.
                raise GuardrailBlocked(
                    verdict, "The answer contained data that must be redacted and could not be rebuilt."
                )
            return rebuilt

    return CharterGuardCapability()
