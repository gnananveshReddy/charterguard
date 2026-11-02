"""Duck-typed helpers shared by the LangChain and LangGraph adapters.

Private module: no stability guarantee outside this package.

Everything here works on three shapes of the same thing -- a `langchain_core`
model instance, a plain dict, or a bare string -- so the LangGraph adapter can
stay import-free while the LangChain adapter gets the same behaviour on real
message objects.
"""
from __future__ import annotations

import copy
from typing import Any
from collections.abc import Mapping, Sequence

# Provider roles ("user", "assistant") and LangChain types ("human", "ai") name
# the same thing. Normalise so callers match on one vocabulary.
_ROLES = {
    "human": "human",
    "user": "human",
    "ai": "ai",
    "assistant": "ai",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


def message_role(msg: Any) -> str:
    if isinstance(msg, Mapping):
        raw = msg.get("role") or msg.get("type") or ""
    else:
        raw = getattr(msg, "type", "") or ""
    lowered = str(raw).lower()
    return _ROLES.get(lowered, lowered)


def message_id(msg: Any) -> str | None:
    got = msg.get("id") if isinstance(msg, Mapping) else getattr(msg, "id", None)
    return got if isinstance(got, str) and got else None


def is_block_content(content: Any) -> bool:
    """True when content is a multimodal block list rather than a plain string."""
    return not isinstance(content, str) and isinstance(content, Sequence) and not isinstance(
        content, (bytes, bytearray)
    )


def content_text(content: Any) -> str:
    """Flatten message content to scannable text.

    In langchain-core >= 0.3 `content` is `str | list[str | dict]` -- the list
    form carries multimodal blocks. `str(content)` on that list would still be
    scanned, but every offset a redaction produced would index into the repr
    (`[{'type': 'text', ...`) instead of the text, so spans would land on the
    wrong characters. Extracting and joining the text blocks keeps offsets
    meaningful and leaves image payloads out of the scan entirely.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if is_block_content(content):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, Mapping)
                and isinstance(block.get("text"), str)
                and block.get("type") in (None, "text")
            ):
                parts.append(block["text"])
        return "\n".join(p for p in parts if p)
    return str(content)


def message_text(msg: Any) -> str:
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, Mapping):
        return content_text(msg.get("content"))
    return content_text(getattr(msg, "content", ""))


def last_message(messages: Any, roles: tuple[str, ...] = ()) -> Any | None:
    """Newest message matching `roles` (any role when empty)."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return None
    for msg in reversed(messages):
        if not roles or message_role(msg) in roles:
            return msg
    return None


def trailing_messages(messages: Any, roles: tuple[str, ...]) -> list[Any]:
    """Unbroken run of `roles` at the end of the list, in original order.

    Used to find the tool results produced by the most recent step without
    re-inspecting (and re-charging the ledger for) results from earlier turns.
    """
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return []
    out: list[Any] = []
    for msg in reversed(messages):
        if message_role(msg) not in roles:
            break
        out.append(msg)
    out.reverse()
    return out


def _copy_with(obj: Any, field: str, value: Any) -> Any:
    """Copy a message/document with one field replaced, preserving identity.

    `model_copy` is the pydantic-v2 path and keeps `id` intact, which is what
    makes an in-place state update possible. The `copy.copy` fallback covers
    plain objects and pydantic v1.
    """
    if isinstance(obj, Mapping):
        clone = dict(obj)
        clone[field] = value
        return clone
    model_copy = getattr(obj, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={field: value})
    clone = copy.copy(obj)
    try:
        setattr(clone, field, value)
    except Exception as exc:  # frozen dataclass, __slots__ mismatch, ...
        raise TypeError(f"cannot rewrite {field!r} on {type(obj).__name__}") from exc
    return clone


def replace_message_text(msg: Any, text: str) -> Any:
    """Copy `msg` with its text replaced, keeping its id.

    Multimodal block lists are collapsed to a single text block: the guarded
    text is a flattened join of the original blocks, so writing it back
    block-by-block would need an alignment the scanner does not produce. Image
    parts are dropped rather than paired with text they no longer match.
    """
    if isinstance(msg, str):
        return text
    content = msg.get("content") if isinstance(msg, Mapping) else getattr(msg, "content", "")
    if is_block_content(content):
        return _copy_with(msg, "content", [{"type": "text", "text": text}])
    return _copy_with(msg, "content", text)


def tool_calls(msg: Any) -> list[dict[str, Any]]:
    """Normalised `[{"name", "args", "id"}]` from an assistant message."""
    raw = msg.get("tool_calls") if isinstance(msg, Mapping) else getattr(msg, "tool_calls", None)
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for call in raw:
        if isinstance(call, Mapping):
            name = call.get("name") or (call.get("function") or {}).get("name") or ""
            args = call.get("args")
            if args is None:
                args = (call.get("function") or {}).get("arguments")
            out.append({"name": str(name), "args": args, "id": call.get("id")})
        else:
            out.append(
                {
                    "name": str(getattr(call, "name", "") or ""),
                    "args": getattr(call, "args", None),
                    "id": getattr(call, "id", None),
                }
            )
    return out


def tool_call_id(msg: Any) -> str | None:
    got = (
        msg.get("tool_call_id")
        if isinstance(msg, Mapping)
        else getattr(msg, "tool_call_id", None)
    )
    return got if isinstance(got, str) and got else None


def tool_name(msg: Any) -> str:
    for attr in ("name", "tool"):
        got = msg.get(attr) if isinstance(msg, Mapping) else getattr(msg, attr, None)
        if isinstance(got, str) and got:
            return got
    return "unknown"


def document_text(doc: Any) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, Mapping):
        return str(doc.get("page_content") or doc.get("content") or "")
    return str(getattr(doc, "page_content", "") or "")


def document_source(doc: Any) -> str:
    meta = doc.get("metadata") if isinstance(doc, Mapping) else getattr(doc, "metadata", None)
    if isinstance(meta, Mapping):
        for key in ("source", "url", "file_path", "id"):
            got = meta.get(key)
            if isinstance(got, str) and got:
                return got
    return "unknown"


def replace_document_text(doc: Any, text: str) -> Any:
    if isinstance(doc, str):
        return text
    field = "page_content"
    if isinstance(doc, Mapping) and "page_content" not in doc and "content" in doc:
        field = "content"
    return _copy_with(doc, field, text)


def message_classes() -> tuple[Any, Any]:
    """(AIMessage, ToolMessage) when langchain-core is importable, else (None, None).

    Lets the LangGraph adapter emit real message objects where they are
    available and dicts where they are not, without importing langchain-core as
    a hard requirement.
    """
    try:
        from langchain_core.messages import AIMessage, ToolMessage
    except Exception:
        return (None, None)
    return (AIMessage, ToolMessage)


def ai_message(text: str) -> Any:
    ai_cls, _ = message_classes()
    return ai_cls(content=text) if ai_cls is not None else {"role": "assistant", "content": text}


def tool_message(text: str, call_id: str | None, name: str | None = None) -> Any:
    _, tool_cls = message_classes()
    if tool_cls is not None:
        kwargs: dict[str, Any] = {"content": text, "tool_call_id": call_id or ""}
        if name:
            kwargs["name"] = name
        return tool_cls(**kwargs)
    out: dict[str, Any] = {"role": "tool", "content": text, "tool_call_id": call_id or ""}
    if name:
        out["name"] = name
    return out
