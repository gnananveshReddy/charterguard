"""Use versus mention: attack phrases quoted in order to discuss them.

"How does prompt injection work? Explain 'ignore previous instructions'
attacks." contains the most classic override phrase there is, and blocking it
fails exactly the users -- security engineers, students, red-teamers -- most
likely to adopt a guardrail library. The phrase is *mentioned*, not *used*.

The distinction is only drawn on the user's own turn (``Layer.INPUT``). In
retrieved documents and tool output, quoting is a free evasion -- an attacker
controls the quotes -- so untrusted layers are never dampened.

Even on input, dampening is deliberately narrow:

* only injection and jailbreak findings, and only those lying entirely inside
  a quoted span, are lowered -- to MEDIUM, which flags rather than allows;
* the text outside the quotes must read as discussion ("explain", "how does",
  "example", "attack", ...);
* the text *outside* the quotes is re-scanned from scratch with the quoted
  spans blanked. The rule detector stops at its first CRITICAL match, so
  without the re-scan a harmless quoted phrase placed first could shadow a real
  unquoted attack placed after it.
"""
from __future__ import annotations

import re

__all__ = ["quoted_spans", "mask_spans", "keep_only_spans", "reads_as_discussion"]

_QUOTE_CHARS = frozenset("\"'`“‘")

# Bounded, newline-free spans. Single quotes count only when they are not
# apostrophes ("don't ... 'x'" must not pair the apostrophe with the quote).
_QUOTED = re.compile(
    r'"[^"\n]{1,240}"'
    r"|“[^”\n]{1,240}”"
    r"|‘[^’\n]{1,240}’"
    r"|`[^`\n]{1,240}`"
    r"|(?<![A-Za-z0-9])'[^'\n]{1,240}'(?![A-Za-z0-9])"
)

_DISCUSSION = re.compile(
    r"\b(?:explain\w*|what\s+(?:is|are|does|do)|how\s+(?:does|do|can|would|is|are)|"
    r"examples?|detect\w*|defen[cs]e\w*|defend\w*|prevent\w*|mitigat\w*|classif\w*|"
    r"attacks?|injections?|jailbreak\w*|red[\s-]?team\w*|research\w*|"
    r"mean|means|meaning|phrase|phrases|payloads?|threat\s+model\w*)\b",
    re.IGNORECASE,
)


def quoted_spans(text: str) -> list[tuple[int, int]]:
    """Spans of quoted segments in ``text``; empty when there are none."""
    if not any(c in _QUOTE_CHARS for c in text):
        return []
    return [m.span() for m in _QUOTED.finditer(text)]


def mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Blank the spans, preserving length so offsets stay valid."""
    parts: list[str] = []
    pos = 0
    for s, e in spans:
        parts.append(text[pos:s])
        parts.append(" " * (e - s))
        pos = e
    parts.append(text[pos:])
    return "".join(parts)


def keep_only_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Blank everything except the spans, preserving length."""
    parts: list[str] = []
    pos = 0
    for s, e in spans:
        parts.append(" " * (s - pos))
        parts.append(text[s:e])
        pos = e
    parts.append(" " * (len(text) - pos))
    return "".join(parts)


def reads_as_discussion(unquoted: str) -> bool:
    return _DISCUSSION.search(unquoted) is not None
