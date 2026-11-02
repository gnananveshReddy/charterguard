"""Use versus mention: attack phrases quoted in order to discuss them.

"How does prompt injection work? Explain 'ignore previous instructions'
attacks." contains the most classic override phrase there is, and blocking it
fails exactly the users -- security engineers, students, red-teamers -- most
likely to adopt a guardrail library. The phrase is *mentioned*, not *used*.

The distinction is only drawn on the user's own turn (``Layer.INPUT``). In
retrieved documents and tool output, quoting is a free evasion -- an attacker
controls the quotes -- so untrusted layers are never dampened.

Even on input, dampening must not become a wrapper an attacker can put around a
real payload ("Do this: '<payload>' (example attack)"). It applies only when
the quote *is* the attack phrase and nothing more, which leaves nothing
actionable to smuggle:

* exactly one quoted span holds injection/jailbreak findings, and it is short;
* the matched phrase covers most of that quote -- no extra instructions ride
  along inside it -- and only one distinct rule fires inside it, so a quote
  carrying two techniques is never softened;
* the discussion cue sits right next to the quote ("explain 'X'", "what does
  'X' mean"), not anywhere in the turn;
* nothing outside the quote tells the model to act on it ("do this",
  "follow", "execute", "treat it as");
* the quote is not most of the turn.

Findings that qualify are lowered to MEDIUM, which flags rather than allows.
The text *outside* the quotes is re-scanned from scratch with the quote
blanked: the rule detector stops at its first CRITICAL match, so without the
re-scan a harmless quoted phrase placed first could shadow a real unquoted
attack placed after it.
"""
from __future__ import annotations

import re

__all__ = [
    "quoted_spans",
    "mask_spans",
    "keep_only_spans",
    "reads_as_discussion",
    "cue_beside",
    "directs_at_quote",
    "MAX_QUOTE_CHARS",
    "MIN_PHRASE_COVERAGE",
    "MAX_TURN_SHARE",
]

# A quoted *mention* is a phrase, not a paragraph.
MAX_QUOTE_CHARS = 120
# Share of the quote's content that the matched phrase must cover.
MIN_PHRASE_COVERAGE = 0.6
# A turn that is mostly one quote is not a discussion of it.
MAX_TURN_SHARE = 0.7
# How far from the quote a discussion cue may sit.
_CUE_REACH = 48

_DIRECTIVE = re.compile(
    r"\b(?:do\s+(?:this|that|it|exactly|the\s+following|as\s+(?:it\s+)?says)|follow|obey|execute|"
    r"run\s+(?:this|it|that)|apply|comply|act\s+on|carry\s+out|perform|treat\s+(?:this|it|that)\s+as|"
    r"take\s+(?:this|it)\s+(?:as|literally)|then\s+do|now\s+do|respond\s+as\s+if)\b",
    re.IGNORECASE,
)

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


def cue_beside(text: str, span: tuple[int, int]) -> bool:
    """A discussion cue within a short reach of the quote, on either side."""
    s, e = span
    return reads_as_discussion(text[max(0, s - _CUE_REACH) : s] + " " + text[e : e + _CUE_REACH])


def directs_at_quote(unquoted: str) -> bool:
    """The text around the quote tells the model to act on it."""
    return _DIRECTIVE.search(unquoted) is not None
