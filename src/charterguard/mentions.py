"""Use versus mention: attack phrases quoted in order to ask about them.

"How does prompt injection work? Explain 'ignore previous instructions'
attacks." contains the most classic override phrase there is, and blocking it
fails exactly the users -- security engineers, students, red-teamers -- most
likely to adopt a guardrail library. The phrase is *mentioned*, not *used*.

The distinction is only drawn on the user's own turn (``Layer.INPUT``). In
retrieved documents and tool output, quoting is a free evasion -- an attacker
controls the quotes -- so untrusted layers are never dampened.

Even on input, dampening must not become a wrapper an attacker can put around a
real payload. Earlier designs used a *denylist* of directives ("do this",
"follow") and a loose "discussion cue nearby" test; both could be phrased
around. This one **fails closed** on an allowlist:

* exactly one quoted span holds injection/jailbreak findings, it is short, and
  the matched phrase covers most of it -- nothing extra rides along inside;
* a thorough scan of the quote finds exactly one rule and nothing outside the
  injection/jailbreak categories;
* the text **outside** the quote, normalized, must *fully match* one of a small
  set of question templates (`frame_is_a_question`): "explain 'X' [attacks]",
  "what does 'X' mean [, and how do I detect it]", "is 'X' a jailbreak",
  optionally after one short leading question. Anything else -- any extra
  clause at all -- keeps the original verdict.

Findings that qualify are lowered to MEDIUM, which flags rather than allows.
Since the frame is a fixed question, there is nothing outside the quote left to
act on, and the quote holds only the phrase itself.
"""
from __future__ import annotations

import re

from .normalize import normalize

__all__ = [
    "quoted_spans",
    "mask_spans",
    "keep_only_spans",
    "frame_is_a_question",
    "MAX_QUOTE_CHARS",
    "MIN_PHRASE_COVERAGE",
]

# A quoted *mention* is a phrase, not a paragraph.
MAX_QUOTE_CHARS = 120
# Share of the quote's content that the matched phrase must cover.
MIN_PHRASE_COVERAGE = 0.6

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

# Placeholder for the quote inside the frame. Lowercase ASCII so it survives
# normalization unchanged.
_Q = "qqquotedqqq"

# ---------------------------------------------------------- frame templates
# Matched against the NORMALIZED frame (lowercased, folded, whitespace
# collapsed) with fullmatch: the whole remainder must be one of these shapes.
_NOUN = r"(?:phrase|prompt|string|payload|attack|example|text|sentence|input|message)"
_TOPIC = (
    r"(?:attacks?|injections?|prompt\s+injections?|jailbreaks?|prompts?|payloads?|"
    r"examples?|attempts?|techniques?)"
)
# One optional short leading question, e.g. "how does prompt injection work?"
_LEAD = r"(?:(?:how|what|why|is|are|can|does|do|which)\b[^?.!\n]{0,80}\?\s+)?"
_POLITE = r"(?:(?:please|can\s+you|could\s+you|would\s+you)\s+)?"
_DETECT = (
    r"(?:,?\s*and\s+(?:how\s+(?:do|can|should|would)\s+(?:i|we|you)\s+)?"
    r"(?:detect|block|stop|prevent|handle|flag|catch|defend\s+against)\s+(?:it|this|that|them))?"
)
_END = r"\s*[?.!]?"
_TEMPLATES = re.compile(
    "|".join(
        (
            # explain / define / describe 'X' [attacks]
            rf"{_LEAD}{_POLITE}(?:explain|define|describe|analy[sz]e|classify)\s+(?:the\s+)?(?:{_NOUN}\s+)?"
            rf"{_Q}(?:\s+{_TOPIC})?{_DETECT}{_END}",
            # what does 'X' mean / do
            rf"{_LEAD}what\s+(?:does|do|would)\s+(?:the\s+)?(?:{_NOUN}\s+)?{_Q}\s+(?:mean|do){_DETECT}{_END}",
            # what is / why is 'X' [a jailbreak|an attack|dangerous]
            rf"{_LEAD}(?:what|why)\s+is\s+(?:the\s+)?(?:{_NOUN}\s+)?{_Q}"
            rf"(?:\s+(?:an?\s+)?(?:attack|injection|jailbreak|dangerous|harmful|risky))?{_DETECT}{_END}",
            # is 'X' a jailbreak / an injection
            rf"{_LEAD}is\s+(?:the\s+)?(?:{_NOUN}\s+)?{_Q}\s+(?:an?\s+)?"
            rf"(?:attack|injection|prompt\s+injection|jailbreak|dangerous|harmful|safe){_DETECT}{_END}",
            # how does 'X' work [as an attack]
            rf"{_LEAD}how\s+(?:does|do|would)\s+(?:the\s+)?(?:{_NOUN}\s+)?{_Q}\s+work"
            rf"(?:\s+as\s+an?\s+(?:attack|injection|jailbreak))?{_DETECT}{_END}",
            # how (do|can) I (detect|block|...) 'X' [attacks]
            rf"{_LEAD}how\s+(?:do|can|should)\s+(?:i|we)\s+(?:detect|block|stop|prevent|handle|flag|catch|"
            rf"defend\s+against)\s+(?:the\s+)?(?:{_NOUN}\s+)?{_Q}(?:\s+{_TOPIC})?{_END}",
        )
    )
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


def frame_is_a_question(text: str, span: tuple[int, int]) -> bool:
    """True only when everything outside the quote is an allowlisted question.

    The frame is normalized first, so casing, confusables and invisible
    characters cannot be used to slip extra words past the template.
    """
    s, e = span
    frame = normalize(f"{text[:s]} {_Q} {text[e:]}").text.strip()
    return _TEMPLATES.fullmatch(frame) is not None
