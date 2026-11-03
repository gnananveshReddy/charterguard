"""Literal prefilter: the difference between 0.5 MB/s and 20 MB/s.

The problem
-----------
The natural way to run ~50 detection rules is to compile them into one big
alternation and call `finditer`. Python's `re` engine has no multi-pattern
automaton: at every character position it re-tries every branch. Measured on a
4 KB document with a 40-branch alternation that costs ~1.5 us *per character* --
about 9 ms per document, which makes the guardrail the dominant latency in an
agent turn and effectively forbids scanning retrieved context.

The approach
------------
Almost every rule contains a mandatory literal ("instructions", "exfiltrate",
"akia", "jailbr"). If that literal is absent from the text, the rule *cannot*
match, so running its regex is wasted work.

So this module derives, statically at import time, a **necessary condition** for
each rule:

  * `required_literals` parses the pattern and returns a set of literals of
    which at least one must be present for any match to exist. It is sound by
    construction: alternation contributes the union of its branches (all
    branches must yield something, or the node yields nothing), optional and
    star-quantified nodes contribute nothing, and concatenation may pick any one
    of its children's conditions.
  * `split_alternation` first splits a rule's top-level alternation into
    branches so each branch gets its own, much sharper, condition. Without this
    the broad "intent" rules degrade to no condition at all.

All literals across all rules are then compiled into a single **trie regex**, so
one pass over the text yields the set of literals present, along with their
positions. Rules whose condition is unsatisfied are skipped entirely; rules that
survive are re-scanned only in windows around the literal occurrence that
triggered them.

Soundness is not assumed: `tests/test_prefilter.py` runs the whole attack corpus
with the prefilter enabled and disabled and asserts identical findings.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Literals shorter than this are too common to be selective ("do", "on", "is").
_MIN_LITERAL = 3

# When no literal of _MIN_LITERAL length is mandatory, a single mandatory symbol
# still prefilters well -- "@" for email, for instance. Common punctuation is
# excluded because it appears in all prose and would filter nothing.
_USEFUL_SYMBOLS = frozenset("@$#%&*+=/\\<>{}[]|~^")

# High-frequency English words make poor triggers: they fire on ordinary prose,
# so the rule gets rescanned constantly and the trie does needless work. When a
# branch offers a choice of necessary conditions, anything outside this set is
# preferred. This only affects *which* valid condition is chosen, never whether
# the condition is sound.
_COMMON_WORDS = frozenset(["about", "all", "and", "answer", "any", "are", "but", "call", "can", "code", "describe", "did", "does", "end", "explain", "following", "for", "from", "get", "give", "had", "has", "have", "help", "here", "how", "into", "its", "just", "like", "make", "more", "most", "much", "need", "new", "not", "now", "one", "only", "other", "our", "out", "own", "plan", "planning", "post", "relative", "respond", "responses", "run", "said", "same", "say", "see", "send", "should", "show", "some", "such", "take", "tell", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "told", "use", "used", "using", "very", "want", "was", "way", "were", "what", "when", "which", "who", "why", "will", "with", "without", "work", "would", "write", "you", "your", "yourself"])

# Character classes that terminate a literal run.
_NON_LITERAL_ESCAPES = frozenset("dDwWsSbBAZzGnrtvf0123456789")

# How far around a triggering literal a match may plausibly extend. Every rule
# in the shipped set matches far less than this; the window is also expanded to
# word boundaries so \b keeps its meaning at the edges.
WINDOW = 320

# Above this many occurrences of one literal, windowing stops paying for itself
# and the variant is scanned over the whole text instead.
MAX_OCCURRENCES = 24

_DIGIT_RUN = re.compile(r"\d+")


class _Cursor:
    __slots__ = ("s", "i")

    def __init__(self, s: str) -> None:
        self.s = s
        self.i = 0

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def done(self) -> bool:
        return self.i >= len(self.s)


class _Unsupported(Exception):
    """A regex construct the literal parser cannot reason about soundly."""


# Inline flag groups: "(?i)" (global) or "(?i:" / "(?-i:" (scoped).
_INLINE_FLAGS = re.compile(r"\?[aiLmsux]*(?:-[imsx]+)?[:)]")


def _quantifier(c: _Cursor) -> bool:
    """Consume a quantifier if present. Return True if the atom stays mandatory.

    Membership tests use tuples deliberately: `"" in "*?"` is True in Python, so
    a string test silently treats end-of-pattern as an optional quantifier and
    discards the last literal of every pattern.
    """
    ch = c.peek()
    if ch in ("*", "?"):
        c.i += 1
        if c.peek() in ("?", "+"):
            c.i += 1
        return False
    if ch == "+":
        c.i += 1
        if c.peek() in ("?", "+"):
            c.i += 1
        return True
    if ch == "{":
        close = c.s.find("}", c.i)
        if close == -1:
            return True
        body = c.s[c.i + 1 : close]
        c.i = close + 1
        if c.peek() in ("?", "+"):
            c.i += 1
        return not body.startswith("0")
    return True


def _parse_alternation(c: _Cursor) -> frozenset[str] | None:
    branches = [_parse_concat(c)]
    while c.peek() == "|":
        c.i += 1
        branches.append(_parse_concat(c))
    if len(branches) == 1:
        return branches[0]
    merged: set[str] = set()
    for b in branches:
        if not b:
            # One branch with no necessary literal makes the whole node
            # unconstrained -- anything else would be unsound.
            return None
        merged |= b
    return frozenset(merged)


def _parse_concat(c: _Cursor) -> frozenset[str] | None:
    conditions: list[frozenset[str]] = []
    symbols: list[frozenset[str]] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            literal = "".join(run)
            if len(literal) >= _MIN_LITERAL:
                conditions.append(frozenset({literal}))
            elif len(literal) == 1 and literal in _USEFUL_SYMBOLS:
                symbols.append(frozenset({literal}))
            run.clear()

    while not c.done() and c.peek() not in ("|", ")"):
        ch = c.peek()
        if ch == "(":
            flush()
            c.i += 1
            capturing_group = True
            if c.peek() == "?":
                # Every extension group is handled explicitly. Anything not
                # recognised makes the pattern unconstrained: treating its
                # syntax as literal text (as "(?<!" once was) is unsound.
                if c.s.startswith("?:", c.i) or c.s.startswith("?>", c.i):
                    c.i += 2
                elif c.s.startswith(("?<=", "?<!"), c.i):
                    c.i += 3
                    capturing_group = False  # lookbehind: contributes nothing
                elif c.s.startswith(("?=", "?!"), c.i):
                    c.i += 2
                    capturing_group = False  # lookahead: contributes nothing
                elif c.s.startswith("?P<", c.i) or (
                    c.s.startswith("?<", c.i) and c.s[c.i + 2 : c.i + 3] not in ("=", "!")
                ):
                    c.i = c.s.index(">", c.i) + 1
                elif c.s.startswith("?#", c.i):
                    c.i = c.s.index(")", c.i) + 1  # comment group
                    continue
                elif (flags := _INLINE_FLAGS.match(c.s, c.i)) is not None:
                    c.i = flags.end()
                    if c.s[c.i - 1] == ")":
                        continue  # global flags such as (?i): no group body
                    # Scoped flags change how the body matches (case, etc.);
                    # keep parsing for structure but never use its literals.
                    capturing_group = False
                else:
                    raise _Unsupported(c.s[c.i : c.i + 4])
            inner = _parse_alternation(c)
            if c.peek() == ")":
                c.i += 1
            mandatory = _quantifier(c)
            if inner and mandatory and capturing_group:
                conditions.append(inner)
        elif ch == "[":
            flush()
            c.i += 1
            if c.peek() == "^":
                c.i += 1
            while not c.done() and c.peek() != "]":
                c.i += 2 if c.peek() == "\\" else 1
            c.i += 1
            _quantifier(c)
        elif ch == "\\":
            nxt = c.s[c.i + 1 : c.i + 2]
            if nxt in _NON_LITERAL_ESCAPES:
                flush()
                c.i += 2
                _quantifier(c)
            else:
                c.i += 2
                if _quantifier(c):
                    run.append(nxt)   # escaped literal such as \. or \+
                else:
                    flush()
        elif ch in ".^$":
            flush()
            c.i += 1
            _quantifier(c)
        else:
            c.i += 1
            if _quantifier(c):
                run.append(ch)
            else:
                flush()
    flush()

    if not conditions:
        # Fall back to a mandatory symbol before giving up entirely.
        return symbols[0] if symbols else None
    # Prefer the sharpest necessary condition: avoid common English words first,
    # then fewest alternatives, then the longest guaranteed literal.
    def sharpness(candidate: frozenset[str]) -> tuple[int, int, int]:
        common = sum(1 for x in candidate if x in _COMMON_WORDS)
        return (common, len(candidate), -min(len(x) for x in candidate))

    return min(conditions, key=sharpness)


def required_literals(pattern: str) -> frozenset[str] | None:
    """A set of literals of which at least one must appear in any match.

    Returns None when no such condition can be proven, in which case the caller
    must scan unconditionally.
    """
    try:
        return _parse_alternation(_Cursor(pattern))
    except (_Unsupported, ValueError):
        # ValueError: an unterminated "(?P<" or "(?#" -- also fail open to a
        # full scan rather than guess.
        return None


def split_alternation(pattern: str) -> list[str]:
    """Split a pattern into top-level alternation branches.

    Unwraps a single enclosing `(?: ... )` first, since that is how the rules in
    this package are written.
    """
    s = pattern
    while s.startswith("(?:") and s.endswith(")"):
        depth = 0
        wraps_whole = True
        escaped = False
        for i, ch in enumerate(s):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    wraps_whole = False
                    break
        if not wraps_whole:
            break
        s = s[3:-1]

    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    escaped = False
    in_class = False
    for ch in s:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            buf.append(ch)
            escaped = True
            continue
        if in_class:
            buf.append(ch)
            if ch == "]":
                in_class = False
            continue
        if ch == "[":
            buf.append(ch)
            in_class = True
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return [p for p in parts if p] or [s]


def _trie_pattern(words: list[str]) -> str:
    """Compile literals into a shared-prefix trie regex: one pass, not N."""
    root: dict[str, object] = {}
    for w in words:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})  # type: ignore[assignment]
        node["$"] = True

    def emit(node: dict[str, object]) -> str:
        alts: list[str] = []
        terminal = False
        for ch, sub in sorted(node.items()):
            if ch == "$":
                terminal = True
                continue
            alts.append(re.escape(ch) + emit(sub))  # type: ignore[arg-type]
        if not alts:
            return ""
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        if not terminal:
            return body
        # A literal that is a prefix of another ends here, so the remainder is
        # optional. The group is mandatory: appending a bare "?" to `body` would
        # quantify only its final token -- and if that token is already a
        # quantifier, would silently make it lazy instead.
        return "(?:" + body + ")?"

    return emit(root)


@dataclass(frozen=True, slots=True)
class Hits:
    """Which trigger literals are present, and where."""

    literals: frozenset[str]
    positions: dict[str, tuple[int, ...]]
    saturated: frozenset[str]        # literals too frequent to window on
    digit_positions: tuple[int, ...] = ()
    digit_saturated: bool = False

    @property
    def has_digit(self) -> bool:
        return bool(self.digit_positions) or self.digit_saturated


class LiteralIndex:
    """One trie regex over every trigger literal in a rule set."""

    __slots__ = ("_trie", "_prefixes", "_needs_digit", "literal_count")

    def __init__(self, literals: set[str], *, needs_digit: bool) -> None:
        words = sorted(literals)
        self.literal_count = len(words)
        self._trie = re.compile(_trie_pattern(words)) if words else None
        # A trie match reports the longest accepted literal, so "instructions"
        # must also register the shorter literal "instruction" that it contains.
        self._prefixes = {
            w: tuple(sorted({p for p in literals if w.startswith(p)})) for w in words
        }
        self._needs_digit = needs_digit

    def probe(self, text: str) -> Hits:
        # Digit-gated rules (payment cards, Aadhaar, SSN, phone numbers) have no
        # literal to key on. Locating digit *runs* rather than just asking
        # "any digit?" matters: almost every real document contains a number
        # somewhere, and a boolean gate would put all of those rules back on the
        # full-document path.
        digits: list[int] = []
        digit_saturated = False
        if self._needs_digit:
            for m in _DIGIT_RUN.finditer(text):
                if len(digits) >= MAX_OCCURRENCES:
                    digit_saturated = True
                    break
                digits.append(m.start())

        if self._trie is None:
            return Hits(frozenset(), {}, frozenset(), tuple(digits), digit_saturated)

        # Collect raw trie matches first and resolve shorter contained literals
        # afterwards, once per distinct matched literal rather than once per
        # match. On prose with a few hundred incidental hits that is the
        # difference between ~4 dict operations per match and ~2.
        raw: dict[str, list[int]] = {}
        raw_saturated: set[str] = set()
        for m in self._trie.finditer(text):
            key = m.group(0)
            slot = raw.get(key)
            if slot is None:
                raw[key] = [m.start()]
            elif len(slot) < MAX_OCCURRENCES:
                slot.append(m.start())
            else:
                raw_saturated.add(key)

        found: dict[str, list[int]] = {}
        saturated: set[str] = set()
        for key, offsets in raw.items():
            for literal in self._prefixes.get(key, (key,)):
                if key in raw_saturated:
                    saturated.add(literal)
                slot = found.get(literal)
                if slot is None:
                    found[literal] = list(offsets)
                else:
                    slot.extend(offsets)
        for literal, slot in found.items():
            if len(slot) > MAX_OCCURRENCES:
                saturated.add(literal)
            else:
                slot.sort()
        return Hits(
            literals=frozenset(found),
            positions={k: tuple(v) for k, v in found.items()},
            saturated=frozenset(saturated),
            digit_positions=tuple(digits),
            digit_saturated=digit_saturated,
        )


def _is_word_char(ch: str) -> bool:
    """Alphanumerics, underscore, and combining marks.

    Indic vowel signs and viramas are category M, not alphanumeric; without
    them a window edge could split "निर्देश" between a consonant and its
    vowel sign.
    """
    return ch.isalnum() or ch == "_" or unicodedata.category(ch).startswith("M")


def windows(text: str, offsets: tuple[int, ...]) -> list[tuple[int, int]]:
    """Merged, word-boundary-safe scan windows around trigger offsets.

    Expanding to a non-word character on both sides matters: `re.search` treats
    `endpos` as end-of-string, so a window cutting through a word would let `\\b`
    match mid-word and manufacture a false positive.
    """
    n = len(text)
    spans: list[tuple[int, int]] = []
    for off in offsets:
        lo = max(0, off - WINDOW)
        hi = min(n, off + WINDOW)
        while lo > 0 and _is_word_char(text[lo - 1]):
            lo -= 1
        while hi < n and _is_word_char(text[hi]):
            hi += 1
        if spans and lo <= spans[-1][1]:
            prev_lo, prev_hi = spans[-1]
            spans[-1] = (prev_lo, max(prev_hi, hi))
        else:
            spans.append((lo, hi))
    return spans
