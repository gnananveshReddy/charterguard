"""Offset-preserving normalization.

Every deterministic detector shares ONE normalization pass. Normalizing per
detector is the most common performance mistake in guardrail stacks.

Defeats: unicode confusables, zero-width and bidi smuggling, full-width and
mathematical-alphanumeric variants, diacritic decoration, case tricks, and
whitespace padding -- while keeping an exact map back to the original so audit
spans are real.

Performance design
------------------
The obvious implementation walks the string character by character and appends
to two Python lists. That costs ~700ns/char and makes guardrails the latency
story on any real document.

This implementation instead works in *slices*:

  * ASCII text is lowercased with `str.lower` (C speed, length-preserving) and
    copied in whole slices; only whitespace runs and control characters break a
    slice.
  * The offset map is a sparse list of (normalized_pos, source_pos) breakpoints
    resolved by bisect, not one entry per character. Clean prose produces a
    single breakpoint.
  * Per-character Unicode work is confined to actual non-ASCII runs, located
    with one regex pass.

Net effect: roughly two orders of magnitude faster on ASCII documents, with
identical outputs and exact span mapping.
"""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass

# Zero-width, soft hyphen, and bidi controls: pure smuggling characters.
_INVISIBLE = frozenset(
    "\u200b\u200c\u200d\u2060\ufeff\u00ad\u034f\u180e"
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)

# Confusable folding.
#
# Three layers, cheapest first:
#   1. an explicit table for lookalikes that NFKC leaves alone (Cyrillic,
#      Greek, Armenian, Cherokee, and the stroked Latin letters that have no
#      decomposition at all);
#   2. combining diacritics in U+0300-U+036F are dropped, which folds "ignoré"
#      to "ignore" -- deliberately limited to that block, because Devanagari,
#      Arabic and Thai vowel signs are also category Mn and are semantically
#      load-bearing, not decoration;
#   3. anything else falls back to NFKD, keeping the base character when that
#      base is ASCII. This covers the long tail (accented Latin, fullwidth,
#      mathematical alphanumerics, circled letters) without enumerating it.
#
# Results are memoized per code point, so the cost is paid once per distinct
# character per process, not once per occurrence.
_HOMOGLYPHS = {
    # Cyrillic -> Latin
    "\u0430": "a", "\u0432": "b", "\u0435": "e", "\u043a": "k", "\u043c": "m",
    "\u043d": "h", "\u043e": "o", "\u0440": "p", "\u0441": "c", "\u0442": "t",
    "\u0443": "y", "\u0445": "x", "\u0456": "i", "\u0457": "i", "\u0455": "s",
    "\u04bb": "h", "\u0458": "j", "\u0491": "r", "\u04cf": "l", "\u0501": "d",
    "\u051b": "q", "\u051d": "w", "\u0433": "r", "\u0438": "n", "\u043f": "n",
    # Greek -> Latin
    "\u03b1": "a", "\u03b2": "b", "\u03b5": "e", "\u03b7": "n", "\u03b9": "i",
    "\u03ba": "k", "\u03bd": "v", "\u03bf": "o", "\u03c1": "p", "\u03c4": "t",
    "\u03c5": "u", "\u03c7": "x", "\u03b3": "y", "\u03bc": "u", "\u03c3": "o",
    "\u0391": "a", "\u0392": "b", "\u0395": "e", "\u0397": "h", "\u0399": "i",
    "\u039a": "k", "\u039c": "m", "\u039d": "n", "\u039f": "o", "\u03a1": "p",
    "\u03a4": "t", "\u03a5": "y", "\u03a7": "x", "\u0396": "z",
    # Armenian / Cherokee / Coptic lookalikes
    "\u0570": "h", "\u0578": "n", "\u057c": "n", "\u0585": "o", "\u13a0": "d",
    "\u13aa": "s", "\u13b3": "w", "\u13c0": "g", "\u13cf": "b", "\u13de": "l",
    "\u2c9e": "o", "\u2c8f": "n",
    # Latin letters with strokes/ligatures: no NFKD decomposition exists
    "\u00f8": "o", "\u00d8": "o", "\u0111": "d", "\u0110": "d", "\u0142": "l",
    "\u0141": "l", "\u00e6": "ae", "\u00c6": "ae", "\u0153": "oe", "\u0152": "oe",
    "\u00f0": "d", "\u00d0": "d", "\u00fe": "th", "\u00de": "th", "\u0131": "i",
    "\u0130": "i", "\u017f": "s", "\u01c0": "l", "\u0294": "j", "\u0269": "i",
    # Punctuation that changes tokenisation
    "\u2019": "'", "\u2018": "'", "\u201b": "'", "\u2032": "'", "\u02bc": "'",
    "\u201c": '"', "\u201d": '"', "\u2033": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2010": "-", "\u2011": "-",
    "\u2044": "/", "\u2236": ":", "\uff1a": ":", "\u060c": ",",
}

# Combining diacritics: Latin/Greek/Cyrillic decoration only.
_COMBINING_LO, _COMBINING_HI = 0x300, 0x36F

_FOLD_CACHE: dict[str, str] = {}


def _fold_char(ch: str) -> str:
    """Memoized confusable fold for one non-ASCII character."""
    cached = _FOLD_CACHE.get(ch)
    if cached is not None:
        return cached
    # Case-fold first: the table is keyed on lowercase, so an uppercase
    # confusable ('\u0410') would otherwise fold only to its lowercase Cyrillic
    # form on the first pass and to 'a' on a second pass, making normalization
    # non-idempotent -- which desynchronises the content-hash cache.
    lowered = ch.casefold()
    if len(lowered) == 1:
        explicit = _HOMOGLYPHS.get(lowered)
    elif any(c in _HOMOGLYPHS for c in lowered):
        explicit = "".join(_HOMOGLYPHS.get(c, c) for c in lowered)
    else:
        explicit = None
    if explicit is not None:
        folded = explicit
    else:
        nfkc = unicodedata.normalize("NFKC", lowered)
        stripped = "".join(
            c
            for c in unicodedata.normalize("NFKD", nfkc)
            if not (_COMBINING_LO <= ord(c) <= _COMBINING_HI)
        )
        # Keep the ASCII base when there is one; otherwise leave the script
        # alone so non-Latin text stays matchable by non-Latin rules.
        folded = (stripped if stripped and stripped.isascii() else nfkc).casefold()
    _FOLD_CACHE[ch] = folded
    return folded


# Slice breakers in ASCII text: multi-character whitespace runs (collapsed to a
# single space) and control characters (dropped). A lone space, tab, or newline
# is left inside the slice -- every rule pattern uses \s, so preserving a single
# newline is both cheaper and strictly better for line-anchored rules.
_ASCII_BREAK = re.compile(r"\s{2,}|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
_NON_ASCII = re.compile(r"[^\x00-\x7f]+")


@dataclass(frozen=True, slots=True)
class NormalizedText:
    original: str
    text: str
    truncated: bool = False
    invisible_chars: int = 0
    confusable_chars: int = 0
    # Sparse offset map. `None` means the identity mapping (no breakpoints).
    _norm_pos: tuple[int, ...] = ()
    _src_pos: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.text)

    def source_index(self, index: int) -> int:
        """Map one normalized index back to its index in `original`."""
        if not self._norm_pos:
            return min(max(index, 0), max(len(self.original) - 1, 0))
        i = bisect_right(self._norm_pos, index) - 1
        if i < 0:
            return 0
        delta = index - self._norm_pos[i]
        return min(self._src_pos[i] + delta, max(len(self.original) - 1, 0))

    def span(self, start: int, end: int) -> tuple[int, int]:
        """Map a normalized span back to an original span."""
        if not self.original:
            return (0, 0)
        s = self.source_index(start)
        e = self.source_index(max(end - 1, start)) + 1
        return (s, max(e, s + 1))

    def excerpt(self, start: int, end: int, pad: int = 24, limit: int = 200) -> str:
        s, e = self.span(start, end)
        lo = max(0, s - pad)
        hi = min(len(self.original), e + pad)
        return self.original[lo:hi][:limit]


class _Builder:
    """Accumulates normalized slices plus a sparse offset map."""

    __slots__ = ("parts", "norm_pos", "src_pos", "length", "last_space", "invisible", "confusable")

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.norm_pos: list[int] = []
        self.src_pos: list[int] = []
        self.length = 0
        self.last_space = True  # suppress leading whitespace
        self.invisible = 0
        self.confusable = 0

    def add_slice(self, chunk: str, src_start: int) -> None:
        if not chunk:
            return
        # Only record a breakpoint when the mapping actually shifts.
        if not self.norm_pos or self.src_pos[-1] + (self.length - self.norm_pos[-1]) != src_start:
            self.norm_pos.append(self.length)
            self.src_pos.append(src_start)
        self.parts.append(chunk)
        self.length += len(chunk)
        self.last_space = chunk[-1].isspace()

    def add_space(self, src_index: int) -> None:
        if self.last_space:
            return
        self.add_slice(" ", src_index)
        self.last_space = True


def _feed_ascii(b: _Builder, chunk: str, base: int) -> None:
    lowered = chunk.lower()  # length-preserving for ASCII
    pos = 0
    for m in _ASCII_BREAK.finditer(chunk):
        s, e = m.span()
        if s > pos:
            b.add_slice(lowered[pos:s], base + pos)
        token = m.group(0)
        if token[0].isspace():
            b.add_space(base + s)
        else:
            b.invisible += len(token)
        pos = e
    if pos < len(chunk):
        b.add_slice(lowered[pos:], base + pos)


def _feed_unicode(b: _Builder, chunk: str, base: int) -> None:
    """Per-character path, entered only for genuinely non-ASCII runs."""
    for i, ch in enumerate(chunk):
        if ch in _INVISIBLE:
            b.invisible += 1
            continue
        if unicodedata.category(ch) in ("Cc", "Cf"):
            b.invisible += 1
            continue
        if _COMBINING_LO <= ord(ch) <= _COMBINING_HI:
            b.confusable += 1  # decomposed accent: fold away the mark
            continue
        pieces = _fold_char(ch)
        if pieces != ch:
            b.confusable += 1
        for piece in pieces:
            if piece.isspace():
                b.add_space(base + i)
            else:
                # One breakpoint per expanded character keeps mapping exact.
                b.add_slice(piece, base + i)


def _feed(b: _Builder, chunk: str, base: int) -> None:
    if chunk.isascii():
        _feed_ascii(b, chunk, base)
        return
    pos = 0
    for m in _NON_ASCII.finditer(chunk):
        s, e = m.span()
        if s > pos:
            _feed_ascii(b, chunk[pos:s], base + pos)
        _feed_unicode(b, chunk[s:e], base + s)
        pos = e
    if pos < len(chunk):
        _feed_ascii(b, chunk[pos:], base + pos)


def normalize(text: str, *, max_chars: int = 200_000, tail_chars: int = 8_000) -> NormalizedText:
    """Normalize `text`, sampling head+tail if it exceeds the budget.

    Oversized inputs are sampled rather than truncated because payloads are
    frequently appended at the very end of a document.
    """
    n = len(text)
    truncated = n > max_chars
    b = _Builder()

    if not truncated:
        _feed(b, text, 0)
    else:
        head_len = max(max_chars - tail_chars, 0)
        tail_start = max(n - tail_chars, head_len)
        _feed(b, text[:head_len], 0)
        b.add_space(tail_start)  # boundary marker: no pattern may span the gap
        b.last_space = True
        _feed(b, text[tail_start:], tail_start)

    return NormalizedText(
        original=text,
        text="".join(b.parts),
        truncated=truncated,
        invisible_chars=b.invisible,
        confusable_chars=b.confusable,
        _norm_pos=tuple(b.norm_pos),
        _src_pos=tuple(b.src_pos),
    )
