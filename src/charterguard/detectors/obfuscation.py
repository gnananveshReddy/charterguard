"""Obfuscation detector: leetspeak, letter spacing, hyphenation, rot13.

The normalizer folds what has a *canonical* form -- confusables, invisible
characters, case, diacritics. It deliberately does not fold obfuscations that
are only meaningful to a reader:

* leetspeak (``1gn0r3 4ll pr3v10us``) -- folding digits globally would destroy
  the secret and PII rules, which depend on them;
* letter spacing (``i g n o r e``, ``i.g.n.o.r.e``) -- collapsing single
  letters in the shared pass would corrupt initials, list markers and code;
* hyphenated line breaks (``prev-\\nious``);
* rot13, which is a cipher rather than an encoding, so the encoded-payload
  detector's printable-ratio gate cannot tell it from prose.

This detector works on the **normalized** text, so it inherits confusable
folding, invisible-character stripping and fullwidth folding: ``１gn０r３`` and
``i\\u200b g n o r e`` reach it as plain ASCII. (Working on the original text
instead would be a parser differential -- every normalizer defence would be
one character away from bypassing this detector.) It builds a separate
de-obfuscated view with a per-character map back to the original, re-scans
only windows around the characters it changed with the instruction-shaped
rule categories, and reports a match only if it overlaps a change. The
concealment is itself the signal, so severity is raised the same way the
encoded detector raises it.

Word gaps survive normalization even though whitespace runs are collapsed: a
collapsed run shows up in the offset map as a source jump greater than one.
Letter spacing uses that to keep ``i g n o r e  a l l`` as two words.

Cost control: each technique has a C-speed gate on a byte "shape" of the text,
and work is proportional to the obfuscated region, not the document.

Limits: words inside a letter-spaced run must be separated by something other
than a single space; ``i g n o r e a l l`` cannot be segmented without a
dictionary and is left to the model tier.
"""
from __future__ import annotations

import codecs
import re
from bisect import bisect_left
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..config import GuardrailConfig
from ..normalize import NormalizedText, normalize
from ..patterns import rules_for
from ..scanner import compile_rules
from ..types import Category, Finding, Layer, Severity

__all__ = ["ObfuscationDetector", "DeobfuscatedView", "deobfuscate"]

_CATEGORIES = (Category.INJECTION, Category.JAILBREAK, Category.EXFIL, Category.COMMAND)

# Characters of context scanned on each side of a changed character. Rule
# patterns are phrase-length, so this comfortably contains any match that
# overlaps the change.
_WINDOW = 160
_MAX_FINDINGS = 8


# ------------------------------------------------------------------ gates
# Python's `re` costs ~0.1 us/char on patterns that start with a character
# class, which would make the gates alone more expensive than the rule scan.
# So gating runs on a byte-level "shape" of the text built by one
# bytes.translate (C speed): letters -> 'a', leet digits/symbols -> '1',
# separators -> '.', everything else -> ' '. Substring tests on the shape are
# necessary conditions; the regexes below run only once a gate has opened.
def _shape_table() -> bytes:
    table = bytearray(b" " * 256)
    for c in b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
        table[c] = ord("a")
    for c in b"013457@$":
        table[c] = ord("1")
    for c in b" ._*|-":
        table[c] = ord(".")
    return bytes(table)


_SHAPE = _shape_table()
# Letter-leet-letter or leet-letter-leet somewhere ("pr3v", "1gn0r3"). Weaker
# shapes such as leet-letter-letter would open on every "3rd" and "5th".
_LEET_SHAPES = (b"a1a", b"1a1")
# Two consecutive single-letter tokens. English has lone "a" and "I", rarely
# two in a row; "i g n o r e" has five.
_SPACED_SHAPE = b".a.a."
# The normalizer keeps a lone newline but collapses "-\r\n" or "- \n" to "- ".
_HYPHEN_MARKERS = ("-\n", "- ")
# Common words as they appear under rot13 (the text is already lowercased):
# "the", "and", "you", "all", "ignore", "previous", "instructions", "system",
# "prompt".
_ROT13_MARKERS = (" gur ", " naq ", " lbh ", " nyy ", "vtaber", "cerivbhf", "vafgehpgvbaf", "flfgrz", "cebzcg")

_SPACED_RUN = re.compile(r"(?<![a-z])(?:[a-z][ ._*|\-]){2,}[a-z](?![a-z])")
_HYPHEN_BREAK = re.compile(r"[a-z]-[ \n][a-z]")
_ROT13_WORDS = re.compile(r"\b(?:gur|naq|lbh|vtaber|cerivbhf|vafgehpgvbaf|flfgrz|cebzcg|nyy)\b")

# "!" is deliberately not a leet character: it ends sentences, and folding it
# would turn "instructions!" into "instructionsi" and break the very match the
# view exists to make.
_LEET_TOKEN = re.compile(r"[a-z0-9@$]{3,}")
_LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


@dataclass(frozen=True, slots=True)
class DeobfuscatedView:
    """A de-obfuscated rendering of some text.

    ``source[i]`` is the index in the *original* text that produced ``text[i]``;
    ``changed`` lists (sorted) the indices of ``text`` that were translated or
    joined across removed characters.
    """

    technique: str
    text: str
    source: tuple[int, ...]
    changed: tuple[int, ...]


def _wide(src: Sequence[int], i: int) -> bool:
    """True when position ``i`` stands for more than one source character,
    i.e. the normalizer collapsed a whitespace run here."""
    return i + 1 < len(src) and src[i + 1] - src[i] > 1


def _join(
    text: str,
    src: list[int],
    changed: list[int],
    pattern: _Finder | re.Pattern[str],
    keep_sep: Callable[[int], bool],
) -> tuple[str, list[int], list[int]]:
    """Within each match drop separators, keeping letters and their source
    indices; ``keep_sep(i)`` keeps a separator that is really a word gap."""
    out: list[str] = []
    out_src: list[int] = []
    out_changed: list[int] = []
    old_changed = set(changed)
    length = 0
    pos = 0

    def copy(lo: int, hi: int) -> None:
        nonlocal length
        out.append(text[lo:hi])
        out_src.extend(src[lo:hi])
        if old_changed:  # usually empty: avoid a per-character Python loop
            out_changed.extend(length + (i - lo) for i in range(lo, hi) if i in old_changed)
        length += hi - lo

    for m in pattern.finditer(text):
        s, e = m.span()
        copy(pos, s)
        for i in range(s, e):
            ch = text[i]
            if ch.isalpha():
                out.append(ch)
                out_src.append(src[i])
                out_changed.append(length)
                length += 1
            elif keep_sep(i):
                out.append(" ")
                out_src.append(src[i])
                length += 1
        pos = e
    copy(pos, len(text))
    return "".join(out), out_src, out_changed


def _unleet(text: str) -> tuple[str, list[int]]:
    """Translate leet characters inside tokens that also contain letters.

    Pure-digit tokens (dates, amounts, ids) are left alone. Length-preserving,
    so the offset map is unchanged; returns the changed positions.
    """
    changed: list[int] = []

    def fix(m: re.Match[str]) -> str:
        token = m.group(0)
        if not any(c.isalpha() for c in token):
            return token
        fixed = token.translate(_LEET_MAP)
        if fixed != token:
            base = m.start()
            changed.extend(base + i for i, (a, b) in enumerate(zip(token, fixed, strict=True)) if a != b)
        return fixed

    return _LEET_TOKEN.sub(fix, text), changed


def deobfuscate(text: str, source: Sequence[int] | None = None) -> Iterator[DeobfuscatedView]:
    """Yield one view per applicable technique family.

    ``text`` should be normalized (lowercase, folded); ``source`` maps each of
    its characters to the original text and defaults to the identity. Spacing,
    hyphenation and leetspeak are combined into one view, because attackers
    combine them. rot13 is a separate view: applying it on top of the others
    would scramble them.
    """
    shape = b"." + text.encode("utf-8", "ignore").translate(_SHAPE) + b"."
    hyphen = any(marker in text for marker in _HYPHEN_MARKERS)
    spaced = _SPACED_SHAPE in shape
    leet = any(s in shape for s in _LEET_SHAPES)
    rot13 = any(marker in text for marker in _ROT13_MARKERS)
    if not (hyphen or spaced or leet or rot13):
        return
    base = list(source) if source is not None else list(range(len(text)))

    techniques: list[str] = []
    view, src = text, base
    changed: list[int] = []
    if hyphen:
        # Join only across a real line break: a newline, or a collapsed
        # whitespace run (CRLF after a hyphen normalizes to "- "). "well- known"
        # with a single space is left alone.
        breaks = _HyphenBreaks(src)
        if next(breaks.finditer(view), None) is not None:
            view, src, changed = _join(view, src, changed, breaks, lambda i: False)
            techniques.append("hyphenation")
    if spaced and _SPACED_RUN.search(view):
        cur_src = src
        view, src, changed = _join(
            view, src, changed, _SPACED_RUN, lambda i: view[i] == " " and _wide(cur_src, i)
        )
        techniques.append("letter-spacing")
    if leet:
        unleeted, leet_changed = _unleet(view)
        if leet_changed:
            view = unleeted
            changed = sorted(set(changed) | set(leet_changed))
            techniques.append("leetspeak")
    if techniques:
        yield DeobfuscatedView("+".join(techniques), view, tuple(src), tuple(sorted(changed)))

    if rot13:
        rot_words = [i for m in _ROT13_WORDS.finditer(text) for i in range(m.start(), m.end())]
        if rot_words:
            yield DeobfuscatedView("rot13", codecs.encode(text, "rot13"), tuple(base), tuple(rot_words))


class _HyphenBreaks:
    """Pattern-like: yields only hyphen breaks that sit on a real line break,
    so `_join` can stay generic."""

    __slots__ = ("_src",)

    def __init__(self, src: Sequence[int]) -> None:
        self._src = src

    def finditer(self, text: str) -> Iterator[re.Match[str]]:
        for m in _HYPHEN_BREAK.finditer(text):
            i = m.start() + 2
            if text[i] == "\n" or _wide(self._src, i):
                yield m


class _Finder(Protocol):
    def finditer(self, text: str) -> Iterator[re.Match[str]]: ...


def _windows(changed: tuple[int, ...], length: int) -> list[tuple[int, int]]:
    """Merge +-_WINDOW neighbourhoods of changed positions into disjoint spans."""
    spans: list[tuple[int, int]] = []
    for i in changed:
        lo, hi = max(0, i - _WINDOW), min(length, i + _WINDOW + 1)
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))
    return spans


def _overlaps(changed: tuple[int, ...], start: int, end: int) -> bool:
    i = bisect_left(changed, start)
    return i < len(changed) and changed[i] < end


class ObfuscationDetector:
    name = "obfuscation"

    __slots__ = ("_ruleset",)

    def __init__(self) -> None:
        self._ruleset = compile_rules(rules_for(layer=Layer.UNTRUSTED, categories=_CATEGORIES))

    def scan(self, nt: NormalizedText, layer: Layer, config: GuardrailConfig) -> list[Finding]:
        if not config.enable_obfuscation_scan or not nt.text:
            return []
        # The normalized text already applies the head+tail sampling budget
        # for oversized inputs, so there is nothing to re-sample here. The
        # dense source map is only built once a gate has opened.
        views = deobfuscate(nt.text, _LazySourceMap(nt))
        findings: list[Finding] = []
        reported: set[str] = set()
        source = nt.original
        for view in views:
            for lo, hi in _windows(view.changed, len(view.text)):
                inner = normalize(view.text[lo:hi], max_chars=len(view.text) + 1)
                for m in self._ruleset.scan(inner.text, limit=4):
                    vs, ve = inner.span(m.start, m.end)
                    vs, ve = lo + vs, lo + ve
                    if m.rule.id in reported or not _overlaps(view.changed, vs, ve):
                        continue
                    start = view.source[vs]
                    end = view.source[ve - 1] + 1
                    findings.append(
                        Finding(
                            rule_id=m.rule.id,
                            category=m.rule.category,
                            severity=Severity.CRITICAL
                            if m.rule.severity >= Severity.HIGH
                            else Severity.HIGH,
                            score=0.85,
                            detector=self.name,
                            start=start,
                            end=end,
                            excerpt=source[start:end][:200],
                            description=f"{m.rule.description} (after undoing {view.technique})",
                            decoded_from=view.technique,
                        )
                    )
                    reported.add(m.rule.id)
                    if len(findings) >= _MAX_FINDINGS:
                        return findings
        return findings


class _LazySourceMap(Sequence[int]):
    """Defers `NormalizedText.source_map()` until `deobfuscate` needs it, so
    clean text that opens no gate never pays for a dense map."""

    __slots__ = ("_nt", "_map")

    def __init__(self, nt: NormalizedText) -> None:
        self._nt = nt
        self._map: list[int] | None = None

    def _get(self) -> list[int]:
        if self._map is None:
            self._map = self._nt.source_map()
        return self._map

    def __len__(self) -> int:
        return len(self._nt.text)

    def __getitem__(self, i):  # type: ignore[no-untyped-def, override]
        return self._get()[i]

    def __iter__(self) -> Iterator[int]:
        return iter(self._get())
