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

This detector builds a *separate* de-obfuscated view of the original text,
keeping a per-character map back to it, and re-scans only windows around the
characters it changed, with only the instruction-shaped rule categories. A
match is reported only if it overlaps a changed character -- the concealment
is what made it invisible to the rule detector, and is itself the signal, so
severity is raised the same way the encoded detector raises it.

Cost control: each technique has a one-pass C-speed gate, and work is
proportional to the obfuscated region, not the document. Ordinary prose trips
no gate and pays a handful of regex searches; a stray ``mp3`` or ``v2`` opens
the leetspeak gate but costs one small window.

Limits: letter spacing is undone within a run of single letters. Words must
still be separated by something else (two spaces, a newline, punctuation);
``i g n o r e a l l`` with uniform single spaces cannot be segmented without a
dictionary, and is left to the model tier.
"""
from __future__ import annotations

import codecs
import re
from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass

from ..config import GuardrailConfig
from ..normalize import normalize
from ..normalize import NormalizedText
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
_HYPHEN_MARKERS = ("-\n", "-\r\n", "- \n")
# Common words as they appear under rot13, matched on lowercased text: "the",
# "and", "you", "all", "ignore", "previous", "instructions", "system", "prompt".
_ROT13_MARKERS = (" gur ", " naq ", " lbh ", " nyy ", "vtaber", "cerivbhf", "vafgehpgvbaf", "flfgrz", "cebzcg")

_SPACED_RUN = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][ ._*|\-]){2,}[A-Za-z](?![A-Za-z])")
_HYPHEN_BREAK = re.compile(r"[A-Za-z]-[ \t]*\r?\n[ \t]*[A-Za-z]")
_ROT13_WORDS = re.compile(
    r"\b(?:gur|naq|lbh|vtaber|cerivbhf|vafgehpgvbaf|flfgrz|cebzcg|nyy)\b", re.IGNORECASE
)

# "!" is deliberately not a leet character: it ends sentences, and folding it
# would turn "instructions!" into "instructionsi" and break the very match the
# view exists to make.
_LEET_TOKEN = re.compile(r"[A-Za-z0-9@$]{3,}")
_LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_LEET_CHARS = frozenset("013457@$")


@dataclass(frozen=True, slots=True)
class DeobfuscatedView:
    """A de-obfuscated rendering of some text.

    ``source[i]`` is the index in the input that produced ``text[i]``;
    ``changed`` lists (sorted) the indices of ``text`` that differ from, or
    were joined across removed characters of, the input.
    """

    technique: str
    text: str
    source: tuple[int, ...]
    changed: tuple[int, ...]


def _keep_letters(
    text: str, src: list[int], changed: list[int], pattern: re.Pattern[str]
) -> tuple[str, list[int], list[int]]:
    """Within each match keep only the letters, preserving their source index."""
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
            if text[i].isalpha():
                out.append(text[i])
                out_src.append(src[i])
                out_changed.append(length)
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


def deobfuscate(text: str) -> Iterator[DeobfuscatedView]:
    """Yield one view per applicable technique family.

    Spacing, hyphenation and leetspeak are combined into one view, because
    attackers combine them. rot13 is a separate view: applying it on top of the
    others would scramble them.
    """
    shape = b"." + text.encode("utf-8", "ignore").translate(_SHAPE) + b"."
    hyphen = any(marker in text for marker in _HYPHEN_MARKERS)
    spaced = _SPACED_SHAPE in shape
    leet = any(s in shape for s in _LEET_SHAPES)

    techniques: list[str] = []
    view = text
    src = list(range(len(text))) if (hyphen or spaced or leet) else []
    changed: list[int] = []
    if hyphen and _HYPHEN_BREAK.search(view):
        view, src, changed = _keep_letters(view, src, changed, _HYPHEN_BREAK)
        techniques.append("hyphenation")
    if spaced and _SPACED_RUN.search(view):
        view, src, changed = _keep_letters(view, src, changed, _SPACED_RUN)
        techniques.append("letter-spacing")
    if leet:
        unleeted, leet_changed = _unleet(view)
        if leet_changed:
            view = unleeted
            changed = sorted(set(changed) | set(leet_changed))
            techniques.append("leetspeak")
    if techniques:
        yield DeobfuscatedView("+".join(techniques), view, tuple(src), tuple(sorted(changed)))

    lowered = text.lower()
    if not any(marker in lowered for marker in _ROT13_MARKERS):
        return
    rot_words = [i for m in _ROT13_WORDS.finditer(text) for i in range(m.start(), m.end())]
    if rot_words:
        yield DeobfuscatedView("rot13", codecs.encode(text, "rot13"), tuple(range(len(text))), tuple(rot_words))


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
        if not config.enable_obfuscation_scan or not nt.original:
            return []
        budget = config.budget
        source = nt.original
        segments = [(0, source)]
        if len(source) > budget.max_chars:
            # Same head+tail sampling as the normalizer: payloads are often
            # appended at the very end of a long document.
            tail_start = len(source) - budget.tail_chars
            segments = [
                (0, source[: budget.max_chars - budget.tail_chars]),
                (tail_start, source[tail_start:]),
            ]

        findings: list[Finding] = []
        reported: set[str] = set()
        for offset, segment in segments:
            for view in deobfuscate(segment):
                for lo, hi in _windows(view.changed, len(view.text)):
                    inner = normalize(view.text[lo:hi], max_chars=budget.max_chars)
                    for m in self._ruleset.scan(inner.text, limit=4):
                        vs, ve = inner.span(m.start, m.end)
                        vs, ve = lo + vs, lo + ve
                        if m.rule.id in reported or not _overlaps(view.changed, vs, ve):
                            continue
                        start = offset + view.source[vs]
                        end = offset + view.source[ve - 1] + 1
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
