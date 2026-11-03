"""Prefiltered multi-pattern scan engine.

Four properties this engine exists to provide:

1. **Almost no regex passes.** The naive implementation runs one `finditer` per
   rule; the slightly-less-naive one compiles every rule into a single giant
   alternation. Both are slow in Python, because `re` has no multi-pattern
   automaton and re-tries every branch at every character position -- roughly
   1.5 us per character for a 40-branch alternation, i.e. ~9 ms for a 4 KB
   document. Instead, `prefilter` derives a mandatory literal for each rule, one
   trie pass reports which literals are present, and only rules that can
   possibly match are run at all. On clean text that is one pass, not fifty.

   The dispatch is *inverted*: rather than walking all 146 variants and asking
   each whether its trigger fired, the compiler builds a trigger -> variant index
   and the scan walks only the literals the trie actually found. On a clean short
   query that is the difference between 146 dict probes and zero, because a clean
   query produces an empty literal set. Measured: 175 us -> 66 us for a 75-char
   query. Candidate indices are visited in ascending variant order, which is
   severity order, so `stop_at` short-circuits and finding order are unchanged.

2. **Windowed rescans.** A rule that survives the prefilter is executed only in
   the neighbourhood of the literal that triggered it, not across the whole
   document, so a single common trigger word cannot reintroduce a full scan.

3. **No shadowing.** Each rule (in fact each top-level alternation branch) is
   matched independently, so a broad rule that later fails validation can never
   consume a region a more specific rule should own -- the case that made a
   12-digit Aadhaar candidate swallow a 16-digit payment card. Rules are visited
   in severity order, which also makes `stop_at` short-circuit on the first
   critical hit.

4. **Reuse.** The content-hash cache lets a chunk scanned during retrieval be
   free when the same bytes reappear in tool output later in the run.
   Compilation is memoized per rule set, so building a pipeline per request --
   the natural pattern for a FastAPI dependency -- costs microseconds.

The prefilter is an optimization, never a policy: `scan(prefilter=False)` runs
every rule unconditionally, and the test suite asserts both modes return
identical findings over the full attack corpus.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from threading import Lock
from collections.abc import Callable, Sequence

from .patterns import Rule
from .prefilter import LiteralIndex, required_literals, split_alternation, windows
from .types import Severity

_FLAGS = re.DOTALL  # text arrives casefolded from the normalizer
_WHOLE_TEXT = None  # sentinel: scan the entire string


@dataclass(frozen=True, slots=True)
class RawMatch:
    rule: Rule
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class _Variant:
    """One independently scannable branch of a rule, with its scan condition."""

    rule: Rule
    pattern: re.Pattern[str]
    triggers: tuple[str, ...]   # any-of literals; empty means no literal condition
    digit_gated: bool           # no literal, but the branch cannot match without a digit


# Compiled programs are immutable and thread-safe, so they are shared by every
# pipeline instance using the same rule set.
_COMPILE_CACHE: dict[tuple[tuple[str, str], ...], CompiledRuleSet] = {}
_COMPILE_LOCK = Lock()


def compile_rules(rules: Sequence[Rule]) -> CompiledRuleSet:
    """Return a shared `CompiledRuleSet`, compiling only on first request."""
    key = tuple((r.id, r.pattern) for r in rules)
    hit = _COMPILE_CACHE.get(key)
    if hit is not None:
        return hit
    with _COMPILE_LOCK:
        hit = _COMPILE_CACHE.get(key)
        if hit is None:
            hit = CompiledRuleSet(rules)
            _COMPILE_CACHE[key] = hit
    return hit


class CompiledRuleSet:
    """Immutable, thread-safe, compiled once per distinct rule set."""

    __slots__ = ("rules", "_variants", "_index", "_by_trigger", "_digit_gated", "_always")

    def __init__(self, rules: Sequence[Rule]) -> None:
        # Precedence: highest severity first, then most specific pattern. Rules
        # are visited in this order so `stop_at` short-circuits as early as
        # possible.
        self.rules = tuple(
            sorted(rules, key=lambda r: (-int(r.severity), -len(r.pattern), r.id))
        )

        variants: list[_Variant] = []
        literals: set[str] = set()
        needs_digit = False
        for rule in self.rules:
            for branch in split_alternation(rule.pattern):
                try:
                    compiled = re.compile(branch, _FLAGS)
                except re.error:
                    # A branch that is not independently valid falls back to the
                    # whole pattern, scanned unconditionally. Correctness first.
                    variants.append(_Variant(rule, re.compile(rule.pattern, _FLAGS), (), False))
                    continue
                triggers = required_literals(branch)
                if triggers:
                    literals |= triggers
                    variants.append(_Variant(rule, compiled, tuple(sorted(triggers)), False))
                else:
                    gated = "\\d" in branch or "0-9" in branch
                    needs_digit = needs_digit or gated
                    variants.append(_Variant(rule, compiled, (), gated))

        self._variants = tuple(variants)
        self._index = LiteralIndex(literals, needs_digit=needs_digit)

        # Inverted dispatch index, built once per rule set. Values are variant
        # positions rather than variants, so candidate sets sort into the
        # original severity order for free.
        by_trigger: dict[str, list[int]] = {}
        digit_gated: list[int] = []
        always: list[int] = []
        for i, variant in enumerate(variants):
            if variant.triggers:
                for trigger in variant.triggers:
                    by_trigger.setdefault(trigger, []).append(i)
            elif variant.digit_gated:
                digit_gated.append(i)
            else:
                always.append(i)
        self._by_trigger = {k: tuple(v) for k, v in by_trigger.items()}
        self._digit_gated = tuple(digit_gated)
        self._always = tuple(always)

    def __len__(self) -> int:
        return len(self.rules)

    @property
    def pattern_count(self) -> int:
        """Compiled programs held: one per rule variant, plus the trie."""
        return len(self._variants) + 1

    @property
    def unconditional_count(self) -> int:
        """Variants with no scan condition. Kept small on purpose."""
        return sum(1 for v in self._variants if not v.triggers and not v.digit_gated)

    def _candidates(self, hits) -> Sequence[int]:
        """Variant positions that can possibly match, in severity order.

        `hits.literals` is exactly the set of trigger literals present in the
        text (saturated ones included), so a clean document yields an empty set
        and the scan does no per-variant work at all. Unconditional variants and
        -- when a digit run exists -- digit-gated variants are always included,
        because the prefilter cannot exclude them.
        """
        if not hits.literals and not self._always and not (
            self._digit_gated and hits.has_digit
        ):
            return ()
        selected: set[int] = set(self._always)
        by_trigger = self._by_trigger
        for literal in hits.literals:
            hit = by_trigger.get(literal)
            if hit is not None:
                selected.update(hit)
        if self._digit_gated and hits.has_digit:
            selected.update(self._digit_gated)
        return sorted(selected)

    def _spans(self, text: str, variant: _Variant, hits) -> tuple[tuple[int, int], ...] | None:
        """Where this variant must be scanned; None means skip it entirely."""
        if hits is None:
            return ((0, len(text)),)
        if variant.triggers:
            offsets: list[int] = []
            for trigger in variant.triggers:
                if trigger in hits.saturated:
                    # Too frequent to window around; scanning once is cheaper.
                    return ((0, len(text)),)
                offsets.extend(hits.positions.get(trigger, ()))
            if not offsets:
                return None
            offsets.sort()
            return tuple(windows(text, tuple(offsets)))
        if variant.digit_gated:
            if not hits.has_digit:
                return None
            if hits.digit_saturated:
                return ((0, len(text)),)
            return tuple(windows(text, hits.digit_positions))
        return ((0, len(text)),)

    def scan(
        self,
        text: str,
        *,
        limit: int = 64,
        stop_at: Severity | None = None,
        keep: Callable[[Rule, str], bool] | None = None,
        prefilter: bool = True,
    ) -> list[RawMatch]:
        """Return matches, honouring a finding cap and optional short-circuit.

        `keep` is the precision hook: validators (Luhn, Verhoeff, entropy) run
        here so rejected candidates never become findings or audit rows.
        `prefilter=False` disables the literal prefilter and is used by the test
        suite to prove the two modes agree.
        """
        if not text:
            return []

        hits = self._index.probe(text) if prefilter else None
        out: list[RawMatch] = []
        seen: set[tuple[str, int, int]] = set()

        variants = self._variants
        # prefilter=False must still visit every variant: the test suite proves
        # the two modes agree, and that proof is worthless if both take the same
        # shortcut.
        candidates = range(len(variants)) if hits is None else self._candidates(hits)

        for position in candidates:
            variant = variants[position]
            spans = self._spans(text, variant, hits)
            if spans is None:
                continue
            pattern = variant.pattern
            rule = variant.rule
            for lo, hi in spans:
                pos = lo
                while pos <= hi:
                    m = pattern.search(text, pos, hi)
                    if m is None:
                        break
                    start, end = m.start(), m.end()
                    matched = m.group(0)
                    if keep is not None and not keep(rule, matched):
                        # Advance past the whole candidate rather than sliding one
                        # character in: sliding inside a numeric run manufactures
                        # false positives, because some 12-to-15-digit window of an
                        # arbitrary long number will satisfy Luhn by chance.
                        pos = end if end > start else start + 1
                        continue
                    key = (rule.id, start, end)
                    if key not in seen:
                        seen.add(key)
                        out.append(RawMatch(rule, start, end, matched))
                        if len(out) >= limit:
                            return out
                        if stop_at is not None and rule.severity >= stop_at:
                            return out
                    pos = end if end > start else start + 1

        return out


def content_key(text: str) -> str:
    """Stable 16-byte key. blake2b is materially faster than sha256 here."""
    return hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=16).hexdigest()


class ResultCache:
    """Insertion-ordered LRU.

    Hand-rolled rather than functools.lru_cache so entries can be invalidated on
    policy change and so one cache can be shared across every layer.
    """

    __slots__ = ("_data", "_cap", "hits", "misses")

    def __init__(self, capacity: int = 4096) -> None:
        self._data: dict[str, object] = {}
        self._cap = max(capacity, 0)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> object | None:
        if self._cap == 0:
            return None
        value = self._data.get(key)
        if value is None:
            self.misses += 1
            return None
        self._data[key] = self._data.pop(key)  # refresh recency
        self.hits += 1
        return value

    def put(self, key: str, value: object) -> None:
        if self._cap == 0:
            return
        if key in self._data:
            self._data.pop(key)
        elif len(self._data) >= self._cap:
            self._data.pop(next(iter(self._data)))
        self._data[key] = value

    def clear(self) -> None:
        self._data.clear()
        self.hits = self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
