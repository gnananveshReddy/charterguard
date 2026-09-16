"""Zero-dependency stand-ins for the two model seams.

Neither of these is a model, and neither should be described to a risk owner as
one. They exist for two honest reasons: a deployment that cannot take any
dependency still gets *something* on these seams, and every downstream
integration (thresholds, findings, redaction, audit) can be exercised end to end
in CI with no weights present.

Measured ceilings, not aspirations:

  * `KeywordIntentClassifier` compares token overlap against charter phrases.
    It cannot recognise paraphrase ("help me sleep" vs "insomnia advice") and
    will return `unclear` for most real traffic. That is the safe direction --
    `ScopeGate` only widens scope on a non-`out_of_scope` label at confidence
    >= 0.6, and this classifier reaches that bar only on strong lexical overlap.
  * `GazetteerEntityRecognizer` finds names by *title cues* ("Dr Priya Sharma",
    "Mr. Chen") and an optional caller-supplied name list. It does not find
    bare names in running prose, and its capitalisation heuristic is off by
    default because on real text it fires on sentence starts, product names and
    place names -- noisy redaction trains users to ignore redaction.

If GDPR or HIPAA scope applies, treat these as placeholders and install
`agent_guardrails[spacy]` or `agent_guardrails[presidio]`.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from ..pii import Entity

__all__ = ["KeywordIntentClassifier", "GazetteerEntityRecognizer"]

_WORD = re.compile(r"[a-z0-9']+")
# Words too common to carry intent signal. Deliberately short: a long stoplist
# tuned on one charter stops generalising to the next.
_STOP = frozenset(
    "a an the and or but if then for to of in on at by with from as is are was were be been "
    "do does did can could would should will shall may might must i you he she it we they me "
    "my your our their this that these those there here what which who whom how why when where "
    "please help need want about into over under again more most some any no not".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 2}


class KeywordIntentClassifier:
    """Token-overlap intent classifier over the charter's own phrases.

    Scoring is asymmetric on purpose. Out-of-scope phrases are matched with a
    lower bar than in-scope phrases, because the cost of the two errors is not
    symmetric: wrongly widening scope is a security failure, wrongly narrowing
    it is a support ticket.
    """

    __slots__ = ("_in_bar", "_out_bar")

    def __init__(self, *, in_scope_bar: float = 0.34, out_of_scope_bar: float = 0.2) -> None:
        self._in_bar = in_scope_bar
        self._out_bar = out_of_scope_bar

    @staticmethod
    def _best_overlap(text_tokens: set[str], phrases: Sequence[str]) -> float:
        best = 0.0
        for phrase in phrases:
            pt = _tokens(phrase)
            if not pt:
                continue
            # Fraction of the phrase's meaningful tokens present in the text.
            # Normalising by the phrase (not the text) keeps a long request from
            # diluting a short, specific charter phrase.
            best = max(best, len(pt & text_tokens) / len(pt))
        return best

    async def classify(
        self, text: str, in_scope: Sequence[str], out_of_scope: Sequence[str]
    ) -> tuple[str, float]:
        tt = _tokens(text)
        if not tt:
            return "unclear", 0.0
        out_hit = self._best_overlap(tt, out_of_scope)
        in_hit = self._best_overlap(tt, in_scope)
        if out_hit >= self._out_bar and out_hit >= in_hit:
            return "out_of_scope", min(1.0, 0.5 + out_hit / 2)
        if in_hit >= self._in_bar:
            return "in_scope", min(1.0, 0.4 + in_hit / 2)
        # Below both bars the honest answer is "no idea". Returning a confident
        # in_scope here is what would turn a lexical gate into a bypass.
        return "unclear", round(max(in_hit, out_hit), 3)


_TITLE = re.compile(
    r"\b(?:Dr|Doctor|Mr|Mrs|Ms|Miss|Prof|Professor|Sir|Madam|Nurse|Patient|Rev|Capt|Sgt|Lt)\b\.?\s+"
    r"((?:[A-Z][\w'’-]+)(?:\s+[A-Z][\w'’-]+){0,2})"
)
_CAP_RUN = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b")


class GazetteerEntityRecognizer:
    """Title-cued and list-based name recognition. No model, no weights.

    Args:
        names: exact names or surnames to match, case-insensitively. This is
            the high-precision path -- an employee directory, a patient roster,
            a customer list. Prefer it over the heuristics below.
        capitalised_pairs: enable the "two capitalised words in a row is a
            person" heuristic. Off by default; it fires on "New York",
            "Blue Cross" and "Monday Morning". Turn it on only where recall
            matters more than user trust in redaction.
        label: entity label emitted, matched against the caller's
            `EntityDetector` label→category map.
    """

    __slots__ = ("_names", "_caps", "_label", "_min_len")

    def __init__(
        self,
        names: Iterable[str] = (),
        *,
        capitalised_pairs: bool = False,
        label: str = "PERSON",
        min_name_len: int = 3,
    ) -> None:
        cleaned = {n.strip().lower() for n in names if n and len(n.strip()) >= min_name_len}
        self._names = cleaned
        self._caps = capitalised_pairs
        self._label = label
        self._min_len = min_name_len

    def recognize(self, text: str) -> Sequence[Entity]:
        spans: dict[tuple[int, int], Entity] = {}

        # 1. Title cue: highest precision available without a model.
        for m in _TITLE.finditer(text):
            spans[(m.start(1), m.end(1))] = Entity(self._label, m.start(1), m.end(1), 0.85)

        # 2. Caller-supplied gazetteer, matched on word boundaries so "Li" does
        #    not fire inside "Lisbon".
        if self._names:
            for m in re.finditer(r"[\w'’-]+", text):
                if m.group(0).lower() in self._names:
                    spans.setdefault((m.start(), m.end()), Entity(self._label, m.start(), m.end(), 0.9))

        # 3. Opt-in capitalisation heuristic, scored low so a MASK-on-high
        #    policy does not act on it alone.
        if self._caps:
            for m in _CAP_RUN.finditer(text):
                spans.setdefault((m.start(), m.end()), Entity(self._label, m.start(), m.end(), 0.45))

        return sorted(spans.values(), key=lambda e: (e.start, e.end))
