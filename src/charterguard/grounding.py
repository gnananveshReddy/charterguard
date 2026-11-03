"""Grounding and citation validation.

Two distinct failures, one checker:
  * fabricated citation -- the answer cites an id that was never retrieved.
    This is treated as CRITICAL: it is the failure that destroys user trust and
    it is trivially detectable.
  * uncited claim -- a claim-bearing sentence with no citation in scope.

The claim heuristic is deliberately conservative (numbers, proper-noun density,
assertive verbs). It is a coverage signal, not a truth oracle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Iterable, Sequence

from .config import GuardrailConfig
from .policy import decide
from .types import Category, Finding, Layer, Severity, Timer, Verdict

_CITATION = re.compile(r"\[{1,2}([a-z0-9_.:#\-]{1,64})\]{1,2}", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n+")
_ASSERTIVE = re.compile(
    r"\b(?:is|are|was|were|has|have|will|must|shall|requires?|shows?|reports?|"
    r"increased|decreased|costs?|totals?|equals?)\b",
    re.IGNORECASE,
)
_NUMERIC = re.compile(r"\d")
_PROPER = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_HEDGE = re.compile(
    r"\b(?:i can|i will|let me|would you like|here(?:'s| is) (?:a|the)|"
    r"summary|next steps?|note that|for example)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class GroundingReport:
    claims: int
    cited_claims: int
    unknown_citations: tuple[str, ...]
    known_citations: tuple[str, ...]

    @property
    def coverage(self) -> float:
        return 1.0 if self.claims == 0 else self.cited_claims / self.claims


class GroundingChecker:
    """Stateless; construct once and pass the allowed id set per call."""

    __slots__ = ("require_citations", "min_coverage")

    def __init__(self, *, require_citations: bool = True, min_coverage: float = 0.8) -> None:
        self.require_citations = require_citations
        self.min_coverage = min_coverage

    @staticmethod
    def _is_claim(sentence: str) -> bool:
        if len(sentence) < 25 or _HEDGE.search(sentence):
            return False
        if _NUMERIC.search(sentence):
            return True
        return bool(len(_PROPER.findall(sentence)) >= 2 and _ASSERTIVE.search(sentence))

    def analyze(self, text: str, allowed_ids: Iterable[str]) -> GroundingReport:
        allowed = {str(a).lower() for a in allowed_ids}
        found_known: list[str] = []
        unknown: list[str] = []
        for m in _CITATION.finditer(text):
            cid = m.group(1).lower()
            (found_known if cid in allowed else unknown).append(cid)

        claims = 0
        cited = 0
        sentences = [s for s in _SENTENCE.split(text) if s and s.strip()]
        for i, sentence in enumerate(sentences):
            if not self._is_claim(sentence):
                continue
            claims += 1
            window = sentence + (sentences[i + 1] if i + 1 < len(sentences) else "")
            if any(m.group(1).lower() in allowed for m in _CITATION.finditer(window)):
                cited += 1

        return GroundingReport(claims, cited, tuple(dict.fromkeys(unknown)), tuple(found_known))

    def check(
        self, text: str, allowed_ids: Sequence[str], config: GuardrailConfig
    ) -> Verdict:
        timer = Timer()
        report = self.analyze(text, allowed_ids)
        findings: list[Finding] = []

        if report.unknown_citations:
            findings.append(
                Finding(
                    rule_id="ground.fabricated_citation",
                    category=Category.UNGROUNDED,
                    severity=Severity.CRITICAL,
                    score=0.95,
                    detector="grounding",
                    description=(
                        "answer cites sources that were never retrieved: "
                        + ", ".join(report.unknown_citations[:5])
                    ),
                )
            )

        if self.require_citations and report.claims and report.coverage < self.min_coverage:
            missing = report.claims - report.cited_claims
            findings.append(
                Finding(
                    rule_id="ground.uncited_claims",
                    category=Category.UNGROUNDED,
                    severity=Severity.HIGH if report.coverage < 0.5 else Severity.MEDIUM,
                    score=min(0.55 + (1 - report.coverage) * 0.4, 0.95),
                    detector="grounding",
                    description=(
                        f"{missing} of {report.claims} claim-bearing sentences carry no "
                        f"citation (coverage {report.coverage:.0%})"
                    ),
                )
            )

        found = tuple(findings)
        d = decide(found, config, Layer.OUTPUT)
        return Verdict(
            layer=Layer.OUTPUT,
            action=d.action,
            score=d.score,
            findings=found,
            policy_id=d.policy_id if found else "guardrails.ok",
            latency_us=timer.us(),
            meta={
                "claims": report.claims,
                "cited_claims": report.cited_claims,
                "coverage": round(report.coverage, 3),
                "unknown_citations": list(report.unknown_citations),
            },
        )
