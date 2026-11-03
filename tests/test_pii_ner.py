"""The NER seam.

The bundled rules cannot find PII that has no format. This asserts the adapter
that lets a real recogniser be plugged in, and that its findings are treated
identically to rule findings by redaction and policy.
"""
from __future__ import annotations

from charterguard import GuardrailPipeline
from charterguard.detectors import (
    EncodedPayloadDetector,
    RuleDetector,
    StructuralDetector,
)
from charterguard.pii import Entity, EntityDetector
from charterguard.types import Category, Severity

TEXT = "Priya Sharma was discharged on Tuesday after treatment."


class StubRecognizer:
    """Stands in for spaCy/Presidio: finds a name and a health context."""

    def recognize(self, text: str):
        out = []
        idx = text.find("priya sharma")
        if idx >= 0:
            out.append(Entity("PERSON", idx, idx + len("priya sharma"), 0.95))
        idx = text.find("discharged")
        if idx >= 0:
            out.append(Entity("MEDICAL", idx, idx + len("discharged"), 0.9))
        return out


def _pipeline(**kw) -> GuardrailPipeline:
    return GuardrailPipeline(
        detectors=(
            RuleDetector(),
            StructuralDetector(),
            EncodedPayloadDetector(),
            EntityDetector(StubRecognizer(), **kw),
        )
    )


def test_format_free_pii_is_invisible_without_ner():
    """Documents the actual limitation rather than implying coverage."""
    baseline = GuardrailPipeline().check_output(TEXT)
    assert not [f for f in baseline.findings if f.category is Category.PII]


def test_ner_findings_flow_through_the_same_policy():
    verdict = _pipeline().check_output(TEXT)
    categories = {f.category for f in verdict.findings}
    assert Category.PII in categories
    assert {f.rule_id for f in verdict.findings} >= {"pii.ner.person", "pii.ner.medical"}


def test_ner_spans_are_redacted_from_output():
    verdict = _pipeline().check_output(TEXT)
    assert "Priya Sharma" not in (verdict.sanitized or "")


def test_entity_values_are_never_echoed_into_findings():
    """Excerpts land in audit rows; a name there is a second uncontrolled copy."""
    verdict = _pipeline().check_output(TEXT)
    for finding in verdict.findings:
        if finding.rule_id.startswith("pii.ner"):
            assert "priya" not in finding.excerpt.lower()


def test_low_confidence_entities_are_dropped():
    verdict = _pipeline(min_score=0.99).check_output(TEXT)
    assert not [f for f in verdict.findings if f.rule_id.startswith("pii.ner")]


def test_label_allowlist_is_honoured():
    verdict = _pipeline(labels=("MEDICAL",)).check_output(TEXT)
    ids = {f.rule_id for f in verdict.findings if f.rule_id.startswith("pii.ner")}
    assert ids == {"pii.ner.medical"}


def test_medical_labels_outrank_names_in_severity():
    verdict = _pipeline().check_output(TEXT)
    by_id = {f.rule_id: f for f in verdict.findings}
    assert by_id["pii.ner.medical"].severity is Severity.HIGH
    assert by_id["pii.ner.person"].severity is Severity.MEDIUM
