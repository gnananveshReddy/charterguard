from charterguard import GuardrailConfig, GuardrailPipeline
from charterguard.grounding import GroundingChecker

CFG = GuardrailConfig.strict()


def test_fabricated_citation_is_critical():
    v = GroundingChecker().check(
        "Revenue grew 12% in Q3 [[doc_99]].", ["doc_1", "doc_2"], CFG
    )
    assert v.blocked
    assert v.findings[0].rule_id == "ground.fabricated_citation"


def test_cited_claims_pass():
    v = GroundingChecker().check(
        "Revenue grew 12% in Q3 [[doc_1]]. Margin held at 40% [[doc_2]].",
        ["doc_1", "doc_2"],
        CFG,
    )
    assert not v.blocked
    assert v.meta["coverage"] == 1.0


def test_uncited_numeric_claims_are_flagged():
    v = GroundingChecker().check(
        "Revenue grew 12% in Q3 and headcount rose to 450 across three regions.",
        ["doc_1"],
        CFG,
    )
    assert v.findings
    assert v.meta["claims"] >= 1


def test_conversational_text_is_not_treated_as_a_claim():
    v = GroundingChecker().check(
        "Here's a summary. Let me know if you would like more detail.", ["doc_1"], CFG
    )
    assert v.meta["claims"] == 0
    assert not v.findings


def test_pipeline_wires_grounding_only_when_citations_supplied():
    p = GuardrailPipeline()
    without = p.check_output("Revenue grew 12% in Q3 and headcount rose to 450.")
    with_ids = p.check_output(
        "Revenue grew 12% in Q3 and headcount rose to 450.", allowed_citations=["doc_1"]
    )
    assert not without.findings
    assert with_ids.findings
