from agent_guardrails import Charter, GuardrailConfig, GuardrailPipeline

CHARTER = Charter(
    agent_id="policy-qa",
    version="1.2",
    purpose="answering questions about internal IT policy and standards",
    in_scope=("policy", "standard", "procedure", "control", "guideline"),
    out_of_scope=("salary", "performance review", "termination", "medical record"),
    allowed_tools=("search_policy", "read_document"),
)


def pipe():
    return GuardrailPipeline(GuardrailConfig.strict(), charter=CHARTER)


def test_in_charter_request_passes():
    v = pipe().check_scope("What does the access control standard require for MFA?")
    assert not v.blocked
    assert v.meta["in_scope_hits"]


def test_explicit_exclusion_is_blocked():
    v = pipe().check_scope("What is the salary band for a senior developer?")
    assert v.blocked
    assert v.policy_id == "guardrails.block.out_of_scope"


def test_no_charter_signal_is_flagged_as_ambiguous():
    v = pipe().check_scope("Can you write me a poem about the sea?")
    assert v.findings
    assert v.meta["ambiguous"] is True


def test_charter_fingerprint_is_stable_and_in_meta():
    v = pipe().check_scope("policy question")
    assert v.meta["charter_fingerprint"] == CHARTER.fingerprint()
    assert len(CHARTER.fingerprint()) == 32


def test_tool_outside_charter_is_blocked():
    v = pipe().check_tool_call("send_email", {"to": "x@y.z"})
    assert v.blocked
    assert v.findings[0].rule_id == "scope.tool_not_in_charter"


def test_tool_inside_charter_is_allowed():
    v = pipe().check_tool_call("search_policy", {"query": "mfa"})
    assert not v.blocked


def test_tool_arguments_are_scanned_for_commands():
    v = pipe().check_tool_call("search_policy", {"query": "drop table users;"})
    assert v.findings


def test_refusal_uses_charter_purpose():
    p = pipe()
    v = p.check_scope("What is the salary band for a senior developer?")
    assert "internal IT policy" in p.refusal(v)


async def test_classifier_rescues_ambiguous_request():
    class Classifier:
        async def classify(self, text, in_scope, out_of_scope):
            return "policy_lookup", 0.9

    p = GuardrailPipeline(
        GuardrailConfig.strict(), charter=CHARTER, intent_classifier=Classifier()
    )
    v = await p.acheck_scope("How do I request an exception for this control gap?")
    assert not v.blocked
