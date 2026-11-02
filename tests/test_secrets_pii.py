from charterguard import Category, GuardrailConfig, GuardrailPipeline
from charterguard.validators import luhn_ok, looks_random, verhoeff_ok

STRICT = GuardrailConfig.strict()


def test_aws_key_is_detected_and_redacted_on_output():
    p = GuardrailPipeline()
    v = p.check_output("Use AKIAIOSFODNN7EXAMPLE for access")
    assert Category.SECRET in v.categories
    assert v.blocked                       # SECRET defaults to BLOCK
    assert v.sanitized is None             # nothing leaves on a block


def test_pii_is_redacted_but_not_blocked_by_default():
    p = GuardrailPipeline()
    v = p.check_output("Contact me at gnananvesh@example.com for details")
    assert Category.PII in v.categories
    assert not v.blocked
    assert "gnananvesh@example.com" not in v.sanitized
    assert "[REDACTED:EMAIL" in v.sanitized


def test_redaction_placeholder_carries_a_digest():
    p = GuardrailPipeline()
    v = p.check_output("mail: a.b@corp.example.com")
    assert v.meta["redaction_count"] == 1
    assert v.meta["redactions"][0]["value_digest"]


def test_luhn_rejects_random_digit_runs():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("1234567890123456")


def test_card_detection_uses_luhn():
    p = GuardrailPipeline()
    hit = p.check_output("card 4111 1111 1111 1111")
    miss = p.check_output("order 1234 5678 9012 3456")
    assert any(f.rule_id == "pii.card" for f in hit.findings)
    assert not any(f.rule_id == "pii.card" for f in miss.findings)


def test_aadhaar_detection_uses_verhoeff():
    assert verhoeff_ok("234567890124")
    # Every single-digit change breaks a Verhoeff checksum.
    assert not any(verhoeff_ok("23456789012" + d) for d in "0123456789" if d != "4")
    assert not verhoeff_ok("134567890124")  # Aadhaar never starts with 0 or 1


def test_generic_credential_requires_entropy():
    p = GuardrailPipeline()
    low = p.check_output("password = password")
    high = p.check_output("api_key = 8fH2kLm9QpZx4Rt7Wy1Bv6Nc3Ds5Ge0A")
    assert not any(f.rule_id == "sec.generic" for f in low.findings)
    assert any(f.rule_id == "sec.generic" for f in high.findings)


def test_entropy_gate_rejects_prose():
    assert not looks_random("this is a perfectly ordinary sentence")
    assert looks_random("8fH2kLm9QpZx4Rt7Wy1Bv6Nc3Ds5Ge0A")


def test_private_key_block_is_critical():
    p = GuardrailPipeline()
    v = p.check_output("-----BEGIN RSA PRIVATE KEY-----\nMIIEow...")
    assert v.blocked


def test_tool_output_secrets_are_redacted_before_entering_context():
    p = GuardrailPipeline()
    r = p.inspect_tool_output(
        "connection ok, token=ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        tool_name="db_health",
    )
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in r.envelope
    assert r.verdict.meta["redactions"]
