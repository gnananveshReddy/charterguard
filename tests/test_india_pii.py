"""India PII: PAN, GSTIN, UPI IDs, mobile numbers, passport numbers."""
from __future__ import annotations

from charterguard import GuardrailPipeline
from charterguard.validators import gstin_ok, india_mobile_ok


def rule_ids(text: str) -> set[str]:
    return {f.rule_id for f in GuardrailPipeline().check_output(text).findings}


# ------------------------------------------------------------------ GSTIN
def test_gstin_checksum():
    assert gstin_ok("29AAGCB7383J1Z4")
    assert gstin_ok("27AAPFU0939F1ZV")
    assert not gstin_ok("27AAPFU0939F1ZA")   # wrong check character
    assert not gstin_ok("27AAPFU0939F1Z")    # too short
    assert not gstin_ok("27AAPFU0939F1Z!")


def test_gstin_is_detected_only_with_a_valid_check_digit():
    assert "pii.gstin.in" in rule_ids("Our GSTIN is 29AAGCB7383J1Z4.")
    assert "pii.gstin.in" not in rule_ids("Our GSTIN is 29AAGCB7383J1Z5.")


# -------------------------------------------------------------------- PAN
def test_pan_requires_a_valid_holder_type():
    assert "pii.pan.in" in rule_ids("PAN: ABCPE1234F")      # P = individual
    assert "pii.pan.in" in rule_ids("company PAN AAACB1234C")  # C = company
    assert "pii.pan.in" not in rule_ids("SKU ABCDE1234F")   # D is not a holder type


def test_pan_shape_is_still_redacted_without_a_known_holder_type():
    """Redaction must not depend on the holder-type list being complete."""
    v = GuardrailPipeline().check_output("ref ABCDE1234F")
    assert "pii.pan.in.shape" in {f.rule_id for f in v.findings}
    assert v.sanitized is not None
    assert "ABCDE1234F" not in v.sanitized
    assert "[REDACTED:PAN" in v.sanitized


# -------------------------------------------------------------------- UPI
def test_upi_ids_are_detected():
    ids = ("ravi.k@okaxis", "9876543210@ybl", "meera@paytm", "shop-12@okhdfcbank")
    for upi in ids:
        assert "pii.upi.in" in rule_ids(f"pay to {upi} today"), upi
    assert "pii.upi.in" in rule_ids("send it to ravi@ybl.")  # sentence-final dot


def test_emails_on_bank_domains_are_not_upi():
    found = rule_ids("write to ravi@sbi.co.in")
    assert "pii.upi.in" not in found
    assert "pii.email" in found


# ----------------------------------------------------------------- mobile
def test_indian_mobile_formats():
    for text in ("call 98765 43210", "call +91 98765 43210", "call 0091 98765-43210", "call 919876543210"):
        found = rule_ids(text)
        assert found & {"pii.phone.in", "pii.phone.intl"}, text


def test_india_mobile_validator():
    assert india_mobile_ok("98765 43210")
    assert india_mobile_ok("+91 98765 43210")
    assert india_mobile_ok("09876543210")
    assert not india_mobile_ok("12345 67890")   # mobiles start with 6-9
    assert not india_mobile_ok("99999 99999")   # padding, not a number
    assert not india_mobile_ok("98765 4321")    # nine digits


def test_five_five_grouping_is_redacted():
    v = GuardrailPipeline().check_output("Reach me on 98765 43210 after six.")
    assert v.sanitized is not None
    assert "98765 43210" not in v.sanitized
    assert "[REDACTED:PHONE" in v.sanitized


# --------------------------------------------------------------- passport
def test_passport_needs_the_word():
    assert "pii.passport.in" in rule_ids("passport no: K1234567")
    assert "pii.passport.in" in rule_ids("Passport number K1234567")
    assert "pii.passport.in" not in rule_ids("flight K1234567 departs at 9")


# -------------------------------------------------------------- redaction
def test_india_pii_redaction_labels():
    v = GuardrailPipeline().check_output(
        "GSTIN 29AAGCB7383J1Z4, UPI ravi@okaxis, passport no K1234567, PAN ABCPE1234F"
    )
    assert v.sanitized is not None
    for label in ("GSTIN", "UPI", "PASSPORT", "PAN"):
        assert f"[REDACTED:{label}" in v.sanitized, label
    for raw in ("29AAGCB7383J1Z4", "ravi@okaxis", "K1234567", "ABCPE1234F"):
        assert raw not in v.sanitized, raw
