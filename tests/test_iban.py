"""IBAN precision: the bare shape matches hex digests; the validator must not."""
from __future__ import annotations

import random

from charterguard import GuardrailPipeline
from charterguard.validators import iban_ok

VALID = ["DE89370400440532013000", "GB82WEST12345698765432", "DE44500105175407324931", "NL91ABNA0417164300"]


def test_valid_ibans_pass():
    for iban in VALID:
        assert iban_ok(iban), iban
        spaced = " ".join(iban[i : i + 4] for i in range(0, len(iban), 4))
        assert iban_ok(spaced), spaced


def test_wrong_checksum_length_or_country_fail():
    assert not iban_ok("GB82WEST12345698765431")  # checksum
    assert not iban_ok("DE8937040044053201300")  # length
    assert not iban_ok("ZZ89370400440532013000")  # no such country


def test_hex_ids_are_not_ibans():
    rng = random.Random(1234)
    for _ in range(5000):
        token = "".join(rng.choice("0123456789abcdef") for _ in range(16))
        assert not iban_ok(token), token
    p = GuardrailPipeline()
    for text in ("commit ab12f9e8c7d6b5a4 fixed it", "id ed20b8f0e5e8e36d", "sha deadbeef ab34cd56ef7890ab12"):
        assert p.check_output(text).sanitized == text


def test_ibans_are_redacted_compact_and_spaced():
    p = GuardrailPipeline()
    for text in ("IBAN DE89370400440532013000 ok", "IBAN DE89 3704 0044 0532 0130 00 ok"):
        out = p.check_output(text).sanitized or ""
        assert "[REDACTED:IBAN:" in out and "3704" not in out, out



def test_a_following_word_does_not_hide_the_iban():
    # A greedy group count swallowed the next word, the length check failed
    # and the IBAN went out unredacted.
    p = GuardrailPipeline()
    cases = [
        ("IBAN BE68 5390 0754 7034 bank transfer", "5390"),
        ("IBAN BE68539007547034 abcd", "5390"),
        ("NO93 8601 1117 947 x", "8601"),
        ("IBAN DE89 3704 0044 0532 0130 00 ab12 ref", "3704"),
        ("GB82 WEST 1234 5698 7654 32 ref 1234", "5698"),
    ]
    for text, inner in cases:
        out = p.check_output(text).sanitized or ""
        assert "[REDACTED:IBAN:" in out and inner not in out, (text, out)
    assert "bank transfer" in (p.check_output(cases[0][0]).sanitized or "")


def test_a_decoy_prefix_does_not_hide_a_validated_value():
    # A rejected candidate used to be skipped whole, so a real value starting
    # inside it was never examined. For strong validators (IBAN, GSTIN)
    # scanning now resumes at the next token start inside the candidate.
    p = GuardrailPipeline()
    for text, inner, label in [
        ("BE00 BE68 5390 0754 7034", "5390", "IBAN"),
        ("xk00 be68 5390 0754 7034 x", "5390", "IBAN"),
    ]:
        out = p.check_output(text).sanitized or ""
        assert f"[REDACTED:{label}:" in out and inner not in out, (text, out)


def test_grouped_numbers_still_do_not_manufacture_card_matches():
    # Luhn passes ~10% of windows, so a rejected card candidate is still
    # skipped whole: "5678 9012 3456" inside an order number is not a card.
    p = GuardrailPipeline()
    assert not any(f.rule_id == "pii.card" for f in p.check_output("order 1234 5678 9012 3456").findings)


def test_long_digit_runs_still_do_not_manufacture_card_matches():
    # The reason rejects never slide one character at a time: some window of
    # an arbitrary long number satisfies Luhn. A single unbroken token has no
    # token start inside it, so it is still skipped whole.
    rng = random.Random(7)
    p = GuardrailPipeline()
    for _ in range(300):
        run = "".join(rng.choice("0123456789") for _ in range(40))
        verdict = p.check_output(f"order ref {run} shipped")
        assert not any(f.rule_id.startswith("pii.card") for f in verdict.findings), run
