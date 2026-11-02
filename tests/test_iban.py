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
