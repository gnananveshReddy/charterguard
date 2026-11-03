"""Cheap precision filters.

Pattern matching alone produces unusable false-positive rates on numeric PII
and on high-entropy-looking prose. These validators are what make the SECRET
and PII categories safe to run in BLOCK mode.
"""
from __future__ import annotations

import math
from collections import Counter

_B64_ISH = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=-_")

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def luhn_ok(value: str) -> bool:
    """Payment cards. Rejects the vast majority of random 13-19 digit runs."""
    d = digits(value)
    if not 12 <= len(d) <= 19:
        return False
    total = 0
    parity = len(d) % 2
    for i, ch in enumerate(d):
        n = ord(ch) - 48
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def verhoeff_ok(value: str) -> bool:
    """Aadhaar (India) checksum. Turns a 12-digit regex into a usable detector."""
    d = digits(value)
    if len(d) != 12 or d[0] in "01":
        return False
    c = 0
    for i, ch in enumerate(reversed(d)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][ord(ch) - 48]]
    return c == 0


def shannon_entropy(value: str) -> float:
    """Bits per character. Random secrets sit high; English prose sits low."""
    if not value:
        return 0.0
    counts = Counter(value)
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def looks_random(value: str, *, min_len: int = 20, min_entropy: float = 3.6) -> bool:
    """Entropy gate for generic credential patterns.

    Requires length, entropy, and mixed character classes together -- any one
    of the three alone misfires on hashes, URLs, and long identifiers.
    """
    if len(value) < min_len:
        return False
    if shannon_entropy(value) < min_entropy:
        return False
    has_lower = any(c.islower() for c in value)
    has_upper = any(c.isupper() for c in value)
    has_digit = any(c.isdigit() for c in value)
    classes = has_lower + has_upper + has_digit
    return classes >= 2


def is_encoded_run(value: str) -> bool:
    """Heuristic gate before attempting a decode: avoids wasted base64 work."""
    if len(value) < 16:
        return False
    sample = value[:256]
    return sum(ch in _B64_ISH for ch in sample) / len(sample) > 0.92


_GSTIN_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def gstin_ok(value: str) -> bool:
    """India GSTIN checksum (mod-36, alternating weights 1 and 2).

    The 15th character is a check digit over the first 14, so a random
    15-character token that happens to fit the shape passes 1 time in 36.
    """
    s = value.strip().casefold()
    if len(s) != 15 or any(ch not in _GSTIN_ALPHABET for ch in s):
        return False
    total = 0
    for i, ch in enumerate(s[:14]):
        product = _GSTIN_ALPHABET.index(ch) * (2 if i % 2 else 1)
        total += product // 36 + product % 36
    return _GSTIN_ALPHABET[(36 - total % 36) % 36] == s[14]


def india_mobile_ok(value: str) -> bool:
    """Ten-digit Indian mobile, after an optional +91 / 0091 / 0 prefix."""
    d = digits(value)
    for prefix in ("0091", "91", "0"):
        if len(d) > 10 and d.startswith(prefix):
            d = d[len(prefix) :]
            break
    return len(d) == 10 and d[0] in "6789" and len(set(d)) > 2
