"""
Canonical Egyptian transliteration normalization.

Maps Unicode Egyptological characters to Manuel de Codage (MdC) ASCII
equivalents, mirroring how sumerian_normalize.py maps ORACC to ATF.

Used by anchor extraction and any downstream scripts that need to match
transliteration forms between the corpus and dictionary.
"""
from __future__ import annotations


_EGYPTIAN_TO_MDC = {
    "ꜣ": "A",
    "ꜥ": "a",
    "ḥ": "H",
    "ḫ": "x",
    "ẖ": "X",
    "ṯ": "T",
    "ḏ": "D",
    "š": "S",
    "Š": "S",
    "ṭ": "d",
    "ṣ": "s",
    "ś": "s",
    "ȝ": "A",
    "ɜ": "A",
    "ʿ": "a",
    "ʾ": "A",
    "ỉ": "i",
    "č": "T",
    "ğ": "D",
    "ḳ": "q",
    "ḍ": "D",
}

# MdC uppercase codes that carry phonemic distinction and must not be
# lowercased after substitution (e.g. H ≠ h, T ≠ t in Manuel de Codage).
_MDC_PRESERVE_UPPER = frozenset("AHTDSX")


def _apply_casing(s: str) -> str:
    """Apply final casing rules.

    If the string (after substitution) is all ASCII uppercase with no
    Egyptological chars remaining, it is a raw-caps input with no MdC codes
    present — fully lowercase it.  Otherwise, use selective lowercasing that
    preserves MdC uppercase codes (A, H, T, D, S, X).
    """
    # If every alphabetic character is uppercase ASCII, treat as all-caps
    # non-MdC input and fully lowercase.
    alphabetic = [c for c in s if c.isalpha()]
    if alphabetic and all(c.isupper() and c.isascii() for c in alphabetic):
        return s.lower()
    # Otherwise preserve MdC uppercase codes.
    return "".join(c if c in _MDC_PRESERVE_UPPER else c.lower() for c in s)


def normalize_egyptian_token(raw) -> str:
    """Canonical normalization for a single Egyptian transliteration token.

    Applies (in order):
      1. Strip whitespace
      2. Egyptian Unicode characters -> MdC ASCII equivalents
      3. Casing: fully lowercase all-caps ASCII tokens (no MdC codes present);
         otherwise preserve MdC uppercase codes (A, H, T, D, S, X)

    Safe on None and empty input (returns "").
    Idempotent: normalize(normalize(x)) == normalize(x).
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    for old, new in _EGYPTIAN_TO_MDC.items():
        s = s.replace(old, new)
    return _apply_casing(s)
