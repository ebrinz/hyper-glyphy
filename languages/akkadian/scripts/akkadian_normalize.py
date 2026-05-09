"""
Canonical Akkadian token normalization.

Single source of truth for mapping eBL/ORACC citation forms and inflected
surface forms to the common ATF-based token form produced by
`scripts/05_clean_and_tokenize.py`.

Used by `scripts/06_extract_anchors.py` (anchor side) and
`scripts/coverage_diagnostic.py` (audit/diagnostic side). Keeps normalization
in one place to prevent drift between anchors and corpus.

Akkadian-specific additions vs sumerian_normalize: explicit NFC and mimation
alternation.
"""
from __future__ import annotations

import re
import unicodedata

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

_ORACC_TO_ATF = {
    "š": "sz", "Š": "SZ",
    "ḫ": "h",  "Ḫ": "H",
    "ṣ": "s",  "Ṣ": "S",
    "ṭ": "t",  "Ṭ": "T",
    "ʾ": "",
    "ā": "a",  "Ā": "A",
    "ē": "e",  "Ē": "E",
    "ī": "i",  "Ī": "I",
    "ū": "u",  "Ū": "U",
    "â": "a",  "Â": "A",
    "ê": "e",  "Ê": "E",
    "î": "i",  "Î": "I",
    "û": "u",  "Û": "U",
    "á": "a",  "Á": "A",
    "é": "e",  "É": "E",
    "í": "i",  "Í": "I",
    "ó": "o",  "Ó": "O",
    "ú": "u",  "Ú": "U",
}

_BRACE_RE = re.compile(r"\{([^}]*)\}")


def normalize_akkadian_token(raw) -> str:
    """Canonical normalization for a single Akkadian token.

    Order:
      1. NFC unicode normalization.
      2. Subscript digits -> ASCII.
      3. Strip {determinative} braces, keep content.
      4. ORACC unicode letters -> ATF (š -> sz, ā -> a, etc.).
      5. Drop hyphens.
      6. Remove consecutive identical vowels.
      7. Lowercase + strip whitespace.

    Safe on None/empty (returns ""). Idempotent.
    """
    if raw is None:
        return ""
    s = unicodedata.normalize("NFC", str(raw))
    s = s.translate(_SUBSCRIPT_MAP)
    s = _BRACE_RE.sub(r"\1", s)
    for old, new in _ORACC_TO_ATF.items():
        s = s.replace(old, new)
    s = s.replace("-", "")
    # Remove consecutive identical vowels (e.g., "aaa" -> "a", "uu" -> "u")
    s = re.sub(r"([aeiou])\1+", r"\1", s)
    return s.lower().strip()


def mimation_alternates(token: str) -> list[str]:
    """Return [token] plus its non-mimation form when applicable.

    OB nominal/adjectival forms typically end in -um/-am/-im (mimation).
    Later periods drop the final -m. For fallback matching, surface both.
    """
    if not token:
        return []
    out = [token]
    if len(token) >= 3 and token.endswith("m") and token[-2] in "aeiou":
        out.append(token[:-1])
    return out
