"""
Canonical Ancient Greek token normalization.

Single source of truth for mapping Diorisis/Perseus citation forms and inflected
surface forms to a common bare-Greek token form produced by
`scripts/05_clean_and_tokenize.py`.

Normalization choices:
  - Strip polytonic accents (acute, grave, circumflex), breathing marks
    (smooth, rough), iota subscript/adscript, diaeresis. Diorisis lemmas
    are polytonic; corpus tokens are polytonic; modern Gemma encoding is
    sensitive to diacritics but the *meaning* doesn't depend on them, so
    bare-Greek matching gives us better lemma/surface alignment.
  - Normalize final sigma `ς` -> `σ` (positional variant of the same letter).
  - Lowercase.
  - NFC + NFD step to decompose accents that then get filtered out.

Used by `scripts/06_extract_anchors.py` and `scripts/coverage_diagnostic.py`.
"""
from __future__ import annotations

import unicodedata

# Combining marks that should be filtered out after NFD decomposition.
# These cover acute (U+0301), grave (U+0300), circumflex/perispomeni (U+0342),
# smooth breathing/koronis (U+0313), rough breathing/dasia (U+0314),
# iota subscript/ypogegrammeni (U+0345), diaeresis (U+0308),
# macron (U+0304), breve (U+0306).
_COMBINING_MARKS_TO_DROP = {
    "̀",  # combining grave accent
    "́",  # combining acute accent
    "̄",  # combining macron
    "̆",  # combining breve
    "̈",  # combining diaeresis
    "̓",  # combining comma above (smooth breathing)
    "̔",  # combining reversed comma above (rough breathing)
    "͂",  # combining greek perispomeni (circumflex)
    "ͅ",  # combining greek ypogegrammeni (iota subscript)
}


def normalize_greek_token(raw) -> str:
    """Canonical normalization for a single Ancient Greek token.

    Order:
      1. NFD decomposition (split precomposed chars into base + combining marks).
      2. Drop polytonic combining marks (accents, breathings, iota subscript,
         diaeresis, macron, breve).
      3. NFC recomposition (rejoin any remaining base+mark pairs).
      4. Normalize final sigma `ς` -> `σ`.
      5. Lowercase.
      6. Strip whitespace.

    Safe on None/empty (returns ""). Idempotent.
    """
    if raw is None:
        return ""
    s = unicodedata.normalize("NFD", str(raw))
    s = "".join(ch for ch in s if ch not in _COMBINING_MARKS_TO_DROP)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("ς", "σ").replace("Σ", "σ")
    return s.lower().strip()
