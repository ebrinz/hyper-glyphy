"""
Canonical Sanskrit (IAST) token normalization.

Single source of truth for mapping DCS conllu forms/lemmas and MW headwords
(after SLP1->IAST conversion in 02) to a common token form.

Normalization choices:
  - Unicode NFC (compose base + combining marks into precomposed chars where
    they exist; candrabindu m̐ has no precomposed form and stays combining).
  - Lowercase.
  - Strip surrounding whitespace.
  - Diacritics are PRESERVED: ā/ī/ū/ṛ/ṝ/ḷ/ḹ/ṃ/ḥ/ṅ/ñ/ṭ/ḍ/ṇ/ś/ṣ are
    phonemic in IAST. Do NOT reuse greek_normalize here — it drops macron
    (U+0304) and breve (U+0306), which would collapse ā->a, ī->i, ū->u.

Used by scripts/02_parse_mw.py, scripts/05_clean_and_tokenize.py, and
scripts/06_extract_anchors.py.
"""
from __future__ import annotations

import unicodedata


def normalize_sanskrit_token(raw) -> str:
    """Canonical normalization for a single Sanskrit (IAST) token.

    Order: NFC -> lowercase -> strip. Safe on None/empty (returns "").
    Idempotent.
    """
    if raw is None:
        return ""
    s = unicodedata.normalize("NFC", str(raw))
    return s.lower().strip()
