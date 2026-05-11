"""
Canonical Hittite token normalization.

Single source of truth for mapping TLHdig/HPM citation forms and inflected
surface forms to the common ATF-style token form produced by
`scripts/05_clean_and_tokenize.py`.

Used by `scripts/06_extract_anchors.py` (anchor side) and
`scripts/coverage_diagnostic.py` (audit/diagnostic side). Keeps normalization
in one place to prevent drift between anchors and corpus.

Hittite-specific concerns vs Akkadian:
- Hittite uses both '-' (syllable boundary) and '=' (clitic boundary).
  Both are dropped in normalization to produce the joined citation form.
- No mimation (Hittite is Indo-European). Instead, `clitic_alternates` returns
  the host form (portion before first '=') in addition to the full form.
- Same NFC + ORACC->ATF letter map as Akkadian (š, ḫ, ṣ, ṭ, ā/ē/ī/ū, etc.).
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


def normalize_hittite_token(raw) -> str:
    """Canonical normalization for a single Hittite token.

    Order:
      1. NFC unicode normalization.
      2. Subscript digits -> ASCII.
      3. Strip {determinative} braces, keep content.
      4. ORACC unicode letters -> ATF (š -> sz, ā -> a, etc.).
      5. Drop hyphens AND equals (Hittite's two morpheme/clitic boundaries).
      6. Lowercase + strip whitespace.

    Safe on None/empty (returns ""). Idempotent.
    """
    if raw is None:
        return ""
    s = unicodedata.normalize("NFC", str(raw))
    s = s.translate(_SUBSCRIPT_MAP)
    s = _BRACE_RE.sub(r"\1", s)
    for old, new in _ORACC_TO_ATF.items():
        s = s.replace(old, new)
    s = s.replace("-", "").replace("=", "")
    # Collapse consecutive identical vowels (e.g., from hyphen-bridges:
    # `nu-uš-ša-an` -> `nuuszszaan` -> `nuszszan`).
    s = re.sub(r"([aeiou])\1+", r"\1", s)
    return s.lower().strip()


def clitic_alternates(token: str) -> list[str]:
    """Return [token] plus its host form (portion before first '=') when applicable.

    Hittite clitic chains like `nu=ššan` consist of a host word `nu` plus one or
    more clitics. For anchor matching against a lemma cited in its bare form,
    we surface both the full token and the host.
    """
    if not token:
        return []
    out = [token]
    if "=" in token:
        host = token.split("=", 1)[0]
        if host and host != token:
            out.append(host)
    return out
