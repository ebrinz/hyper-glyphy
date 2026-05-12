import pytest


def test_strips_polytonic_accents():
    """Acute, grave, circumflex all strip to bare base letter."""
    from languages.greek.scripts.greek_normalize import normalize_greek_token
    # ά (acute) -> α
    assert normalize_greek_token("λόγος") == "λογοσ"
    # ὰ (grave) -> α
    assert normalize_greek_token("πατὴρ") == "πατηρ"
    # ᾶ (circumflex) -> α
    assert normalize_greek_token("πᾶς") == "πασ"


def test_strips_breathing_marks():
    """Smooth (᾿) and rough (῾) breathings strip."""
    from languages.greek.scripts.greek_normalize import normalize_greek_token
    # ἁ (rough) -> α; ἀ (smooth) -> α
    assert normalize_greek_token("ἁγιος") == "αγιοσ"
    assert normalize_greek_token("ἀνήρ") == "ανηρ"


def test_strips_iota_subscript():
    """Iota subscript (ᾳ) -> α (the subscript is lost, the base remains)."""
    from languages.greek.scripts.greek_normalize import normalize_greek_token
    assert normalize_greek_token("τῇ") == "τη"
    assert normalize_greek_token("ᾄδω") == "αδω"


def test_strips_diaeresis():
    """Diaeresis (ϊ, ϋ) -> bare ι, υ."""
    from languages.greek.scripts.greek_normalize import normalize_greek_token
    assert normalize_greek_token("προϊστημι") == "προιστημι"


def test_normalizes_final_sigma():
    """Final sigma ς -> σ (same letter, positional variant)."""
    from languages.greek.scripts.greek_normalize import normalize_greek_token
    assert normalize_greek_token("λόγος") == "λογοσ"
    assert normalize_greek_token("ἄνθρωπος") == "ανθρωποσ"


def test_lowercases_uppercase_greek():
    """Capital letters (Α, Β, ...) lowercase."""
    from languages.greek.scripts.greek_normalize import normalize_greek_token
    assert normalize_greek_token("Ἀθηνᾶ") == "αθηνα"
    assert normalize_greek_token("ΛΟΓΟΣ") == "λογοσ"


def test_strips_whitespace():
    from languages.greek.scripts.greek_normalize import normalize_greek_token
    assert normalize_greek_token(" λόγος ") == "λογοσ"
    assert normalize_greek_token("\tπατὴρ\n") == "πατηρ"


def test_handles_empty_and_none():
    from languages.greek.scripts.greek_normalize import normalize_greek_token
    assert normalize_greek_token("") == ""
    assert normalize_greek_token(None) == ""


def test_idempotent():
    from languages.greek.scripts.greek_normalize import normalize_greek_token
    for raw in ("λόγος", "Ἀθηνᾶ", "πᾶς", "τῇ", "ἁγιος", "ΛΟΓΟΣ"):
        once = normalize_greek_token(raw)
        twice = normalize_greek_token(once)
        assert once == twice, f"not idempotent on {raw!r}: {once!r} -> {twice!r}"


def test_combined_chain():
    """Polytonic + uppercase + final sigma + diaeresis all together."""
    from languages.greek.scripts.greek_normalize import normalize_greek_token
    assert normalize_greek_token("Λόγος") == "λογοσ"
    assert normalize_greek_token("Ὁμηρος") == "ομηροσ"
    # Uppercase iota subscript (capital alpha with iota adscript: ᾼ)
    assert normalize_greek_token("ᾌδης") == "αδησ"
