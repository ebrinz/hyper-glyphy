import pytest


def test_nfc_normalization():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    decomposed = "šesz"  # s + combining caron + esz
    precomposed = "šesz"   # š + esz
    assert normalize_akkadian_token(decomposed) == normalize_akkadian_token(precomposed)


def test_subscripts_to_ascii():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token("šar₂ru") == "szar2ru"
    assert normalize_akkadian_token("₀₁₂₃₄₅₆₇₈₉") == "0123456789"


def test_strips_determinative_braces():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token("{d}šamaš") == "dszamasz"
    assert normalize_akkadian_token("{lú}šarru") == "luszarru"


def test_oracc_to_atf_letters():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token("šarru") == "szarru"
    assert normalize_akkadian_token("ḫamru") == "hamru"
    assert normalize_akkadian_token("ṣabum") == "sabum"
    assert normalize_akkadian_token("ṭuppum") == "tuppum"


def test_drops_hyphens():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token("šar-ru-um") == "szarrum"
    assert normalize_akkadian_token("a-na") == "ana"


def test_lowercases_logograms_and_mixedcase():
    """ALL-CAPS logograms and mixed-case forms both lowercase."""
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token("LUGAL") == "lugal"
    assert normalize_akkadian_token("Šarrum") == "szarrum"


def test_strips_whitespace():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token(" šarrum ") == "szarrum"
    assert normalize_akkadian_token("\tšarru\n") == "szarru"


def test_handles_empty_and_none():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token("") == ""
    assert normalize_akkadian_token(None) == ""


def test_idempotent():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    for raw in ("šarrum", "{d}šamaš", "šar-ru-um", "ŠARRUM", "ʾanāku"):
        once = normalize_akkadian_token(raw)
        twice = normalize_akkadian_token(once)
        assert once == twice, f"not idempotent on {raw!r}: {once!r} -> {twice!r}"


def test_mimation_alternates():
    from languages.akkadian.scripts.akkadian_normalize import mimation_alternates
    assert set(mimation_alternates("szarrum")) == {"szarrum", "szarru"}
    assert set(mimation_alternates("szarru")) == {"szarru"}
    assert set(mimation_alternates("ana")) == {"ana"}
    assert set(mimation_alternates("")) == set()
