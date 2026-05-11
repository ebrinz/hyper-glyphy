import pytest


def test_nfc_normalization():
    from languages.hittite.scripts.hittite_normalize import normalize_hittite_token
    decomposed = "ša"  # s + combining caron + a
    precomposed = "ša"
    assert normalize_hittite_token(decomposed) == normalize_hittite_token(precomposed)


def test_subscripts_to_ascii():
    from languages.hittite.scripts.hittite_normalize import normalize_hittite_token
    assert normalize_hittite_token("hu₂uk") == "hu2uk"
    assert normalize_hittite_token("₀₁₂₃₄₅₆₇₈₉") == "0123456789"


def test_strips_determinative_braces():
    from languages.hittite.scripts.hittite_normalize import normalize_hittite_token
    assert normalize_hittite_token("{d}šamaš") == "dszamasz"
    assert normalize_hittite_token("{lú}šarru") == "luszarru"


def test_oracc_to_atf_letters():
    from languages.hittite.scripts.hittite_normalize import normalize_hittite_token
    assert normalize_hittite_token("šarru") == "szarru"
    assert normalize_hittite_token("ḫamru") == "hamru"
    assert normalize_hittite_token("nekuzi") == "nekuzi"


def test_drops_hyphens_and_morpheme_equals():
    """Hittite uses both '-' (syllable break) and '=' (clitic boundary).
    Both should drop in normalization to produce the joined citation form."""
    from languages.hittite.scripts.hittite_normalize import normalize_hittite_token
    assert normalize_hittite_token("nu-uš-ša-an") == "nuszszan"
    assert normalize_hittite_token("nu=ššan") == "nuszszan"
    assert normalize_hittite_token("a-da-an-zi") == "adanzi"


def test_lowercases_logograms_and_mixedcase():
    from languages.hittite.scripts.hittite_normalize import normalize_hittite_token
    assert normalize_hittite_token("LUGAL") == "lugal"
    assert normalize_hittite_token("Šarrum") == "szarrum"


def test_strips_whitespace():
    from languages.hittite.scripts.hittite_normalize import normalize_hittite_token
    assert normalize_hittite_token(" maḫḫan ") == "mahhan"
    assert normalize_hittite_token("\tnekuzi\n") == "nekuzi"


def test_handles_empty_and_none():
    from languages.hittite.scripts.hittite_normalize import normalize_hittite_token
    assert normalize_hittite_token("") == ""
    assert normalize_hittite_token(None) == ""


def test_idempotent():
    from languages.hittite.scripts.hittite_normalize import normalize_hittite_token
    for raw in ("šarrum", "{d}šamaš", "nu=ššan", "MAḪḪAN", "ʾanāku", "a-da-an-zi"):
        once = normalize_hittite_token(raw)
        twice = normalize_hittite_token(once)
        assert once == twice, f"not idempotent on {raw!r}: {once!r} -> {twice!r}"


def test_clitic_alternates_returns_host_and_full():
    """Clitic chain like 'nu=ššan' should expose [normalized_full, normalized_host].
    Host = portion before first '='. Useful for anchor matching against a lemma
    cited in its bare (no-clitic) form."""
    from languages.hittite.scripts.hittite_normalize import clitic_alternates
    assert set(clitic_alternates("nu=ššan")) == {"nu=ššan", "nu"}
    assert set(clitic_alternates("kuit")) == {"kuit"}  # no clitic
    assert set(clitic_alternates("")) == set()
