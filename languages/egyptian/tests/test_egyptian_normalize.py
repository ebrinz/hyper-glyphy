import pytest


def test_normalize_strips_diacritics_to_mdc():
    from languages.egyptian.scripts.egyptian_normalize import normalize_egyptian_token

    assert normalize_egyptian_token("nṯr") == "nTr"
    assert normalize_egyptian_token("ḥr,w") == "Hr,w"
    assert normalize_egyptian_token("ḫꜥ") == "xa"
    assert normalize_egyptian_token("ꜣḫ") == "Ax"


def test_normalize_lowercases_and_strips():
    from languages.egyptian.scripts.egyptian_normalize import normalize_egyptian_token

    assert normalize_egyptian_token("  NTR  ") == "ntr"
    assert normalize_egyptian_token("WSIR") == "wsir"


def test_normalize_handles_none_and_empty():
    from languages.egyptian.scripts.egyptian_normalize import normalize_egyptian_token

    assert normalize_egyptian_token(None) == ""
    assert normalize_egyptian_token("") == ""


def test_normalize_idempotent():
    from languages.egyptian.scripts.egyptian_normalize import normalize_egyptian_token

    for raw in ["nṯr", "ḥr,w", "ḫꜥ", "ꜣḫ", "ms(w),t", "wsjr"]:
        once = normalize_egyptian_token(raw)
        twice = normalize_egyptian_token(once)
        assert once == twice, f"not idempotent: {raw!r} -> {once!r} -> {twice!r}"


def test_normalize_preserves_parenthetical_notation():
    from languages.egyptian.scripts.egyptian_normalize import normalize_egyptian_token

    assert normalize_egyptian_token("ms(w),t") == "ms(w),t"
    assert normalize_egyptian_token("ḥm(,t)") == "Hm(,t)"
