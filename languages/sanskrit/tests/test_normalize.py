import unicodedata

from languages.sanskrit.scripts.sanskrit_normalize import normalize_sanskrit_token


def test_lowercase_and_strip():
    assert normalize_sanskrit_token("  Agni ") == "agni"


def test_nfc_composition():
    # "a" + combining macron (U+0304) must compose to precomposed ā (U+0101)
    decomposed = "ātman"
    out = normalize_sanskrit_token(decomposed)
    assert out == "ātman"
    assert unicodedata.is_normalized("NFC", out)


def test_diacritics_preserved():
    # Unlike greek_normalize, macrons and dots must survive
    assert normalize_sanskrit_token("ṛṣi") == "ṛṣi"
    assert normalize_sanskrit_token("kṛṣṇa") == "kṛṣṇa"
    assert normalize_sanskrit_token("ahiṃsā") == "ahiṃsā"


def test_idempotent():
    once = normalize_sanskrit_token("Āsīt")
    assert normalize_sanskrit_token(once) == once


def test_none_and_empty():
    assert normalize_sanskrit_token(None) == ""
    assert normalize_sanskrit_token("") == ""
