import pytest


def test_normalize_anchor_format():
    from languages.egyptian.scripts.anchors_06 import normalize_anchors

    raw = [
        {"hieroglyphic": "nṯr", "english": "god", "german": "gott", "confidence": 0.87, "frequency": 234},
        {"hieroglyphic": "ḥr,w", "english": "horus", "german": "horus", "confidence": 0.95, "frequency": 150},
    ]

    result = normalize_anchors(raw)

    assert len(result) == 2
    assert result[0]["egyptian"] == "Hr,w"
    assert result[0]["english"] == "horus"
    assert result[0]["confidence"] == 0.95
    assert result[0]["source"] == "TLA/Ramses"
    assert "hieroglyphic" not in result[0]
    assert "german" not in result[0]


def test_filter_single_char_english():
    from languages.egyptian.scripts.anchors_06 import normalize_anchors

    raw = [
        {"hieroglyphic": "n", "english": "the", "german": "der", "confidence": 0.34, "frequency": 8829},
        {"hieroglyphic": "m", "english": "a", "german": "ein", "confidence": 0.37, "frequency": 8467},
        {"hieroglyphic": "x", "english": "x", "german": "x", "confidence": 0.50, "frequency": 100},
        {"hieroglyphic": "t", "english": "bread", "german": "brot", "confidence": 0.80, "frequency": 500},
    ]

    result = normalize_anchors(raw)

    english_words = [a["english"] for a in result]
    # stopword-only glosses rejected by shared gw_is_usable (suite v2)
    assert "the" not in english_words
    assert "a" not in english_words
    assert "x" not in english_words
    assert "bread" in english_words


def test_filter_low_frequency():
    from languages.egyptian.scripts.anchors_06 import normalize_anchors

    raw = [
        {"hieroglyphic": "nṯr", "english": "god", "german": "gott", "confidence": 0.87, "frequency": 10},
        {"hieroglyphic": "rare", "english": "rare-word", "german": "selten", "confidence": 0.90, "frequency": 3},
    ]

    result = normalize_anchors(raw, min_frequency=5)

    assert len(result) == 1
    assert result[0]["english"] == "god"


def test_filter_numeric_english():
    from languages.egyptian.scripts.anchors_06 import normalize_anchors

    raw = [
        {"hieroglyphic": "nṯr", "english": "god", "german": "gott", "confidence": 0.87, "frequency": 100},
        {"hieroglyphic": "num", "english": "123", "german": "123", "confidence": 0.50, "frequency": 100},
    ]

    result = normalize_anchors(raw)

    assert len(result) == 1
    assert result[0]["english"] == "god"


def test_deduplicates_by_egyptian_key():
    from languages.egyptian.scripts.anchors_06 import normalize_anchors

    raw = [
        {"hieroglyphic": "nṯr", "english": "god", "german": "gott", "confidence": 0.87, "frequency": 234},
        {"hieroglyphic": "nṯr", "english": "divine", "german": "goettlich", "confidence": 0.60, "frequency": 100},
    ]

    result = normalize_anchors(raw)

    ntr_entries = [a for a in result if a["egyptian"] == "nTr"]
    assert len(ntr_entries) == 1
    assert ntr_entries[0]["confidence"] == 0.87
