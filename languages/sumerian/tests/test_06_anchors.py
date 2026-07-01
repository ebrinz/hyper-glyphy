import pytest


def test_extract_epsd2_anchors():
    """Extract Sumerian-English pairs from ORACC lemma data."""
    from languages.sumerian.scripts.anchors_06 import extract_epsd2_anchors

    lemmas = [
        {"form": "lugal", "cf": "lugal", "gw": "king", "pos": "N"},
        {"form": "lugal-e", "cf": "lugal", "gw": "king", "pos": "N"},
        {"form": "e2", "cf": "e", "gw": "house", "pos": "N"},
        {"form": "an", "cf": "an", "gw": "heaven", "pos": "N"},
        {"form": "mu", "cf": "mu", "gw": "name", "pos": "N"},
        {"form": "mu", "cf": "mu", "gw": "name", "pos": "N"},
    ]

    anchors = extract_epsd2_anchors(lemmas, min_occurrences=1)

    cfs = [a["sumerian"] for a in anchors]
    assert "lugal" in cfs
    assert "e" in cfs
    assert "an" in cfs


def test_extract_epsd2_anchors_min_occurrences():
    """Anchors below min_occurrences threshold should be filtered."""
    from languages.sumerian.scripts.anchors_06 import extract_epsd2_anchors

    lemmas = [
        {"form": "lugal", "cf": "lugal", "gw": "king", "pos": "N"},
        {"form": "lugal", "cf": "lugal", "gw": "king", "pos": "N"},
        {"form": "lugal", "cf": "lugal", "gw": "king", "pos": "N"},
        {"form": "rare", "cf": "rare", "gw": "rare-word", "pos": "N"},
    ]

    anchors = extract_epsd2_anchors(lemmas, min_occurrences=2)

    cfs = [a["sumerian"] for a in anchors]
    assert "lugal" in cfs
    assert "rare" not in cfs


def test_extract_cooccurrence_anchors():
    """Extract anchors from ETCSL parallel translations via co-occurrence."""
    from languages.sumerian.scripts.anchors_06 import extract_cooccurrence_anchors

    parallel_lines = [
        {"transliteration": "lugal e2-gal-la-ka", "translation": "the king of the palace"},
        {"transliteration": "lugal kalam-ma", "translation": "the king of the land"},
        {"transliteration": "dingir gal-gal-e-ne", "translation": "the great gods"},
    ]

    anchors = extract_cooccurrence_anchors(parallel_lines, min_cooccurrences=2, min_confidence=0.3)

    sumerians = [a["sumerian"] for a in anchors]
    assert "lugal" in sumerians


def test_merge_anchors():
    """Merge dictionary and co-occurrence anchors, preferring higher confidence."""
    from languages.sumerian.scripts.anchors_06 import merge_anchors

    dict_anchors = [
        {"sumerian": "lugal", "english": "king", "confidence": 0.95, "source": "ePSD2"},
    ]
    cooc_anchors = [
        {"sumerian": "lugal", "english": "king", "confidence": 0.60, "source": "ETCSL"},
        {"sumerian": "e2", "english": "house", "confidence": 0.55, "source": "ETCSL"},
    ]

    merged = merge_anchors(dict_anchors, cooc_anchors)

    assert len(merged) == 2
    lugal = next(a for a in merged if a["sumerian"] == "lugal")
    assert lugal["confidence"] == 0.95


def test_extract_epsd2_anchors_applies_full_normalization():
    """After the 2b normalization fix, extract_epsd2_anchors must apply the
    full canonical normalization chain (subscripts, braces, hyphens) and NOT
    just the ORACC letter map that normalize_oracc_cf used to apply.
    """
    # Load the leading-digit script by file path.
    import importlib.util
    from pathlib import Path

    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(
        "extract_anchors_06_mod",
        root / "scripts" / "06_extract_anchors.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lemmas = [
        # Hyphenated, subscripted, braced citation form — pre-fix this would
        # normalize to "{tug₂}mug" (only lowercase + ORACC letters applied).
        # Post-fix it must normalize to "tug2mug".
        {"cf": "{tug₂}mug",       "form": "{tug₂}mug",       "gw": "garment"},
        {"cf": "{tug₂}mug",       "form": "{tug₂}mug",       "gw": "garment"},
        {"cf": "{tug₂}mug",       "form": "{tug₂}mug",       "gw": "garment"},
        {"cf": "{tug₂}mug",       "form": "{tug₂}mug",       "gw": "garment"},
        {"cf": "{tug₂}mug",       "form": "{tug₂}mug",       "gw": "garment"},
        # Hyphenated citation form: "za₃-sze₃-la₂" -> "za3sze3la2".
        {"cf": "za₃-sze₃-la₂",    "form": "za₃-sze₃-la₂",    "gw": "container"},
        {"cf": "za₃-sze₃-la₂",    "form": "za₃-sze₃-la₂",    "gw": "container"},
        {"cf": "za₃-sze₃-la₂",    "form": "za₃-sze₃-la₂",    "gw": "container"},
        {"cf": "za₃-sze₃-la₂",    "form": "za₃-sze₃-la₂",    "gw": "container"},
        {"cf": "za₃-sze₃-la₂",    "form": "za₃-sze₃-la₂",    "gw": "container"},
    ]
    anchors = mod.extract_epsd2_anchors(lemmas, min_occurrences=5)

    sumerian_keys = {a["sumerian"] for a in anchors}
    assert "tug2mug" in sumerian_keys, (
        f"expected 'tug2mug' after full normalization, got {sumerian_keys!r}"
    )
    assert "za3sze3la2" in sumerian_keys, (
        f"expected 'za3sze3la2' after full normalization, got {sumerian_keys!r}"
    )
    # Regression: the unnormalized forms must NOT appear.
    assert "{tug₂}mug" not in sumerian_keys
    assert "za₃-sze₃-la₂" not in sumerian_keys


def test_epsd2_anchors_carry_lemmas():
    from languages.sumerian.scripts.anchors_06 import extract_epsd2_anchors

    lemmas = [{"cf": "lugal", "form": "lugal-e", "gw": "king"}] * 5
    anchors = extract_epsd2_anchors(lemmas, min_occurrences=5)
    by_surface = {a["sumerian"]: a for a in anchors}
    # Both the cf anchor and the form anchor attribute to the cf.
    # Note: normalization strips hyphens, so 'lugal-e' becomes 'lugale'
    assert by_surface["lugal"]["lemmas"] == ["lugal"]
    assert by_surface["lugale"]["lemmas"] == ["lugal"]


def test_cooccurrence_anchors_carry_singleton_lemmas():
    from languages.sumerian.scripts.anchors_06 import extract_cooccurrence_anchors

    lines = [{"transliteration": "lugal kur", "translation": "the king of the mountain"}] * 4
    anchors = extract_cooccurrence_anchors(lines, min_cooccurrences=3, min_confidence=0.3)
    assert anchors
    for a in anchors:
        assert a["lemmas"] == [a["sumerian"]]
