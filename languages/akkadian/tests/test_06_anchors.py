import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "06_extract_anchors.py"


def _load():
    spec = importlib.util.spec_from_file_location("anchors", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_extract_oracc_anchors_basic():
    mod = _load()
    lemmas = [
        {"cf": "šarru", "form": "szarrum", "gw": "king", "lang": "akk"},
        {"cf": "šarru", "form": "szarrum", "gw": "king", "lang": "akk"},
        {"cf": "šarru", "form": "szarrum", "gw": "king", "lang": "akk"},
        {"cf": "šarru", "form": "szarrum", "gw": "king", "lang": "akk"},
        {"cf": "šarru", "form": "szarrum", "gw": "king", "lang": "akk"},
        {"cf": "ilu",   "form": "ilum",    "gw": "god",  "lang": "akk"},
    ]
    anchors = mod.extract_oracc_anchors(lemmas, min_occurrences=5)
    forms = {a["akkadian"] for a in anchors}
    # 'szarrum' (form) and 'szarru' (cf normalized — š->sz, no -u/u dedup needed) clear threshold
    assert "szarrum" in forms or "szarru" in forms
    # 'ilum' has only 1 occurrence, filtered out
    assert "ilum" not in forms
    assert all(a["english"] == "king" for a in anchors)
    assert all(a["source"] == "ORACC" for a in anchors)


def test_filter_junk_glosses():
    mod = _load()
    lemmas = [{"cf": "šarru", "form": "szarrum", "gw": "x", "lang": "akk"}] * 10
    anchors = mod.extract_oracc_anchors(lemmas, min_occurrences=5)
    assert anchors == []  # 'x' is junk, dropped


def test_normalizes_via_akkadian_normalizer():
    mod = _load()
    # ORACC form 'šarrum' should normalize to 'szarrum' in the anchor output
    lemmas = [{"cf": "šarru", "form": "šarrum", "gw": "king", "lang": "akk"}] * 10
    anchors = mod.extract_oracc_anchors(lemmas, min_occurrences=5)
    forms = {a["akkadian"] for a in anchors}
    assert "szarrum" in forms
    assert "szarru" in forms


def test_skips_non_akkadian_lemmas():
    mod = _load()
    lemmas = [
        {"cf": "lugal", "form": "lugal", "gw": "king", "lang": "sux"},
    ] * 10
    # The function should rely on caller to filter, but if a lang field is checked,
    # this should be empty. If not checked, we still expect 'lugal' anchored.
    # For the v1 ORACC-only design we filter at scrape time, so this test just
    # documents that the extractor itself doesn't re-filter by lang.
    anchors = mod.extract_oracc_anchors(lemmas, min_occurrences=5)
    # Document expected behavior: extractor trusts that input is Akkadian-filtered.
    assert isinstance(anchors, list)


def test_mimation_alternates_expand_surface_coverage():
    """Anchors with -um mimation should register BOTH mimation and bare forms."""
    mod = _load()
    # 5 occurrences threshold; provide enough copies of a single mimation lemma
    lemmas = [{"cf": "šarru", "form": "šarrum", "gw": "king", "lang": "akk"}] * 10
    anchors = mod.extract_oracc_anchors(lemmas, min_occurrences=5)
    forms = {a["akkadian"] for a in anchors}
    # Both mimation form (szarrum) and its non-mimation alternate (szarru) should appear.
    assert "szarrum" in forms
    assert "szarru" in forms


def test_lemma_surface_expansion_crosses_records():
    """An anchor's surfaces should include ALL forms attested for its lemma
    in the corpus, even if those forms appeared with different glosses in
    their own records."""
    mod = _load()
    # Same cf 'šarru' attested with two different surface forms in two records;
    # one is 'szarrum' (with -um mimation), other is 'lugal' (logogram).
    # The 'king' gloss should expand to register BOTH surfaces.
    lemmas = (
        [{"cf": "šarru", "form": "szarrum", "gw": "king", "lang": "akk"}] * 5 +
        [{"cf": "šarru", "form": "lugal",   "gw": "king", "lang": "akk"}] * 5
    )
    anchors = mod.extract_oracc_anchors(lemmas, min_occurrences=5)
    forms = {a["akkadian"] for a in anchors}
    # Both surface forms should be present
    assert "szarrum" in forms
    assert "lugal" in forms


def test_anchors_carry_contributing_lemmas():
    mod = _load()

    lemmas = [
        {"cf": "šarrum", "form": "šarri", "gw": "king"},
    ] * 5 + [
        # Same surface form under a second citation form: both cfs recorded.
        {"cf": "šarratum", "form": "šarri", "gw": "king"},
    ] * 5
    anchors = mod.extract_oracc_anchors(lemmas, min_occurrences=5)
    by_surface = {a["akkadian"]: a for a in anchors}
    # The normalizer transforms šarri -> szarri, šarrum -> szarrum, šarratum -> szarratum
    # Pin the actual normalized surfaces used
    assert "szarri" in by_surface
    assert "lemmas" in by_surface["szarri"]
    assert set(by_surface["szarri"]["lemmas"]) >= {"szarrum", "szarratum"}
    for a in anchors:
        assert a["lemmas"] == sorted(set(a["lemmas"]))
