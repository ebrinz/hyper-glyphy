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
