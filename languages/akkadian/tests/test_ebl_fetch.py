import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ebl_fetch.py"


def _load():
    spec = importlib.util.spec_from_file_location("ebl_fetch", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_filter_ob_lemmas_keeps_ob_only():
    mod = _load()
    lemmas = [
        {"lemma": ["šarrum"], "guideWord": "king", "attested": True,
         "amplifiedMeanings": [], "logograms": [],
         "periodAttestation": ["Old Babylonian"]},
        {"lemma": ["bēl"], "guideWord": "lord", "attested": True,
         "amplifiedMeanings": [], "logograms": [],
         "periodAttestation": ["Neo-Assyrian"]},
        {"lemma": ["ilum"], "guideWord": "god", "attested": True,
         "amplifiedMeanings": [], "logograms": [],
         "periodAttestation": ["Old Babylonian", "Middle Babylonian"]},
    ]
    ob = mod.filter_ob_lemmas(lemmas)
    assert len(ob) == 2
    assert {entry["lemma"] for entry in ob} == {"šarrum", "ilum"}


def test_filter_ob_drops_no_gloss():
    mod = _load()
    lemmas = [
        {"lemma": ["x"], "guideWord": "", "attested": True,
         "amplifiedMeanings": [], "logograms": [],
         "periodAttestation": ["Old Babylonian"]},
        {"lemma": ["šarrum"], "guideWord": "king", "attested": True,
         "amplifiedMeanings": [], "logograms": [],
         "periodAttestation": ["Old Babylonian"]},
    ]
    ob = mod.filter_ob_lemmas(lemmas)
    assert len(ob) == 1
    assert ob[0]["lemma"] == "šarrum"


def test_extract_logogram_forms():
    mod = _load()
    entry = {"lemma": "šarrum", "gloss": "king", "period": ["Old Babylonian"],
             "logograms": ["LUGAL"], "raw": {}}
    forms = mod.extract_surface_forms(entry)
    assert "šarrum" in forms
    assert "LUGAL" in forms
