import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "01_scrape_oracc_ob.py"


def _load():
    spec = importlib.util.spec_from_file_location("scrape_oracc_ob", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_walk_cdl_extracts_akkadian_lemmas():
    mod = _load()
    text = {
        "cdl": [{
            "cdl": [
                {"f": {"lang": "akk", "form": "szarrum", "cf": "šarru", "gw": "king", "pos": "N"}},
                {"f": {"lang": "sux", "form": "lugal", "cf": "lugal", "gw": "king"}},
                {"f": {"lang": "akk-x-stdbab", "form": "ilum", "cf": "ilu", "gw": "god"}},
            ]
        }]
    }
    lemmas = mod.extract_lemmas(text)
    assert len(lemmas) == 2
    assert lemmas[0]["cf"] == "šarru"
    assert lemmas[1]["cf"] == "ilu"


def test_extract_lines_returns_text_lines():
    mod = _load()
    text = {
        "cdl": [{
            "type": "line-start",
            "cdl": [{"f": {"lang": "akk", "form": "szarrum"}}],
        }, {
            "type": "line-start",
            "cdl": [{"f": {"lang": "akk", "form": "ilum"}}],
        }]
    }
    lines = mod.extract_lines(text)
    assert lines


def test_oracc_projects_seed_list_present():
    mod = _load()
    assert isinstance(mod.ORACC_PROJECTS, list) and len(mod.ORACC_PROJECTS) > 0
