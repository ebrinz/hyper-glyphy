import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "03_scrape_dcclt.py"


def _load():
    spec = importlib.util.spec_from_file_location("scrape_dcclt", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_extract_pairs_from_aligned_columns():
    mod = _load()
    text = {
        "cdl": [{
            "type": "line-start",
            "cdl": [
                {"f": {"lang": "sux", "form": "lugal", "cf": "lugal", "gw": "king"}},
                {"f": {"lang": "akk", "form": "szarrum", "cf": "šarru", "gw": "king"}},
            ]
        }, {
            "type": "line-start",
            "cdl": [
                {"f": {"lang": "sux", "form": "dingir", "cf": "dingir", "gw": "god"}},
                {"f": {"lang": "akk", "form": "ilum", "cf": "ilu", "gw": "god"}},
            ]
        }]
    }
    pairs = mod.extract_pairs(text)
    assert len(pairs) == 2
    assert pairs[0]["sumerian_cf"] == "lugal"
    assert pairs[0]["akkadian_cf"] == "šarru"
    assert pairs[1]["sumerian_cf"] == "dingir"
    assert pairs[1]["akkadian_cf"] == "ilu"


def test_skips_lines_with_only_one_language():
    mod = _load()
    text = {
        "cdl": [{
            "type": "line-start",
            "cdl": [{"f": {"lang": "sux", "form": "lugal", "cf": "lugal", "gw": "king"}}],
        }, {
            "type": "line-start",
            "cdl": [{"f": {"lang": "akk", "form": "ilum", "cf": "ilu", "gw": "god"}}],
        }]
    }
    pairs = mod.extract_pairs(text)
    assert pairs == []


def test_pair_includes_glosses():
    mod = _load()
    text = {
        "cdl": [{
            "type": "line-start",
            "cdl": [
                {"f": {"lang": "sux", "form": "lugal", "cf": "lugal", "gw": "king"}},
                {"f": {"lang": "akk", "form": "szarrum", "cf": "šarru", "gw": "king"}},
            ]
        }]
    }
    pairs = mod.extract_pairs(text)
    assert pairs[0]["sumerian_gw"] == "king"
    assert pairs[0]["akkadian_gw"] == "king"
