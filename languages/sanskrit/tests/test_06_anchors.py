from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parent.parent / "scripts" / "06_extract_anchors.py"
_spec = spec_from_file_location("san_anchors_06", str(_path))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _entry(norm, glosses):
    return {"lemma_norm": norm, "gloss_first": glosses[0], "glosses": glosses}


def test_anchors_carry_lemmas_and_surfaces():
    lemmas = [{"cf": "Deva", "form": "devāḥ"}] * 5
    mw_index = {"deva": _entry("deva", ["heavenly", "divine"])}
    anchors, stats = _mod.extract_anchors(lemmas, mw_index, {"heavenly"},
                                          min_occurrences=5)
    by_surface = {a["sanskrit"]: a for a in anchors}
    assert by_surface["deva"]["lemmas"] == ["deva"]
    assert by_surface["devāḥ"]["lemmas"] == ["deva"]
    assert by_surface["deva"]["english"] == "heavenly"
    assert stats["token_hit_rate"] == 1.0


def test_negated_gloss_regression():
    # end-to-end guard that the slot actually routes through shared filters
    lemmas = [{"cf": "ahiṃsā", "form": "ahiṃsā"}] * 5
    mw_index = {"ahiṃsā": _entry("ahiṃsā",
                                 ["not injuring anything", "harmlessness"])}
    anchors, _ = _mod.extract_anchors(lemmas, mw_index,
                                      {"injuring", "harmlessness"},
                                      min_occurrences=5)
    assert anchors and all(a["english"] == "harmlessness" for a in anchors)


def test_hit_rate_in_stats():
    lemmas = [{"cf": "deva", "form": "deva"}] * 3 + \
             [{"cf": "nope", "form": "nope"}] * 7
    mw_index = {"deva": _entry("deva", ["heavenly"])}
    _, stats = _mod.extract_anchors(lemmas, mw_index, {"heavenly"},
                                    min_occurrences=1)
    assert stats["hits"] == 3 and stats["misses"] == 7
    assert abs(stats["token_hit_rate"] - 0.30) < 1e-9
