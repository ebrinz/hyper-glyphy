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
    anchors, stats = _mod.extract_anchors(lemmas, mw_index, {"heavenly"}, min_occurrences=5)
    by_surface = {a["sanskrit"]: a for a in anchors}
    assert by_surface["deva"]["lemmas"] == ["deva"]      # cf normalized (lowercased)
    assert by_surface["devāḥ"]["lemmas"] == ["deva"]     # surface registered too
    assert by_surface["deva"]["english"] == "heavenly"
    assert by_surface["deva"]["source"] == "DCS+MW"
    assert stats["token_hit_rate"] == 1.0


def test_negated_gloss_skipped_not_harvested():
    # MW ahiṃsā: "not injuring anything, harmlessness". The Greek recipe
    # would anchor to "injuring" (antonym). We must fall through to
    # "harmlessness".
    lemmas = [{"cf": "ahiṃsā", "form": "ahiṃsā"}] * 5
    mw_index = {"ahiṃsā": _entry("ahiṃsā", ["not injuring anything", "harmlessness"])}
    vocab = {"injuring", "harmlessness"}
    anchors, _ = _mod.extract_anchors(lemmas, mw_index, vocab, min_occurrences=5)
    assert anchors
    assert all(a["english"] == "harmlessness" for a in anchors)


def test_all_glosses_negated_drops_anchor():
    lemmas = [{"cf": "abhāva", "form": "abhāva"}] * 5
    mw_index = {"abhāva": _entry("abhāva", ["not existing", "without form"])}
    anchors, stats = _mod.extract_anchors(
        lemmas, mw_index, {"existing", "form"}, min_occurrences=5)
    assert anchors == []
    assert stats["gloss_no_eng"] == 5


def test_hit_rate_in_stats():
    lemmas = [{"cf": "deva", "form": "deva"}] * 3 + [{"cf": "nope", "form": "nope"}] * 7
    mw_index = {"deva": _entry("deva", ["heavenly"])}
    _, stats = _mod.extract_anchors(lemmas, mw_index, {"heavenly"}, min_occurrences=1)
    assert stats["mw_hits"] == 3 and stats["mw_misses"] == 7
    assert abs(stats["token_hit_rate"] - 0.30) < 1e-9
    assert stats["token_hit_rate"] < _mod.MIN_HIT_RATE
