from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parent.parent / "scripts" / "06_extract_anchors.py"
_spec = spec_from_file_location("grc_anchors_06", str(_path))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_anchors_carry_lemmas():
    lemmas = [{"cf": "θάλασσα", "form": "θαλάσσης"}] * 5
    lsj_index = {"θαλασσα": {"lemma_norm": "θαλασσα", "gloss_first": "the sea",
                             "glosses": ["the sea"]}}
    anchors, stats = _mod.extract_anchors(lemmas, lsj_index, {"sea"},
                                          min_occurrences=5)
    assert anchors
    by_surface = {a["greek"]: a for a in anchors}
    assert by_surface["θαλασσα"]["lemmas"] == ["θαλασσα"]
    assert by_surface["θαλασσησ"]["lemmas"] == ["θαλασσα"]
    assert stats["token_hit_rate"] == 1.0


def test_negated_lsj_gloss_falls_through():
    # LSJ "not to be injured, inviolable" must not anchor to "injured"
    lemmas = [{"cf": "ἄτρωτος", "form": "ἄτρωτος"}] * 5
    lsj_index = {"ατρωτοσ": {"lemma_norm": "ατρωτοσ",
                             "gloss_first": "not to be injured",
                             "glosses": ["not to be injured", "inviolable"]}}
    anchors, _ = _mod.extract_anchors(lemmas, lsj_index,
                                      {"injured", "inviolable"},
                                      min_occurrences=5)
    assert anchors
    assert all(a["english"] == "inviolable" for a in anchors)
