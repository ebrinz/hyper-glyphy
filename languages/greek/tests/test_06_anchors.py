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
    anchors = _mod.extract_anchors(lemmas, lsj_index, {"sea"}, min_occurrences=5)
    assert anchors
    by_surface = {a["greek"]: a for a in anchors}
    assert by_surface["θαλασσα"]["lemmas"] == ["θαλασσα"]
    assert by_surface["θαλασσησ"]["lemmas"] == ["θαλασσα"]
