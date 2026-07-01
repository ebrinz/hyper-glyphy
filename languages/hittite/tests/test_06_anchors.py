from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parent.parent / "scripts" / "06_extract_anchors.py"
_spec = spec_from_file_location("hit_anchors_06", str(_path))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_german_anchors_carry_lemmas(monkeypatch):
    import numpy as np
    lemmas = [{"cf": "pai-", "form": "pait", "gw": "gehen"}] * 5
    monkeypatch.setattr(
        _mod, "translate_german_glosses",
        lambda glosses, vec, vocab: {g: "walking" for g in glosses},
    )
    anchors = _mod.extract_german_anchors(
        lemmas, np.zeros((1, 768), dtype=np.float32), ["walking"]
    )
    assert anchors
    for a in anchors:
        assert "lemmas" in a and a["lemmas"]
