import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
_spec = spec_from_file_location("doc_eval_mod", str(_ROOT / "shared" / "scripts" / "doc_eval.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_sanskrit_documents_load(tmp_path, monkeypatch):
    fixture = [
        {"p_number": "dcs-450-9905", "text_name": "Ṛgveda", "chapter": "ṚV, 10, 129",
         "lines": ["nāsad āsīt no sad āsīt tadānīm"], "source": "DCS"},
        {"p_number": "dcs-450-9864", "text_name": "Ṛgveda", "chapter": "ṚV, 10, 90",
         "lines": ["sahasraśīrṣā puruṣaḥ"], "source": "DCS"},
    ]
    p = tmp_path / "sanskrit_texts.json"
    p.write_text(json.dumps(fixture), encoding="utf-8")
    monkeypatch.setattr(_mod, "SANSKRIT_TEXTS_PATH", p)
    docs = _mod._slot_documents("sanskrit")
    assert set(docs) == {"dcs-450-9905", "dcs-450-9864"}
    # normalizer applied: lowercase IAST, diacritics preserved
    assert "nāsad" in docs["dcs-450-9905"]
    assert "sahasraśīrṣā" in docs["dcs-450-9864"]
