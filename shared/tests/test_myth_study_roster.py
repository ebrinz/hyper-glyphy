import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent.parent
_spec = spec_from_file_location("myth_study_mod", str(_ROOT / "shared" / "scripts" / "myth_study.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_slots_and_pairs():
    assert _mod.SLOTS == ("sumerian", "hittite", "greek", "sanskrit")
    pairs = _mod.enumerate_slot_pairs()
    assert len(pairs) == 6
    assert ("greek", "sanskrit") in pairs or ("sanskrit", "greek") in [tuple(p) for p in pairs]


def _fixture_texts():
    docs = []
    # pinned cosmogonic + vrtra + hymnic + royal chapters (content irrelevant)
    for p in (_mod.SANSKRIT_COSMOGONIC + tuple(x for m in _mod.SANSKRIT_MERGES.values() for x in m)
              + _mod.SANSKRIT_HYMNIC + _mod.SANSKRIT_ROYAL):
        book = "3" if p.startswith("dcs-464") else "10"
        docs.append({"p_number": p, "text_name": "Ṛgveda" if p.startswith("dcs-450") else "Atharvaveda (Śaunaka)",
                     "chapter": f"X, {book}, 1", "lines": ["indraḥ vṛtram ahan"] * 3, "source": "DCS"})
    # wisdom texts, two chapters each
    for name in _mod.SANSKRIT_WISDOM_TEXTS:
        for i in (1, 2):
            docs.append({"p_number": f"dcs-999-{abs(hash(name)) % 10000}{i}", "text_name": name,
                         "chapter": f"U, {i}", "lines": ["ātmā vai idam"] * 4, "source": "DCS"})
    # AV magical candidates: two long non-royal chapters in allowed books + one excluded-book chapter
    docs.append({"p_number": "dcs-464-90001", "text_name": "Atharvaveda (Śaunaka)",
                 "chapter": "AVŚ, 11, 3", "lines": ["ucchiṣṭaḥ"] * 50, "source": "DCS"})
    docs.append({"p_number": "dcs-464-90002", "text_name": "Atharvaveda (Śaunaka)",
                 "chapter": "AVŚ, 12, 1", "lines": ["bhūmiḥ"] * 40, "source": "DCS"})
    docs.append({"p_number": "dcs-464-90003", "text_name": "Atharvaveda (Śaunaka)",
                 "chapter": "AVŚ, 18, 4", "lines": ["funerary"] * 100, "source": "DCS"})
    return docs


def test_sanskrit_roster_from_fixture(tmp_path, monkeypatch):
    p = tmp_path / "sanskrit_texts.json"
    p.write_text(json.dumps(_fixture_texts()), encoding="utf-8")
    monkeypatch.setattr(_mod, "SANSKRIT_TEXTS_PATH", p)
    roster, tokens = _mod.build_sanskrit_roster()
    themes = set(roster)
    assert themes == {"cosmogonic", "hymnic", "wisdom", "royal_control", "magical"}
    cos_ids = {d["doc_id"] for d in roster["cosmogonic"]}
    assert set(_mod.SANSKRIT_COSMOGONIC) <= cos_ids and "vrtra" in cos_ids
    # vrtra merge concatenates all member tokens
    n_member_lines = sum(3 for _ in _mod.SANSKRIT_MERGES["vrtra"])
    assert len(tokens["vrtra"]) == n_member_lines * 3  # 3 tokens per line
    # wisdom grouped by text_name → one doc per text
    assert len(roster["wisdom"]) == len(_mod.SANSKRIT_WISDOM_TEXTS)
    # magical excludes books 14/18 and royal picks
    mag_ids = {d["doc_id"] for d in roster["magical"]}
    assert "dcs-464-90003" not in mag_ids and "dcs-464-90001" in mag_ids
    assert not (mag_ids & set(_mod.SANSKRIT_ROYAL))


def test_missing_pinned_id_raises(tmp_path, monkeypatch):
    docs = _fixture_texts()
    docs = [d for d in docs if d["p_number"] != _mod.SANSKRIT_COSMOGONIC[0]]
    p = tmp_path / "sanskrit_texts.json"
    p.write_text(json.dumps(docs), encoding="utf-8")
    monkeypatch.setattr(_mod, "SANSKRIT_TEXTS_PATH", p)
    with pytest.raises(ValueError, match=_mod.SANSKRIT_COSMOGONIC[0]):
        _mod.build_sanskrit_roster()
