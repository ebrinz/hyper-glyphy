from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parent.parent / "scripts" / "02_parse_mw.py"
_spec = spec_from_file_location("san_parse_02", str(_path))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_mw.xml"


def test_slp1_to_iast_full_coverage():
    cases = {
        "ahiMsA": "ahiṃsā",
        "kfzRa": "kṛṣṇa",
        "SAstra": "śāstra",
        "buddhO": "buddhau",   # O -> au digraph expansion
        "vEdya": "vaidya",     # E -> ai digraph expansion
        "guRa": "guṇa",
        "duHKa": "duḥkha",
        "aNga": "aṅga",
        "jYAna": "jñāna",
        "wIkA": "ṭīkā",
        "QORqa": "ḍhauṇḍa",
        "pfTvI": "pṛthvī",
        "kAvya": "kāvya",
    }
    for slp, iast in cases.items():
        assert _mod.slp1_to_iast(slp) == iast, slp


def test_parse_entries_and_glosses():
    entries = _mod.parse_mw_file(FIXTURE)
    by_norm = {e["lemma_norm"]: e for e in entries}

    deva = by_norm["deva"]
    assert deva["key1"] == "deva"
    assert deva["glosses"][:2] == ["heavenly", "divine"]
    assert deva["gloss_first"] == "heavenly"

    ahimsa = by_norm["ahiṃsā"]
    # parenthetical stripped; sources (<ls>) and grammar (<lex>) excluded
    assert ahimsa["glosses"] == ["not injuring anything", "harmlessness"]

    akara = by_norm["akāra"]
    assert akara["glosses"] == ["the letter or sound"]


def test_dedup_keeps_first_homonym():
    entries = _mod.parse_mw_file(FIXTURE)
    deva_entries = [e for e in entries if e["lemma_norm"] == "deva"]
    assert len(deva_entries) == 1
    assert deva_entries[0]["gloss_first"] == "heavenly"
