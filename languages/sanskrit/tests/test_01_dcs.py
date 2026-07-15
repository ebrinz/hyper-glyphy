from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parent.parent / "scripts" / "01_parse_dcs.py"
_spec = spec_from_file_location("san_parse_01", str(_path))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.conllu"


def test_parse_file_meta_and_lines():
    parsed = _mod.parse_file(FIXTURE)
    assert parsed["text"] == "Aitareyopaniṣad"
    assert parsed["text_id"] == "421"
    assert parsed["chapter_id"] == "8816"
    # sentence 1: 7 forms; sentence 2: saḥ īkṣata (malformed "iti" line and
    # PUNCT excluded, range/empty-node lines skipped)
    assert parsed["lines"][0] == "ātmā vai idam ekaḥ eva agre āsīt"
    assert parsed["lines"][1] == "saḥ īkṣata"


def test_lemma_records():
    parsed = _mod.parse_file(FIXTURE)
    recs = parsed["lemmas"]
    first = recs[0]
    assert first == {"form": "ātmā", "cf": "ātman", "gw": "", "pos": "NOUN", "lang": "san"}
    # gold lemma for saḥ is tad (suppletive) — the join key later
    by_form = {r["form"]: r for r in recs}
    assert by_form["saḥ"]["cf"] == "tad"
    # multiword surface string and empty node never become records
    assert "tacchrutvā" not in by_form
    assert "," not in by_form


def test_parse_loss_accounting():
    parsed = _mod.parse_file(FIXTURE)
    assert parsed["bad_lines"] == 1          # the 4-column "iti" line
    assert parsed["token_lines"] >= 10       # counted before validity checks
