import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "05_clean_and_tokenize.py"


def _load():
    spec = importlib.util.spec_from_file_location("clean", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_strips_brackets_and_keeps_content():
    mod = _load()
    assert "szarrum" in mod.clean_atf_line("[šar-ru-um]").split()


def test_strips_determinatives_keeps_content():
    mod = _load()
    cleaned = mod.clean_atf_line("{d}šamaš").split()
    assert "szamasz" in cleaned or "dszamasz" in cleaned


def test_drops_uppercase_only_signs():
    mod = _load()
    out = mod.clean_atf_line("LUGAL ša-ar")
    assert "lugal" not in out.split()
    assert any(t.startswith("sza") for t in out.split())


def test_normalizes_akkadian_diacritics():
    mod = _load()
    out = mod.clean_atf_line("šar-ru ḫar-ra")
    tokens = out.split()
    assert any(t.startswith("sz") for t in tokens)
    assert any(t.startswith("h") for t in tokens)
