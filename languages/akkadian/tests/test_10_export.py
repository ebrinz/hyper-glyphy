import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "10_export_production.py"


def test_export_module_imports():
    spec = importlib.util.spec_from_file_location("export", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert hasattr(m, "main")
    assert hasattr(m, "project_all_vectors")
