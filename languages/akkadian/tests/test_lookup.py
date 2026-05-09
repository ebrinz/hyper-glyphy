import importlib.util
import sys
from pathlib import Path

import pytest

_LOOKUP_PATH = Path(__file__).resolve().parents[1] / "final_output" / "akkadian_lookup.py"


def _load():
    spec = importlib.util.spec_from_file_location("akkadian_lookup", _LOOKUP_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["akkadian_lookup"] = m
    spec.loader.exec_module(m)
    return m


def test_lookup_class_importable():
    m = _load()
    assert hasattr(m, "AkkadianLookup")


def test_lookup_rejects_unknown_space():
    m = _load()
    with pytest.raises(ValueError):
        m.AkkadianLookup(space="bogus")
