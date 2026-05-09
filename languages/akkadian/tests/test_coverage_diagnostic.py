"""Tests for the akkadian coverage diagnostic, focused on the logogram_unmatched
bucket added by T14.

The full diagnostic requires numpy, gensim, fasttext model, ridge weights, and a
real corpus — those are integration concerns. This test file validates the
classifier function in isolation with a synthetic context.
"""
import importlib.util
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coverage_diagnostic.py"


def _load():
    import sys
    spec = importlib.util.spec_from_file_location("cov", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    # Register in sys.modules so @dataclass can resolve string annotations.
    sys.modules["cov"] = m
    spec.loader.exec_module(m)
    return m


def test_logogram_unmatched_in_primary_cause_order():
    mod = _load()
    assert "logogram_unmatched" in mod.PRIMARY_CAUSE_ORDER


def test_classify_miss_uppercase_anchor_returns_logogram_unmatched():
    """An all-uppercase anchor that doesn't recover via earlier buckets falls
    into logogram_unmatched."""
    pytest.importorskip("numpy")
    import numpy as np

    mod = _load()

    class _Ctx:
        # Empty vocab — nothing recovers via normalization or surface map.
        fused_vocab = frozenset()
        glove_vocab = frozenset()
        gemma_vocab = frozenset()
        corpus_frequency: dict = {}
        lemma_surface_map: dict = {}
        # Unused attributes for this test path:
        fasttext_model = None
        gemma_english_vocab: list = []
        gemma_english_vectors = np.zeros((0, 768), dtype=np.float32)
        ridge_gemma_coef = np.zeros((768, 1536), dtype=np.float32)
        ridge_gemma_intercept = np.zeros((768,), dtype=np.float32)

    bucket = mod.classify_miss(
        anchor={"akkadian": "LUGAL", "english": "king"},
        ctx=_Ctx(),
        trained_ngrams=frozenset(),
    )
    assert bucket == "logogram_unmatched"


def test_classify_miss_normalization_still_wins_over_logogram():
    """If a non-uppercase anchor recovers via normalization, that bucket fires
    first — logogram_unmatched only catches all-caps cases that don't recover."""
    pytest.importorskip("numpy")
    import numpy as np

    mod = _load()

    class _Ctx:
        fused_vocab = frozenset({"szarrum"})
        glove_vocab = frozenset()
        gemma_vocab = frozenset()
        corpus_frequency: dict = {}
        lemma_surface_map: dict = {}
        fasttext_model = None
        gemma_english_vocab: list = []
        gemma_english_vectors = np.zeros((0, 768), dtype=np.float32)
        ridge_gemma_coef = np.zeros((768, 1536), dtype=np.float32)
        ridge_gemma_intercept = np.zeros((768,), dtype=np.float32)

    bucket = mod.classify_miss(
        anchor={"akkadian": "šarrum", "english": "king"},
        ctx=_Ctx(),
        trained_ngrams=frozenset(),
    )
    assert bucket == "normalization_recoverable"
