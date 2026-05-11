"""L5 test: OOV anchors with sufficient n-gram overlap are recovered via
FastText subword inference at evaluation time."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "09_align_and_evaluate.py"


def _load():
    spec = importlib.util.spec_from_file_location("align", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_build_training_data_uses_subword_inference_for_oov():
    """When fasttext_model is provided, OOV s_words are inferred (not dropped)."""
    mod = _load()

    class _FakeModel:
        vector_size = 4

        class wv:
            @staticmethod
            def get_vector(word):
                # Deterministic fake: vector is a function of the string length.
                return np.array([len(word), 0.5, -0.2, 0.1], dtype=np.float32)

    anchors = [
        {"akkadian": "in_vocab_word", "english": "king"},
        {"akkadian": "oov_word",      "english": "god"},
    ]
    sum_vocab = {"in_vocab_word": 0}
    sum_vectors = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    eng_vocab = {"king": 0, "god": 1}
    eng_vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    # Without model: only 1 valid anchor
    X1, Y1, valid1 = mod.build_training_data(
        anchors, sum_vocab, sum_vectors, eng_vocab, eng_vectors
    )
    assert len(valid1) == 1

    # With model: OOV anchor is recovered via subword inference
    X2, Y2, valid2 = mod.build_training_data(
        anchors, sum_vocab, sum_vectors, eng_vocab, eng_vectors,
        fasttext_model=_FakeModel()
    )
    assert len(valid2) == 2
    assert any(a.get("subword_inferred") for a in valid2)
