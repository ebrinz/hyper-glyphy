import json

import numpy as np
import pytest

from shared.scripts.doc_eval import (
    GENRE_CLASSES,
    doc_centroid,
    loo_nearest_centroid,
    parse_etcsl_compositions,
    sif_weights,
)


def test_parse_etcsl_compositions():
    records = [
        {"line_id": "c141.A.1", "transliteration": "lugal kur-ra", "translation": ""},
        {"line_id": "c141.A.2", "transliteration": "e2 gal", "translation": ""},
        {"line_id": "c2554.A.1", "transliteration": "lugal an-na", "translation": ""},
        {"line_id": "c341.A.1", "transliteration": "x", "translation": ""},  # class 3 -> dropped
    ]
    comps = parse_etcsl_compositions(records)
    assert set(comps) == {"c141", "c2554"}
    assert comps["c141"]["genre"] == GENRE_CLASSES["1"]
    assert comps["c2554"]["genre"] == GENRE_CLASSES["2"]
    assert "lugal" in comps["c141"]["tokens"]


def test_sif_weights_downweight_frequent():
    w = sif_weights({"the": 1000, "rare": 1})
    assert w["rare"] > w["the"]


def test_doc_centroid_skips_oov():
    vocab = {"lugal": 0}
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    c = doc_centroid(["lugal", "notavocabword"], vocab, vectors, {"lugal": 1.0})
    assert np.allclose(c, [1.0, 0.0])
    assert doc_centroid(["notavocabword"], vocab, vectors, {}) is None


def test_loo_nearest_centroid_separable():
    rng = np.random.RandomState(0)
    a = rng.randn(10, 4) + np.array([10, 0, 0, 0])
    b = rng.randn(10, 4) + np.array([0, 10, 0, 0])
    acc = loo_nearest_centroid(np.vstack([a, b]), ["A"] * 10 + ["B"] * 10)
    assert acc == 100.0


# PRE-STEP: _load_space sidecar resolution tests


def test_load_space_json_sidecar(tmp_path):
    from shared.scripts.doc_eval import _load_space

    npz_path = tmp_path / "foo_aligned_vectors.npz"
    np.savez(str(npz_path), vectors=np.array([[1.0, 0.0]], dtype=np.float32))

    sidecar = tmp_path / "foo_aligned_vocab.json"
    with open(sidecar, "w") as f:
        json.dump(["word1", "word2"], f)

    vocab, vectors = _load_space(npz_path)
    assert vocab == {"word1": 0, "word2": 1}
    assert vectors.shape == (1, 2)


def test_load_space_no_sidecar_raises(tmp_path):
    from shared.scripts.doc_eval import _load_space

    npz_path = tmp_path / "bar_aligned_vectors.npz"
    np.savez(str(npz_path), vectors=np.array([[1.0]], dtype=np.float32))

    with pytest.raises(FileNotFoundError) as exc_info:
        _load_space(npz_path)

    assert "bar_aligned_vocab.json" in str(exc_info.value)
    assert "bar_aligned_vocab.pkl" in str(exc_info.value)


# Step 2: MRR test


def test_mrr():
    from shared.scripts.doc_eval import mean_reciprocal_rank

    # ranks are 1-based positions of the true parallel
    assert mean_reciprocal_rank([1, 2, 4]) == round((1 + 0.5 + 0.25) / 3, 4)


def test_parallel_space_npz_resolution():
    from shared.scripts.doc_eval import parallel_space_npz

    ridge = parallel_space_npz("hittite", "ridge")
    assert ridge.name == "hittite_aligned_gemma_vectors.npz"
    assert ridge.parts[-3:-1] == ("hittite", "final_output")

    proc = parallel_space_npz("greek", "procrustes")
    assert proc.name == "greek_procrustes_gemma_vectors.npz"

    import pytest
    with pytest.raises(KeyError):
        parallel_space_npz("hittite", "bogus")
