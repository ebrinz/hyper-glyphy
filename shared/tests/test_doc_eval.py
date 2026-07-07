import numpy as np

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
