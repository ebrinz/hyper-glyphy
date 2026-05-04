import json
import os
import tempfile

import numpy as np
import pytest


def test_project_all_vectors():
    from languages.egyptian.scripts.export_10 import project_all_vectors

    eg_vectors = np.random.randn(100, 1536).astype(np.float32)
    coef = np.random.randn(300, 1536).astype(np.float32)
    intercept = np.random.randn(300).astype(np.float32)

    projected = project_all_vectors(eg_vectors, coef, intercept)

    assert projected.shape == (100, 300)
    assert projected.dtype == np.float16


def test_export_writes_both_spaces_and_v2_metadata(tmp_path, monkeypatch):
    import languages.egyptian.scripts.export_10 as export_10_module

    n_eg = 4
    fused_dim = 1536
    glove_dim = 300
    gemma_dim = 768

    rng = np.random.default_rng(7)
    eg_vocab = ["nTr", "Hr", "wsjr", "Ast"]
    fused = rng.standard_normal((n_eg, fused_dim)).astype(np.float32)

    glove_coef = rng.standard_normal((glove_dim, fused_dim)).astype(np.float32)
    glove_intercept = rng.standard_normal(glove_dim).astype(np.float32)

    gemma_coef = rng.standard_normal((gemma_dim, fused_dim)).astype(np.float32)
    gemma_intercept = rng.standard_normal(gemma_dim).astype(np.float32)

    models = tmp_path / "models"
    results = tmp_path / "results"
    final = tmp_path / "final_output"
    models.mkdir()
    results.mkdir()

    np.savez_compressed(
        str(models / "fused_embeddings_1536d.npz"),
        vectors=fused,
        vocab=np.array(eg_vocab),
    )
    np.savez_compressed(
        str(models / "ridge_weights.npz"),
        coef=glove_coef,
        intercept=glove_intercept,
    )
    np.savez_compressed(
        str(models / "ridge_weights_gemma_whitened.npz"),
        coef=gemma_coef,
        intercept=gemma_intercept,
    )
    (results / "alignment_results.json").write_text(json.dumps({
        "accuracy": {"top1": 32.35, "top5": 41.47, "top10": 45.13},
        "config": {
            "alignment": "Ridge", "alpha": 0.001, "train_size": 5132,
            "test_size": 1283, "valid_anchors": 6415, "total_anchors": 8541,
            "egyptian_vocab": 10833, "fused_dim": 1536,
        },
    }))
    (results / "alignment_results_gemma_whitened.json").write_text(json.dumps({
        "accuracy": {"top1": 40.00, "top5": 50.00, "top10": 55.00},
        "config": {
            "alignment": "Ridge", "alpha": 100,
            "gemma_model": "google/embeddinggemma-300m", "gloss_hit_rate": 21.39,
            "test_size_count": 1283, "train_size": 5132, "valid_anchors": 6415,
            "total_anchors": 8541, "random_state": 42,
        },
    }))

    monkeypatch.setattr(export_10_module, "MODELS_DIR", models)
    monkeypatch.setattr(export_10_module, "RESULTS_DIR", results)
    monkeypatch.setattr(export_10_module, "FINAL_OUTPUT", final)
    export_10_module.main()

    assert (final / "egyptian_aligned_vectors.npz").exists()
    assert (final / "egyptian_aligned_gemma_vectors.npz").exists()
    assert (final / "egyptian_aligned_vocab.pkl").exists()
    assert (final / "metadata.json").exists()

    glove_npz = np.load(str(final / "egyptian_aligned_vectors.npz"))
    gemma_npz = np.load(str(final / "egyptian_aligned_gemma_vectors.npz"))
    assert glove_npz["vectors"].shape == (n_eg, glove_dim)
    assert gemma_npz["vectors"].shape == (n_eg, gemma_dim)
    assert glove_npz["vectors"].dtype == np.float16
    assert gemma_npz["vectors"].dtype == np.float16

    metadata = json.loads((final / "metadata.json").read_text())
    assert metadata["schema_version"] == 2
    assert metadata["shared"]["vocab_size"] == n_eg
    assert metadata["spaces"]["gemma"]["dim"] == 768
    assert metadata["spaces"]["glove"]["dim"] == 300


# --- EgyptianLookup tests ---


def _build_tiny_lookup(tmpdir: str, seed: int = 42):
    from languages.egyptian.final_output.egyptian_lookup import EgyptianLookup

    rng = np.random.default_rng(seed)

    n_eg = 3
    n_eng = 5
    gemma_dim = 768
    glove_dim = 300

    eg_vocab = ["nTr", "Hr", "wsjr"]

    eng_gemma = rng.standard_normal((n_eng, gemma_dim)).astype(np.float32)
    eng_vocab = [f"word_{i}" for i in range(n_eng)]

    eg_gemma = eng_gemma[:n_eg].astype(np.float16)
    eng_glove = rng.standard_normal((n_eng, glove_dim)).astype(np.float32)
    eg_glove = eng_glove[:n_eg].astype(np.float16)

    np.savez_compressed(
        os.path.join(tmpdir, "egyptian_aligned_gemma_vectors.npz"),
        vectors=eg_gemma,
    )
    np.savez_compressed(
        os.path.join(tmpdir, "egyptian_aligned_vectors.npz"),
        vectors=eg_glove,
    )
    import pickle as _pkl
    with open(os.path.join(tmpdir, "egyptian_aligned_vocab.pkl"), "wb") as f:
        _pkl.dump(eg_vocab, f)
    np.savez_compressed(
        os.path.join(tmpdir, "english_gemma_whitened_768d.npz"),
        vocab=np.array(eng_vocab),
        vectors=eng_gemma,
    )

    return EgyptianLookup(
        gemma_vectors_path=os.path.join(tmpdir, "egyptian_aligned_gemma_vectors.npz"),
        glove_vectors_path=os.path.join(tmpdir, "egyptian_aligned_vectors.npz"),
        vocab_path=os.path.join(tmpdir, "egyptian_aligned_vocab.pkl"),
        gemma_english_path=os.path.join(tmpdir, "english_gemma_whitened_768d.npz"),
        glove_english_vectors=eng_glove,
        glove_english_vocab=eng_vocab,
    ), eg_vocab, eng_vocab


def test_egyptian_lookup_find_gemma():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, eg_vocab, eng_vocab = _build_tiny_lookup(tmpdir)
        results = lookup.find("word_0", top_k=3, space="gemma")
        assert len(results) == 3
        assert results[0][0] == "nTr"
        assert results[0][1] > 0.99


def test_egyptian_lookup_find_glove():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, eg_vocab, eng_vocab = _build_tiny_lookup(tmpdir)
        results = lookup.find("word_1", top_k=3, space="glove")
        assert len(results) == 3
        assert results[0][0] == "Hr"
        assert results[0][1] > 0.99


def test_egyptian_lookup_find_both():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        result = lookup.find_both("word_2", top_k=2)
        assert set(result.keys()) == {"gemma", "glove"}
        assert result["gemma"][0][0] == "wsjr"
        assert result["glove"][0][0] == "wsjr"


def test_egyptian_lookup_unknown_space_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        with pytest.raises(ValueError, match="space must be"):
            lookup.find("word_0", space="bert")


def test_egyptian_lookup_oov_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        assert lookup.find("not_a_real_word", space="gemma") == []


def test_egyptian_lookup_analogy():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        result = lookup.find_analogy("word_0", "word_1", "word_2", top_k=3, space="gemma")
        assert len(result) > 0


def test_egyptian_lookup_blend_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        assert lookup.find_blend({"unknown_xyz": 1.0}, space="gemma") == []
        assert lookup.find_blend({}, space="gemma") == []
