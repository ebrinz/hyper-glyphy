import json

import numpy as np

from shared.scripts.procrustes_align import (
    EXPECTED_SOURCE_DIM,
    EXPECTED_TARGET_DIM,
    SWADESH_207,
    build_xy,
    fit_semi_orthogonal,
    isometry_rho,
    mean_cosine,
    monosemous_pairs,
    run_slot,
    select_variant,
)


def _random_semi_orthogonal(d_in, d_out, rng):
    U, _, Vt = np.linalg.svd(rng.randn(d_in, d_out), full_matrices=False)
    return U @ Vt


def test_fit_returns_orthonormal_columns():
    rng = np.random.RandomState(0)
    X, Y = rng.randn(500, 40), rng.randn(500, 12)
    W = fit_semi_orthogonal(X, Y)
    assert W.shape == (40, 12)
    assert np.allclose(W.T @ W, np.eye(12), atol=1e-8)


def test_fit_recovers_planted_map():
    # Exact-whiten X so X.T @ X = n*I; then X.T @ Y = n*W_true and the
    # trace-maximising UVt equals the planted map exactly.
    rng = np.random.RandomState(2)
    X = rng.randn(800, 40)
    C = X.T @ X / len(X)
    evals, evecs = np.linalg.eigh(C)
    X = X @ evecs @ np.diag(evals**-0.5) @ evecs.T
    W_true = _random_semi_orthogonal(40, 12, np.random.RandomState(1))
    W = fit_semi_orthogonal(X, X @ W_true)
    assert np.allclose(W, W_true, atol=1e-8)


def test_fit_has_no_shrinkage():
    # Every singular value of W is 1 — the property Ridge lacks.
    rng = np.random.RandomState(3)
    W = fit_semi_orthogonal(rng.randn(300, 30), rng.randn(300, 10))
    s = np.linalg.svd(W, compute_uv=False)
    assert np.allclose(s, 1.0, atol=1e-8)


def test_monosemous_filter_drops_polysemes_and_dedupes():
    anchors = [
        {"x": "a", "english": "king"},
        {"x": "a", "english": "king"},     # duplicate pair -> one row
        {"x": "b", "english": "king"},
        {"x": "b", "english": "lord"},     # polysemous surface -> dropped
        {"x": "c", "english": "water"},
    ]
    kept = monosemous_pairs(anchors, "x")
    assert [(a["x"], a["english"]) for a in kept] == [("a", "king"), ("c", "water")]


def test_mean_cosine_is_one_for_exact_map():
    rng = np.random.RandomState(4)
    W = _random_semi_orthogonal(20, 8, rng)
    X = rng.randn(50, 20)
    assert mean_cosine(X, X @ W, W) > 0.999999


def test_isometry_rho_is_one_for_square_rotation():
    # A square orthogonal map preserves all pairwise cosines exactly.
    rng = np.random.RandomState(5)
    Q, _ = np.linalg.qr(rng.randn(16, 16))
    X = rng.randn(60, 16)
    assert isometry_rho(X, Q) > 0.999999


def test_isometry_rho_subsamples_deterministically():
    rng = np.random.RandomState(6)
    Q, _ = np.linalg.qr(rng.randn(10, 10))
    X = rng.randn(50, 10)
    assert isometry_rho(X, Q, max_rows=20) == isometry_rho(X, Q, max_rows=20)


def test_select_variant_prefers_higher_val_cosine_full_wins_ties():
    assert select_variant({"full": {"val_cosine": 0.4},
                           "stable": {"val_cosine": 0.5}}) == "stable"
    assert select_variant({"full": {"val_cosine": 0.5},
                           "stable": {"val_cosine": 0.5}}) == "full"


def test_swadesh_list_is_plausible():
    assert 180 <= len(SWADESH_207) <= 210
    assert {"water", "fire", "hand", "name", "night"} <= SWADESH_207


def test_build_xy_skips_oov():
    anchors = [{"x": "a", "english": "king"}, {"x": "zz", "english": "king"},
               {"x": "b", "english": "zz"}]
    src_vocab = {"a": 0, "b": 1}
    eng_vocab = {"king": 0}
    X, Y, valid = build_xy(anchors, "x", src_vocab, np.eye(2), eng_vocab, np.eye(1))
    assert len(valid) == 1 and valid[0]["x"] == "a"
    assert X.shape == (1, 2) and Y.shape == (1, 1)


def _synthetic_slot(tmp_path, n_anchors=40):
    """Minimal on-disk slot: fused npz, anchors json, gemma cache."""
    rng = np.random.RandomState(7)
    surfaces = [f"w{i}" for i in range(n_anchors)]
    glosses = [f"g{i}" for i in range(n_anchors)]
    slot_root = tmp_path / "toy"
    (slot_root / "models").mkdir(parents=True)
    (slot_root / "data" / "processed").mkdir(parents=True)
    np.savez(slot_root / "models" / "fused_embeddings_1536d.npz",
             vectors=rng.randn(n_anchors, EXPECTED_SOURCE_DIM).astype(np.float32),
             vocab=np.array(surfaces))
    gemma_cache = tmp_path / "english_gemma_whitened_768d.npz"
    np.savez(gemma_cache,
             vectors=rng.randn(n_anchors, EXPECTED_TARGET_DIM).astype(np.float32),
             vocab=np.array(glosses))
    anchors = [{"toy": s, "english": g} for s, g in zip(surfaces, glosses)]
    with open(slot_root / "data" / "processed" / "english_anchors.json", "w") as f:
        json.dump(anchors, f)
    return slot_root, gemma_cache


def test_run_slot_end_to_end(tmp_path, monkeypatch):
    import shared.scripts.procrustes_align as pa
    monkeypatch.setitem(pa.SLOTS, "toy", {"surface_key": "toy"})
    slot_root, gemma_cache = _synthetic_slot(tmp_path)

    results = run_slot("toy", slot_root, gemma_cache)

    out_npz = slot_root / "final_output" / "toy_procrustes_gemma_vectors.npz"
    w_npz = slot_root / "models" / "procrustes_W_gemma.npz"
    res_json = slot_root / "results" / "procrustes_results.json"
    assert out_npz.exists() and w_npz.exists() and res_json.exists()

    W = np.load(w_npz)["W"]
    assert W.shape == (EXPECTED_SOURCE_DIM, EXPECTED_TARGET_DIM)
    assert np.allclose(W.T @ W, np.eye(EXPECTED_TARGET_DIM), atol=1e-6)

    proj = np.load(out_npz)
    assert proj["vectors"].shape == (40, EXPECTED_TARGET_DIM)
    assert proj["vectors"].dtype == np.float32
    assert list(proj["vocab"]) == [f"w{i}" for i in range(40)]

    for key in ("slot", "variants", "chosen_variant", "isometry_rho_val",
                "swadesh_diagnostic", "n_fit_pairs"):
        assert key in results
    assert results["chosen_variant"] in ("full", "stable")
    assert set(results["variants"]) == {"full", "stable"}
    for v in results["variants"].values():
        assert set(v) >= {"n_pairs", "val_cosine"}
    assert json.load(open(res_json)) == results
