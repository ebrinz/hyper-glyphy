import numpy as np

from shared.scripts.procrustes_align import (
    SWADESH_207,
    fit_semi_orthogonal,
    isometry_rho,
    mean_cosine,
    monosemous_pairs,
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
