import numpy as np
import pytest

from shared.scripts.doc_eval import ETCSL_PATH
from shared.scripts.myth_study import (
    THEMES,
    HITTITE_MERGES,
    build_roster,
    doc_profile,
    load_corpora,
    percentile_in_null,
    rank_delta_report,
    rsa_permutation,
    theme_sim_matrix,
    upper_tri,
)


# ---------- pure-function tests (toy data) ----------


def test_doc_profile_loo():
    vecs = {"A1": np.array([1.0, 0.0]), "A2": np.array([0.0, 1.0]),
            "B1": np.array([1.0, 1.0])}
    members = {"A": ["A1", "A2"], "B": ["B1"]}
    prof = doc_profile("A1", vecs, members, ladder=("A", "B"))
    # theme A centroid excludes A1 -> [0,1], orthogonal to A1
    assert prof[0] == pytest.approx(0.0, abs=1e-6)
    assert prof[1] == pytest.approx(1 / np.sqrt(2), abs=1e-6)


def test_doc_profile_singleton_own_theme_raises():
    vecs = {"A1": np.array([1.0, 0.0]), "B1": np.array([1.0, 1.0])}
    members = {"A": ["A1"], "B": ["B1"]}
    with pytest.raises(ValueError):
        doc_profile("A1", vecs, members, ladder=("A", "B"))


def test_theme_sim_matrix_cosine():
    cents = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 2.0]),
             "C": np.array([1.0, 1.0])}
    M = theme_sim_matrix(cents, ladder=("A", "B", "C"))
    assert M.shape == (3, 3)
    assert M[0, 0] == pytest.approx(1.0)
    assert M[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert M[0, 2] == pytest.approx(1 / np.sqrt(2), abs=1e-5)
    assert np.allclose(M, M.T)


def _sym_from_upper(k, vals):
    M = np.eye(k)
    i, j = np.triu_indices(k, 1)
    M[i, j] = vals
    M[j, i] = vals
    return M


def test_rsa_perfect_match():
    rng = np.random.default_rng(0)
    vals = rng.permutation(np.linspace(0.1, 0.9, 10))  # k=5 -> 10 distinct uppers
    A = _sym_from_upper(5, vals)
    out = rsa_permutation(A, A.copy(), rng=np.random.default_rng(1))
    assert out["rho"] == pytest.approx(1.0)
    assert out["exhaustive"] is True
    assert out["n_perms"] == 120
    assert out["p"] <= 2 / 120


def test_rsa_anti_match():
    vals = np.linspace(0.1, 0.9, 6)  # k=4
    A = _sym_from_upper(4, vals)
    B = _sym_from_upper(4, vals[::-1])
    out = rsa_permutation(A, B, rng=np.random.default_rng(1))
    assert out["rho"] == pytest.approx(-1.0)


def test_rsa_unrelated_not_significant():
    rng = np.random.default_rng(3)
    A = _sym_from_upper(5, rng.uniform(size=10))
    B = _sym_from_upper(5, rng.uniform(size=10))
    out = rsa_permutation(A, B, rng=np.random.default_rng(4))
    assert abs(out["rho"]) < 1.0
    assert out["p"] > 0.05


def test_upper_tri():
    M = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 4.0, 1.0]])
    assert list(upper_tri(M)) == [2.0, 3.0, 4.0]


def test_rank_delta_identical_matrices():
    vals = np.linspace(0.1, 0.9, 6)
    M = _sym_from_upper(4, vals)
    rho, entries = rank_delta_report(M, M.copy(), ids=["a", "b", "c", "d"], top=3)
    assert rho == pytest.approx(1.0)
    assert all(e["delta"] == 0 for e in entries)


def test_rank_delta_detects_swap():
    ids = ["a", "b", "c", "d"]
    vn = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    va = vn.copy()
    va[0], va[5] = va[5], va[0]  # swap ranks of pairs (a,b) and (c,d)
    rho, entries = rank_delta_report(_sym_from_upper(4, vn), _sym_from_upper(4, va),
                                     ids=ids, top=2)
    top_pairs = {tuple(e["pair"]) for e in entries}
    assert top_pairs == {("a", "b"), ("c", "d")}
    assert all(abs(e["delta"]) == 5 for e in entries)
    assert rho < 1.0


def test_percentile_in_null():
    null = [0.1, 0.2, 0.3, 0.4]
    assert percentile_in_null(0.5, null) == 100.0
    assert percentile_in_null(0.0, null) == 0.0
    assert percentile_in_null(0.25, null) == 50.0
    assert percentile_in_null(0.2, null) == pytest.approx(37.5)  # midrank on tie


# ---------- roster invariants (real corpora) ----------

needs_data = pytest.mark.skipif(not ETCSL_PATH.exists(),
                                reason="corpora not available in this checkout")


@pytest.fixture(scope="module")
def roster_bundle():
    corpora = load_corpora()
    roster, notes, roster_tokens = build_roster(corpora)
    return corpora, roster, notes, roster_tokens


@needs_data
def test_roster_theme_keys_and_min_viability(roster_bundle):
    _, roster, _, _ = roster_bundle
    for slot in ("sumerian", "hittite", "greek"):
        assert set(roster[slot]) == set(THEMES)
        for theme in ("cosmogonic", "hymnic", "royal_control"):
            assert roster[slot][theme], f"{slot}/{theme} unexpectedly empty"


@needs_data
def test_roster_no_duplicate_doc_ids(roster_bundle):
    _, roster, _, _ = roster_bundle
    for slot in roster:
        ids = [e["doc_id"] for theme in THEMES for e in roster[slot][theme]]
        assert len(ids) == len(set(ids)), f"duplicate doc ids in {slot}"


@needs_data
def test_roster_pinned_docs_present(roster_bundle):
    _, roster, _, _ = roster_bundle
    sum_cosmo = {e["doc_id"] for e in roster["sumerian"]["cosmogonic"]}
    assert sum_cosmo == {"c141", "c174", "c111", "c112", "c113"}
    hit_cosmo = {e["doc_id"] for e in roster["hittite"]["cosmogonic"]}
    assert hit_cosmo == {"kumarbi", "ullikummi", "illuyanka"}
    gre_cosmo = {e["doc_id"] for e in roster["greek"]["cosmogonic"]}
    assert gre_cosmo == {"Hesiod (0020) - Theogony (001)",
                         "Hesiod (0020) - Works and Days (002)"}


@needs_data
def test_hittite_merged_union_token_counts(roster_bundle):
    corpora, roster, _, roster_tokens = roster_bundle
    hdocs = corpora["docs"]["hittite"]
    by_id = {e["doc_id"]: e for e in roster["hittite"]["cosmogonic"]}
    for merged_id, components in HITTITE_MERGES.items():
        expected = sum(len(hdocs[c]) for c in components)
        assert by_id[merged_id]["n_tokens"] == expected
        assert len(roster_tokens["hittite"][merged_id]) == expected
        assert list(by_id[merged_id]["components"]) == list(components)


@needs_data
def test_empty_themes_logged_with_reasons(roster_bundle):
    _, roster, notes, _ = roster_bundle
    assert roster["sumerian"]["magical"] == []
    assert notes["sumerian"]["magical"]["reason_empty"]
    assert roster["greek"]["magical"] == []
    assert "PGM" in notes["greek"]["magical"]["reason_empty"]
    assert roster["greek"]["wisdom"] == []
    assert notes["greek"]["wisdom"]["reason_empty"]
