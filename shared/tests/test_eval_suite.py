import numpy as np

from shared.scripts.eval_suite import CAND_SIZE, K_CSLS, csls_topk


def _brute_csls(Q, C, pool, k):
    def norm(M):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    Qn, Cn, Pn = norm(Q), norm(C), norm(pool)
    S_pool = Pn @ Cn.T                       # p x m
    r_C = np.array([np.sort(S_pool[:, j])[-k:].mean() for j in range(C.shape[0])])
    S = Qn @ Cn.T
    return 2 * S - r_C[None, :]


def test_csls_matches_bruteforce():
    rng = np.random.RandomState(0)
    Q, C, pool = rng.randn(7, 16), rng.randn(40, 16), rng.randn(25, 16)
    idx = csls_topk(Q, C, pool, k_csls=5, topk=3)
    ref = np.argsort(-_brute_csls(Q, C, pool, 5), axis=1)[:, :3]
    assert np.array_equal(idx, ref)


def test_csls_demotes_hub():
    # Plain cosine prefers the hub; CSLS must flip the ranking.
    rng = np.random.RandomState(1)
    pool = rng.randn(50, 8) + np.array([5.0] + [0.0] * 7)   # pool crowds the hub
    hub = np.array([[5.0] + [0.0] * 7])
    niche = np.array([[0.0] * 7 + [5.0]])
    C = np.vstack([hub, niche])
    q = np.array([[3.0, 0, 0, 0, 0, 0, 0, 2.64]])           # cos_hub=.75 > cos_niche=.66

    def norm(M):
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    plain = (norm(q) @ norm(C).T)[0]
    assert plain[0] > plain[1]                # cosine prefers the hub...
    idx = csls_topk(q, C, pool, k_csls=10, topk=2)
    assert idx[0, 0] == 1                     # ...CSLS flips to the niche


def test_csls_chunking_consistent():
    rng = np.random.RandomState(2)
    Q, C, pool = rng.randn(30, 8), rng.randn(60, 8), rng.randn(20, 8)
    a = csls_topk(Q, C, pool, k_csls=4, topk=5, chunk=7)
    b = csls_topk(Q, C, pool, k_csls=4, topk=5, chunk=1000)
    assert np.array_equal(a, b)


def test_constants():
    assert CAND_SIZE == 50000 and K_CSLS == 10


from shared.scripts.eval_suite import (
    score_regime,
    stratify,
    synonym_set,
    val_top1_csls,
)


def test_stratify():
    strata = stratify(["king", "reed"], {"king", "water"})
    assert strata == ["interpolation", "zero_shot"]


def test_synonym_set_contains_wordnet_synonyms():
    s = synonym_set("king")
    assert "king" in s and "male monarch" in s


def test_synonym_set_unknown_word_is_identity():
    assert synonym_set("zzzznotaword") == {"zzzznotaword"}


def _identity_setup():
    # Candidates are unit basis vectors; query i equals candidate i exactly.
    C = np.eye(6, dtype=np.float32)
    vocab = ["king", "water", "reed", "house", "ruler", "sea"]
    return C, vocab


def test_score_regime_exact_and_syn():
    C, vocab = _identity_setup()
    Q = C[[0, 1]]                      # predict "king", "water" exactly
    r = score_regime(Q, ["ruler", "sea"], C, vocab, query_pool=C, ks=(1,))
    assert r["n"] == 2
    assert r["top1"]["exact"] == 0.0   # retrieved words differ from golds
    # "king" shares a synset with "ruler"? WordNet: king/ruler NOT synonyms;
    # but water/sea are not either — use a real pair: predict index of "king"
    r2 = score_regime(C[[0]], ["king"], C, vocab, query_pool=C, ks=(1,))
    assert r2["top1"]["exact"] == 100.0 and r2["top1"]["syn"] == 100.0


def test_score_regime_gold_oov_candidates():
    C, vocab = _identity_setup()
    r = score_regime(C[[0]], ["notinvocab"], C, vocab, query_pool=C, ks=(1,))
    assert r["n"] == 0 and r["gold_oov_candidates"] == 1


def test_val_top1_csls():
    C, vocab = _identity_setup()
    acc = val_top1_csls(C[[0, 2]], ["king", "reed"], C, vocab)
    assert acc == 100.0
