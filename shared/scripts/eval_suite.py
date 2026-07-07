"""
Stratified CSLS evaluation suite over saved alignment artifacts.

See: docs/superpowers/specs/2026-07-06-eval-redesign-design.md
"""
import numpy as np

CAND_SIZE = 50000
K_CSLS = 10


def _normalize(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def csls_topk(Q, C, query_pool, k_csls=K_CSLS, topk=10, chunk=2048):
    """Top-k candidate indices per query under CSLS retrieval.

    CSLS(x, y) = 2 cos(x, y) - r(y); the query-side hubness term is constant
    per query and cannot change ranking. r(y) = mean cosine of candidate y to
    its k_csls nearest neighbors in query_pool.
    """
    Qn, Cn, Pn = _normalize(np.asarray(Q, dtype=np.float32)), _normalize(
        np.asarray(C, dtype=np.float32)
    ), _normalize(np.asarray(query_pool, dtype=np.float32))

    m = Cn.shape[0]
    r_C = np.empty(m, dtype=np.float32)
    for j0 in range(0, m, chunk):
        S_pool = Pn @ Cn[j0 : j0 + chunk].T          # p x chunk
        k = min(k_csls, S_pool.shape[0])
        top = np.partition(S_pool, -k, axis=0)[-k:]
        r_C[j0 : j0 + chunk] = top.mean(axis=0)

    out = np.empty((Qn.shape[0], topk), dtype=np.int64)
    for i0 in range(0, Qn.shape[0], chunk):
        scores = 2 * (Qn[i0 : i0 + chunk] @ Cn.T) - r_C[None, :]
        part = np.argpartition(-scores, topk - 1, axis=1)[:, :topk]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        out[i0 : i0 + chunk] = np.take_along_axis(part, order, axis=1)
    return out


_SYN_CACHE = {}


def synonym_set(word):
    """The word plus all WordNet lemma names sharing any synset with it."""
    if word in _SYN_CACHE:
        return _SYN_CACHE[word]
    out = {word}
    try:
        from nltk.corpus import wordnet as wn

        for syn in wn.synsets(word):
            out.update(l.name().lower().replace("_", " ") for l in syn.lemmas())
    except Exception:
        pass  # WordNet unavailable -> exact-only degrade, never a crash
    _SYN_CACHE[word] = out
    return out


def stratify(test_golds, trainval_golds):
    tv = set(trainval_golds)
    return ["interpolation" if g in tv else "zero_shot" for g in test_golds]


def score_regime(Q, golds, cand_vectors, cand_vocab, query_pool, ks=(1, 5, 10)):
    """Score one regime: CSLS retrieval over the restricted candidates.

    Items whose gold is outside cand_vocab are excluded and counted.
    """
    cand_set = set(cand_vocab)
    keep = [i for i, g in enumerate(golds) if g in cand_set]
    oov = len(golds) - len(keep)
    result = {"n": len(keep), "gold_oov_candidates": oov}
    if not keep:
        result.update({f"top{k}": {"exact": 0.0, "syn": 0.0} for k in ks})
        return result

    Qk = np.asarray(Q)[keep]
    gk = [golds[i] for i in keep]
    idx = csls_topk(Qk, cand_vectors, query_pool, topk=max(ks))
    for k in ks:
        exact = syn = 0
        for i, g in enumerate(gk):
            words = [cand_vocab[j] for j in idx[i, :k]]
            if g in words:
                exact += 1
            gsyn = synonym_set(g)
            if any(w in gsyn for w in words):
                syn += 1
        result[f"top{k}"] = {
            "exact": exact / len(gk) * 100,
            "syn": syn / len(gk) * 100,
        }
    return result


def val_top1_csls(Y_pred_val, val_golds, cand_vectors, cand_vocab):
    """Alpha-selection scorer: exact CSLS top-1 on the restricted candidates.

    Query pool = the val predictions themselves.
    """
    r = score_regime(
        Y_pred_val, val_golds, cand_vectors, cand_vocab, query_pool=Y_pred_val, ks=(1,)
    )
    return r["top1"]["exact"]


def score_suite(artifacts, cand_vectors, cand_vocab):
    """Full suite from an artifact bundle (see load_artifacts, Task 3)."""
    pool = np.vstack([artifacts["Q_train"], artifacts["Q_val"], artifacts["Q_test"]])
    strata = artifacts["meta"]["test_strata"]
    test_golds = [a["gold"] for a in artifacts["meta"]["test"]]
    Q_test = artifacts["Q_test"]

    def subset(name):
        sel = [i for i, s in enumerate(strata) if s == name]
        return Q_test[sel], [test_golds[i] for i in sel]

    suite = {
        "dictionary_in_sample": score_regime(
            artifacts["Q_train"],
            [a["gold"] for a in artifacts["meta"]["train_sample"]],
            cand_vectors, cand_vocab, pool,
        ),
    }
    for name in ("interpolation", "zero_shot"):
        Qs, gs = subset(name)
        suite[name] = score_regime(Qs, gs, cand_vectors, cand_vocab, pool)
    suite["test_combined"] = score_regime(Q_test, test_golds, cand_vectors, cand_vocab, pool)
    return suite
