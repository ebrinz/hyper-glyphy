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
