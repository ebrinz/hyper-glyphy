"""
Gemma accuracy experiments for Egyptian alignment.

Tests approaches to improve Gemma top-1 beyond the current 33.50%:
1. Finer Ridge alpha sweep
2. PCA target reduction (768d -> smaller) then project back
3. Procrustes (orthogonal mapping)
4. Partial Least Squares regression
"""
import json
import time
import warnings

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")

FUSED_PATH = "languages/egyptian/models/fused_embeddings_1536d.npz"
GEMMA_PATH = "shared/models/english_gemma_whitened_768d.npz"
ANCHOR_PATH = "languages/egyptian/data/processed/english_anchors_normalized.json"


def load_data():
    print("\n--- Loading data ---")

    print("  [1/3] Loading fused Egyptian vectors...")
    fused = np.load(FUSED_PATH)
    eg_vecs = fused["vectors"]
    eg_vocab = {str(w): i for i, w in enumerate(fused["vocab"])}
    print(f"         {eg_vecs.shape[0]} words x {eg_vecs.shape[1]}d")

    print("  [2/3] Loading whitened Gemma vectors...")
    gemma = np.load(GEMMA_PATH)
    eng_vecs = gemma["vectors"].astype(np.float32)
    eng_vocab_list = [str(w) for w in gemma["vocab"]]
    eng_vocab = {w: i for i, w in enumerate(eng_vocab_list)}
    print(f"         {eng_vecs.shape[0]} words x {eng_vecs.shape[1]}d")

    print("  [3/3] Building anchor training pairs...")
    anchors = json.load(open(ANCHOR_PATH))

    X_list, Y_list, eng_words = [], [], []
    for a in anchors:
        e = a.get("egyptian_raw", a["egyptian"])
        eng = a["english"]
        if e in eg_vocab and eng in eng_vocab:
            X_list.append(eg_vecs[eg_vocab[e]])
            Y_list.append(eng_vecs[eng_vocab[eng]])
            eng_words.append(eng)

    X, Y = np.array(X_list), np.array(Y_list)
    print(f"         {len(X)} valid anchor pairs from {len(anchors)} total")
    return X, Y, eng_words, eng_vecs, eng_vocab_list


def eval_topk(Yp, test_eng, eng_norm, eng_vocab_list, show_progress=False):
    norms = np.linalg.norm(Yp, axis=1, keepdims=True)
    norms[norms == 0] = 1
    Yp_n = Yp / norms

    if show_progress:
        print("         Computing cosine distances against 400k English vocab...")
    dists = cdist(Yp_n, eng_norm, metric="cosine")

    c = {1: 0, 5: 0, 10: 0}
    iterator = enumerate(test_eng)
    if show_progress:
        iterator = tqdm(iterator, total=len(test_eng), desc="         Evaluating", unit="query")

    for i, eng in iterator:
        nn = np.argsort(dists[i])[:10]
        words = [eng_vocab_list[j] for j in nn]
        if eng == words[0]:
            c[1] += 1
        if eng in words[:5]:
            c[5] += 1
        if eng in words[:10]:
            c[10] += 1
    n = len(test_eng)
    return {k: round(v / n * 100, 2) for k, v in c.items()}


def main():
    t0 = time.time()

    print("=" * 70)
    print("  GEMMA ACCURACY EXPERIMENTS -- Egyptian Alignment")
    print("  Goal: improve Gemma top-1 beyond current 33.50%")
    print("=" * 70)

    X, Y, eng_words, eng_vecs, eng_vocab_list = load_data()

    print(f"\n--- Preprocessing ---")
    print(f"  Normalizing {len(eng_vecs):,} English vectors for cosine eval...")
    g_norms = np.linalg.norm(eng_vecs, axis=1, keepdims=True)
    g_norms[g_norms == 0] = 1
    eng_norm = eng_vecs / g_norms
    print(f"  done ({time.time() - t0:.1f}s elapsed)")

    Xtr, Xte, Ytr, Yte, etr, ete = train_test_split(
        X, Y, eng_words, test_size=0.2, random_state=42
    )
    print(f"\n--- Data split ---")
    print(f"  Train: {len(Xtr):,} anchors")
    print(f"  Test:  {len(Xte):,} anchors")
    print(f"  Input dim:  {X.shape[1]}d (fused Egyptian)")
    print(f"  Target dim: {Y.shape[1]}d (whitened Gemma)")
    print(f"  Samples-per-output-dim: {len(Xtr)/Y.shape[1]:.1f}")
    print(f"  (Sumerian has 6,867 train -> ratio 8.9; we have {len(Xtr)/768:.1f})")

    print(f"\n--- Baseline ---")
    print(f"  Current production: Ridge alpha=0.1 -> 33.50% top-1")
    print(f"  Target (Sumerian-like lift): ~47-50% top-1")

    # =========================================================================
    # 1. Ridge alpha sweep
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  [1/4] RIDGE ALPHA SWEEP")
    print(f"  Hypothesis: finer alpha tuning may squeeze out 1-2pp")
    print(f"{'=' * 70}")

    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0]
    best_ridge = (0, 0)
    for alpha in tqdm(alphas, desc="  Ridge sweep", unit="alpha"):
        t1 = time.time()
        m = Ridge(alpha=alpha).fit(Xtr, Ytr)
        r = eval_topk(m.predict(Xte), ete, eng_norm, eng_vocab_list)
        dt = time.time() - t1
        marker = " <-- CURRENT" if alpha == 0.1 else ""
        best_marker = ""
        if r[1] > best_ridge[1]:
            best_ridge = (alpha, r[1])
            best_marker = " ** BEST"
        tqdm.write(f"    alpha={alpha:<6}  top1={r[1]:>6.2f}%  top5={r[5]:>6.2f}%  top10={r[10]:>6.2f}%  ({dt:.1f}s){marker}{best_marker}")

    print(f"\n  >>> Best Ridge: alpha={best_ridge[0]} -> {best_ridge[1]:.2f}% top-1")

    # =========================================================================
    # 2. PCA target reduction
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  [2/4] PCA TARGET REDUCTION + RIDGE")
    print(f"  Hypothesis: compress 768d Gemma target to fewer dims so Ridge")
    print(f"  has more samples-per-dim; inverse_transform lifts back to 768d")
    print(f"{'=' * 70}")

    pca_configs = [64, 128, 256, 300, 512]
    best_pca = ("", 0)
    for n_comp in tqdm(pca_configs, desc="  PCA dims", unit="dim"):
        tqdm.write(f"\n    --- PCA to {n_comp} components ---")
        t1 = time.time()
        pca = PCA(n_components=n_comp).fit(eng_vecs[:50000])
        variance_kept = pca.explained_variance_ratio_.sum() * 100
        tqdm.write(f"    Variance retained: {variance_kept:.1f}%")
        tqdm.write(f"    Effective samples-per-dim: {len(Xtr)}/{n_comp} = {len(Xtr)/n_comp:.1f}")
        Ytr_r = pca.transform(Ytr)

        for alpha in [0.1, 1.0, 10.0]:
            m = Ridge(alpha=alpha).fit(Xtr, Ytr_r)
            Yp_r = m.predict(Xte)
            Yp_full = pca.inverse_transform(Yp_r)
            r = eval_topk(Yp_full, ete, eng_norm, eng_vocab_list)
            dt = time.time() - t1
            label = f"PCA-{n_comp} alpha={alpha}"
            best_marker = ""
            if r[1] > best_pca[1]:
                best_pca = (label, r[1])
                best_marker = " ** BEST"
            tqdm.write(f"      alpha={alpha:<6}  top1={r[1]:>6.2f}%  top5={r[5]:>6.2f}%  top10={r[10]:>6.2f}%  ({dt:.1f}s){best_marker}")

    print(f"\n  >>> Best PCA: {best_pca[0]} -> {best_pca[1]:.2f}% top-1")

    # =========================================================================
    # 3. Procrustes
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  [3/4] PROCRUSTES (ORTHOGONAL ROTATION)")
    print(f"  Hypothesis: distance-preserving rotation has fewer free params,")
    print(f"  may generalize better with limited training data")
    print(f"{'=' * 70}")

    t1 = time.time()
    print(f"  Reducing input 1536d -> 768d via PCA...")
    pca_in = PCA(n_components=768).fit(Xtr)
    Xtr_768 = pca_in.transform(Xtr)
    Xte_768 = pca_in.transform(Xte)
    print(f"  Computing optimal orthogonal matrix R (768x768)...")
    R, scale = orthogonal_procrustes(Xtr_768, Ytr)
    print(f"  Scale factor: {scale:.4f}")
    Yp_proc = Xte_768 @ R * scale
    r_proc = eval_topk(Yp_proc, ete, eng_norm, eng_vocab_list, show_progress=True)
    dt = time.time() - t1
    print(f"\n  >>> Procrustes: top1={r_proc[1]:>6.2f}%  top5={r_proc[5]:>6.2f}%  top10={r_proc[10]:>6.2f}%  ({dt:.1f}s)")

    # =========================================================================
    # 4. PLS
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  [4/4] PARTIAL LEAST SQUARES (PLS)")
    print(f"  Hypothesis: PLS finds latent components maximizing X-Y covariance")
    print(f"  -- designed specifically for high-dim Y with few samples")
    print(f"{'=' * 70}")

    pls_configs = [50, 100, 200, 300]
    best_pls = ("", 0)
    for n_comp in tqdm(pls_configs, desc="  PLS components", unit="config"):
        t1 = time.time()
        tqdm.write(f"\n    Fitting PLS with {n_comp} components...")
        try:
            pls = PLSRegression(n_components=n_comp, max_iter=1000)
            pls.fit(Xtr, Ytr)
            tqdm.write(f"    Predicting + evaluating...")
            Yp = pls.predict(Xte)
            r = eval_topk(Yp, ete, eng_norm, eng_vocab_list)
            dt = time.time() - t1
            label = f"PLS-{n_comp}"
            best_marker = ""
            if r[1] > best_pls[1]:
                best_pls = (label, r[1])
                best_marker = " ** BEST"
            tqdm.write(f"    n_comp={n_comp:<4}  top1={r[1]:>6.2f}%  top5={r[5]:>6.2f}%  top10={r[10]:>6.2f}%  ({dt:.1f}s){best_marker}")
        except Exception as ex:
            tqdm.write(f"    n_comp={n_comp:<4}  FAILED: {ex}")

    print(f"\n  >>> Best PLS: {best_pls[0]} -> {best_pls[1]:.2f}% top-1")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"  Total experiment time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"")
    print(f"  {'Method':<35s} {'Top-1':>7s}  {'Delta vs baseline':>18s}")
    print(f"  {'-' * 62}")
    print(f"  {'Baseline (Ridge alpha=0.1)':<35s} {'33.50%':>7s}  {'---':>18s}")
    print(f"  {f'Best Ridge (alpha={best_ridge[0]})':<35s} {best_ridge[1]:>6.2f}%  {best_ridge[1]-33.50:>+17.2f}pp")
    print(f"  {f'Best PCA ({best_pca[0]})':<35s} {best_pca[1]:>6.2f}%  {best_pca[1]-33.50:>+17.2f}pp")
    print(f"  {'Procrustes':<35s} {r_proc[1]:>6.2f}%  {r_proc[1]-33.50:>+17.2f}pp")
    print(f"  {f'Best PLS ({best_pls[0]})':<35s} {best_pls[1]:>6.2f}%  {best_pls[1]-33.50:>+17.2f}pp")
    print(f"")
    print(f"  Reference:")
    print(f"    Sumerian Gemma top-1:  52.13% (6,867 train samples, alpha=100)")
    print(f"    Egyptian train samples: {len(Xtr)} (ratio {len(Xtr)/768:.1f} per output dim)")
    print(f"    Sumerian train samples: 6,867 (ratio {6867/768:.1f} per output dim)")
    print(f"")
    print(f"{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
