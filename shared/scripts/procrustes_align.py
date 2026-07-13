"""
Semi-orthogonal Procrustes maps: native fused 1536d -> whitened-Gemma 768d.

Parallel document-level plane alongside the production Ridge maps. Ridge
regularization contracts predictions toward the mean, differently per slot,
collapsing cross-slot cosines into a non-discriminative blob (journal
2026-07-09, Gate 2 FAIL). W = U V^T from the thin SVD of X^T Y has all
singular values equal to 1: no shrinkage by construction. The word-level
suite is intentionally NOT run on this map, and the word-level test split
is never touched.

See: docs/superpowers/specs/2026-07-12-procrustes-remap-design.md
"""
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.scripts.anchor_split import group_split  # noqa: E402

EXPECTED_SOURCE_DIM = 1536
EXPECTED_TARGET_DIM = 768

GEMMA_CACHE = _ROOT / "shared" / "models" / "english_gemma_whitened_768d.npz"

SLOTS = {
    "sumerian": {"surface_key": "sumerian"},
    "hittite": {"surface_key": "hittite"},
    "greek": {"surface_key": "greek"},
}

# Swadesh-207 English glosses (diagnostic only — reported, never a filter).
SWADESH_207 = frozenset("""
i you he we they this that here there who what where when how not all many
some few other one two three four five big long wide thick heavy small short
narrow thin woman man person child wife husband mother father animal fish
bird dog louse snake worm tree forest stick fruit seed leaf root bark flower
grass rope skin meat blood bone fat egg horn tail feather hair head ear eye
nose mouth tooth tongue fingernail foot leg knee hand wing belly guts neck
back breast heart liver drink eat bite suck spit vomit blow breathe laugh
see hear know think smell fear sleep live die kill fight hunt hit cut split
stab scratch dig swim fly walk come lie sit stand turn fall give hold
squeeze rub wash wipe pull push throw tie sew count say sing play float
flow freeze swell sun moon star water rain river lake sea salt stone sand
dust earth cloud fog sky wind snow ice smoke fire ash burn road mountain
red green yellow white black night day year warm cold full new old good
bad rotten dirty straight round sharp dull smooth wet dry correct near far
right left at in with and if because name
""".split())


def fit_semi_orthogonal(X, Y):
    """W = U V^T from the thin SVD of X^T Y.

    Maximises tr(W^T X^T Y) subject to W^T W = I; all singular values of W
    are 1, so the map applies zero shrinkage.
    """
    U, _, Vt = np.linalg.svd((X.T @ Y).astype(np.float64), full_matrices=False)
    return U @ Vt


def monosemous_pairs(anchors, surface_key):
    """Anchors whose surface has exactly one distinct gloss, (surface, gloss)
    deduped, original order preserved."""
    glosses = {}
    for a in anchors:
        glosses.setdefault(a[surface_key], set()).add(a["english"])
    seen, out = set(), []
    for a in anchors:
        key = (a[surface_key], a["english"])
        if len(glosses[a[surface_key]]) == 1 and key not in seen:
            seen.add(key)
            out.append(a)
    return out


def mean_cosine(X, Y, W):
    """Mean cosine between mapped rows X@W and their gold targets Y."""
    P = X @ W
    num = (P * Y).sum(axis=1)
    den = np.linalg.norm(P, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12
    return float((num / den).mean())


def isometry_rho(X, W, max_rows=1000, seed=42):
    """Spearman rho between pairwise cosines of X before vs after mapping.

    ~1.0 by construction for a (semi-)orthogonal map; materially lower
    means projection loss. Rows are subsampled deterministically to cap
    the pairwise matrix.
    """
    from scipy.stats import spearmanr

    if len(X) > max_rows:
        idx = np.random.RandomState(seed).choice(len(X), max_rows, replace=False)
        X = X[idx]

    def cos_upper(M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        C = Mn @ Mn.T
        return C[np.triu_indices(len(M), k=1)]

    return float(spearmanr(cos_upper(X), cos_upper(X @ W)).statistic)


def select_variant(variants):
    """Winning variant name by val cosine; 'full' wins ties."""
    return max(("full", "stable"), key=lambda n: variants[n]["val_cosine"])


def build_xy(anchors, surface_key, src_vocab, src_vectors, eng_vocab, eng_vectors):
    """Aligned X/Y rows for anchors present in both vocabularies
    (same convention as 09b's build_training_data)."""
    X, Y, valid = [], [], []
    for a in anchors:
        s, e = a[surface_key], a["english"]
        if s in src_vocab and e in eng_vocab:
            X.append(src_vectors[src_vocab[s]])
            Y.append(eng_vectors[eng_vocab[e]])
            valid.append(a)
    if not X:
        return np.array([]), np.array([]), []
    return np.array(X), np.array(Y), valid


def run_slot(slot, slot_root, gemma_cache):
    """Fit both variants, select on val, refit winner on train+val,
    project the full slot vocab, write artifacts. Returns the results dict."""
    surface_key = SLOTS[slot]["surface_key"]

    fused = np.load(str(slot_root / "models" / "fused_embeddings_1536d.npz"))
    src_vectors = fused["vectors"]
    src_vocab_list = [str(w) for w in fused["vocab"]]
    src_vocab = {w: i for i, w in enumerate(src_vocab_list)}
    assert src_vectors.shape[1] == EXPECTED_SOURCE_DIM, (
        f"{slot}: fused dim {src_vectors.shape[1]}, expected {EXPECTED_SOURCE_DIM}")

    gemma = np.load(str(gemma_cache))
    eng_vectors = gemma["vectors"]
    eng_vocab = {str(w): i for i, w in enumerate(gemma["vocab"])}
    assert eng_vectors.shape[1] == EXPECTED_TARGET_DIM, (
        f"gemma cache dim {eng_vectors.shape[1]}, expected {EXPECTED_TARGET_DIM}")

    with open(slot_root / "data" / "processed" / "english_anchors.json") as f:
        anchors = json.load(f)
    # Word-level test split intentionally unused — never touched.
    train_a, val_a, _ = group_split(anchors, surface_key=surface_key)

    X_val, Y_val, val_valid = build_xy(
        val_a, surface_key, src_vocab, src_vectors, eng_vocab, eng_vectors)
    if not len(X_val):
        raise SystemExit(f"{slot}: no valid val anchors — cannot select a variant")

    pools = {"full": train_a, "stable": monosemous_pairs(train_a, surface_key)}
    variants = {}
    for name, pool in pools.items():
        X, Y, valid = build_xy(
            pool, surface_key, src_vocab, src_vectors, eng_vocab, eng_vectors)
        if not len(X):
            raise SystemExit(f"{slot}: variant '{name}' has zero valid pairs")
        if name == "stable" and len(valid) < EXPECTED_TARGET_DIM:
            print(f"WARNING: {slot} stable variant has {len(valid)} pairs "
                  f"(< {EXPECTED_TARGET_DIM}) — rank-deficient subspace; "
                  "val selection decides.", file=sys.stderr)
        W = fit_semi_orthogonal(X, Y)
        variants[name] = {"n_pairs": len(valid),
                          "val_cosine": mean_cosine(X_val, Y_val, W)}
        print(f"{slot} {name:<7} n={len(valid):>6}  "
              f"val_cosine={variants[name]['val_cosine']:.4f}")

    chosen = select_variant(variants)
    fit_pool = train_a + val_a
    if chosen == "stable":
        fit_pool = monosemous_pairs(fit_pool, surface_key)
    X_fit, Y_fit, fit_valid = build_xy(
        fit_pool, surface_key, src_vocab, src_vectors, eng_vocab, eng_vectors)
    W = fit_semi_orthogonal(X_fit, Y_fit)

    results = {
        "slot": slot,
        "method": "semi-orthogonal procrustes (W = UVt of XtY), no scale",
        "source_dim": EXPECTED_SOURCE_DIM,
        "target_dim": EXPECTED_TARGET_DIM,
        "target_cache": gemma_cache.name,
        "split": {"method": "lemma-group", "seed": 42,
                  "selection": "val mean cosine; winner refit on train+val; "
                               "test split untouched"},
        "variants": variants,
        "chosen_variant": chosen,
        "n_fit_pairs": len(fit_valid),
        "isometry_rho_val": isometry_rho(X_val, W),
        "swadesh_diagnostic": {
            "n_fit_pairs_in_swadesh207":
                sum(1 for a in fit_valid if a["english"] in SWADESH_207)},
    }

    (slot_root / "final_output").mkdir(parents=True, exist_ok=True)
    (slot_root / "results").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(slot_root / "models" / "procrustes_W_gemma.npz", W=W)
    projected = (src_vectors @ W.astype(src_vectors.dtype)).astype(np.float32)
    np.savez_compressed(
        slot_root / "final_output" / f"{slot}_procrustes_gemma_vectors.npz",
        vectors=projected, vocab=np.array(src_vocab_list))
    with open(slot_root / "results" / "procrustes_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"{slot}: chose '{chosen}' (n_fit={len(fit_valid)}), "
          f"isometry rho={results['isometry_rho_val']:.4f}")
    return results


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Fit semi-orthogonal Procrustes maps (document-level plane).")
    p.add_argument("--slot", choices=sorted(SLOTS), default=None,
                   help="Single slot (default: all)")
    args = p.parse_args()
    for slot in ([args.slot] if args.slot else sorted(SLOTS)):
        run_slot(slot, _ROOT / "languages" / slot, GEMMA_CACHE)


if __name__ == "__main__":
    main()
