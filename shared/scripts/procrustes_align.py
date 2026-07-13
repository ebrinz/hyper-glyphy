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
