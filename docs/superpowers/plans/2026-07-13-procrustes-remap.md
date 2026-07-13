# Procrustes Remap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit a shrinkage-free semi-orthogonal map per doc-corpus slot (sumerian, hittite, greek) from native fused 1536d space into whitened-Gemma 768d, and remeasure Gate 2 (cross-language parallel retrieval) — the pre-registered route to reinstating Plane A.

**Architecture:** One shared fitter (`shared/scripts/procrustes_align.py`) mirroring `doc_eval.py`'s multi-slot pattern: per slot, fit `W = UVᵀ` from the thin SVD of `XᵀY` on two anchor variants (full / stable-monosemous), select on val cosine, refit the winner on train+val, project the full slot vocab. `doc_eval.py` gains a `--space ridge|procrustes` flag; the Gate 2 rerun writes to a new results file so the shipped Ridge FAIL stays on record.

**Tech Stack:** numpy (SVD), scipy.stats (spearman), existing `shared/scripts/anchor_split.group_split`, pytest.

**Spec:** `docs/superpowers/specs/2026-07-12-procrustes-remap-design.md`

## Global Constraints

- Ridge remains the production word-level map; this is a **parallel plane**. Do not modify 09/09b/10 scripts.
- The word-level suite is **NOT** run on the Procrustes map, and the word-level **test split is never touched** (only train/val).
- Slots: sumerian, hittite, greek only.
- Split: shared `group_split` with its default seed 42 — identical membership to the Ridge fit.
- No scale term in the map (cosine cancels per-slot global scale).
- Success criteria pre-registered, verbatim from `myth_study_plan.md` Gate 2: "parallel-text retrieval MRR ≥ 0.1, with the positive-control pair ranking in the top quartile of its pool (top ~205 of 820)" — applied to all three measured control pairs. **No threshold adjustment, no retry with tweaked variants on FAIL.**
- Artifact hygiene: `.npz` outputs land in gitignored dirs (`languages/*/models/`, `languages/*/final_output/*.npz`); only `results/*.json` and doc updates are committed.
- Branch: all work on `procrustes-remap`.

---

### Task 1: Core math, filters, and selection (pure functions)

**Files:**
- Create: `shared/scripts/procrustes_align.py` (module skeleton + pure functions)
- Create: `shared/tests/test_procrustes_align.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (Task 2 relies on these exact signatures):
  - `fit_semi_orthogonal(X: np.ndarray, Y: np.ndarray) -> np.ndarray` — W (d_src × d_tgt), `WᵀW = I`
  - `monosemous_pairs(anchors: list[dict], surface_key: str) -> list[dict]`
  - `mean_cosine(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> float`
  - `isometry_rho(X: np.ndarray, W: np.ndarray, max_rows=1000, seed=42) -> float`
  - `select_variant(variants: dict) -> str` — `"full"` or `"stable"`; full wins ties
  - Constants: `EXPECTED_SOURCE_DIM = 1536`, `EXPECTED_TARGET_DIM = 768`, `SWADESH_207: frozenset`, `SLOTS: dict`

- [ ] **Step 1: Write the failing tests**

Create `shared/tests/test_procrustes_align.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest shared/tests/test_procrustes_align.py -q`
Expected: FAIL at import — `ModuleNotFoundError` / `ImportError` (module doesn't exist yet).

- [ ] **Step 3: Write the module with the pure functions**

Create `shared/scripts/procrustes_align.py`:

```python
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
```

(`json`, `group_split`, `GEMMA_CACHE`, `SLOTS` are consumed by Task 2's driver in this same file.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest shared/tests/test_procrustes_align.py -q`
Expected: 9 passed. (An unused-import warning for `json`/`group_split` is acceptable — Task 2 uses them.)

- [ ] **Step 5: Commit**

```bash
git add shared/scripts/procrustes_align.py shared/tests/test_procrustes_align.py
git commit -m "feat(shared): procrustes core — semi-orthogonal fit, monosemy filter, isometry diagnostic"
```

---

### Task 2: Per-slot driver + CLI

**Files:**
- Modify: `shared/scripts/procrustes_align.py` (append driver below the pure functions)
- Test: `shared/tests/test_procrustes_align.py` (append integration test)

**Interfaces:**
- Consumes (from Task 1, same file): `fit_semi_orthogonal`, `monosemous_pairs`, `mean_cosine`, `isometry_rho`, `select_variant`, `SLOTS`, `GEMMA_CACHE`, `SWADESH_207`, `EXPECTED_SOURCE_DIM`, `EXPECTED_TARGET_DIM`, `group_split`.
- Produces:
  - `build_xy(anchors, surface_key, src_vocab, src_vectors, eng_vocab, eng_vectors) -> (X, Y, valid)` — same row-building convention as 09b's `build_training_data`
  - `run_slot(slot: str, slot_root: Path, gemma_cache: Path) -> dict` — fits, writes the three artifacts, returns the results dict
  - Artifacts Task 4/5 rely on:
    - `{slot_root}/final_output/{slot}_procrustes_gemma_vectors.npz` (keys: `vectors` float32, `vocab`)
    - `{slot_root}/models/procrustes_W_gemma.npz` (key: `W`)
    - `{slot_root}/results/procrustes_results.json`
  - CLI: `python3 shared/scripts/procrustes_align.py [--slot sumerian|hittite|greek]` (default: all three)

- [ ] **Step 1: Write the failing integration test**

Append to `shared/tests/test_procrustes_align.py`:

```python
import json

from shared.scripts.procrustes_align import (
    EXPECTED_SOURCE_DIM,
    EXPECTED_TARGET_DIM,
    build_xy,
    run_slot,
)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest shared/tests/test_procrustes_align.py -q`
Expected: 2 new tests FAIL with `ImportError: cannot import name 'build_xy'`; the 9 Task-1 tests still pass.

- [ ] **Step 3: Append driver to the module**

Append to `shared/scripts/procrustes_align.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest shared/tests/test_procrustes_align.py -q`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add shared/scripts/procrustes_align.py shared/tests/test_procrustes_align.py
git commit -m "feat(shared): procrustes driver — variant selection on val, full-vocab projection, CLI"
```

---

### Task 3: `--space` flag on doc_eval parallels

**Files:**
- Modify: `shared/scripts/doc_eval.py` (`run_parallels()` at ~line 209, `main()` at end of file)
- Test: `shared/tests/test_doc_eval.py` (append)

**Interfaces:**
- Consumes: Task 2's artifact naming `{slot}_procrustes_gemma_vectors.npz` (contains `vocab`, so `_load_space` reads it without a sidecar).
- Produces:
  - `parallel_space_npz(slot: str, space: str) -> Path` — module-level helper
  - `run_parallels(space: str = "ridge")` — default unchanged; `"procrustes"` reads the procrustes npz per slot and writes `shared/results/doc_eval_parallels_procrustes.json`
  - CLI: `python3 shared/scripts/doc_eval.py parallels --space procrustes`

- [ ] **Step 1: Write the failing test**

Append to `shared/tests/test_doc_eval.py`:

```python
def test_parallel_space_npz_resolution():
    from shared.scripts.doc_eval import parallel_space_npz

    ridge = parallel_space_npz("hittite", "ridge")
    assert ridge.name == "hittite_aligned_gemma_vectors.npz"
    assert ridge.parts[-3:-1] == ("hittite", "final_output")

    proc = parallel_space_npz("greek", "procrustes")
    assert proc.name == "greek_procrustes_gemma_vectors.npz"

    import pytest
    with pytest.raises(KeyError):
        parallel_space_npz("hittite", "bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest shared/tests/test_doc_eval.py -q`
Expected: 1 new FAIL (`ImportError: cannot import name 'parallel_space_npz'`); existing tests pass.

- [ ] **Step 3: Implement the flag**

In `shared/scripts/doc_eval.py`, add above `run_parallels()`:

```python
SPACE_NPZ = {"ridge": "{slot}_aligned_gemma_vectors.npz",
             "procrustes": "{slot}_procrustes_gemma_vectors.npz"}
SPACE_RESULTS = {"ridge": "doc_eval_parallels.json",
                 "procrustes": "doc_eval_parallels_procrustes.json"}


def parallel_space_npz(slot, space):
    """final_output npz for a slot under the given alignment space."""
    return (_ROOT / "languages" / slot / "final_output"
            / SPACE_NPZ[space].format(slot=slot))
```

Change `run_parallels()`'s signature and the two hardcoded paths:

```python
def run_parallels(space="ridge"):
    docs = _slot_documents()
    aligned = {}
    for slot in docs:
        path = parallel_space_npz(slot, space)
        if path.exists():
            aligned[slot] = _load_space(path)
```

(the rest of the function body is unchanged until the output block, which becomes:)

```python
    out = {"space": space, "pairs": results,
           "mrr": mean_reciprocal_rank(ranks) if ranks else None}
    res = _ROOT / "shared" / "results" / SPACE_RESULTS[space]
    res.parent.mkdir(exist_ok=True)
    with open(res, "w") as f:
        json.dump(out, f, indent=2)
    print(f"MRR: {out['mrr']}  Saved to: {res}")
```

And in `main()`:

```python
def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("benchmark", choices=("genre", "parallels"))
    p.add_argument("--space", choices=("ridge", "procrustes"), default="ridge",
                   help="Alignment space for parallels (genre ignores this)")
    args = p.parse_args()
    if args.benchmark == "genre":
        run_genre()
    else:
        run_parallels(space=args.space)
```

Note: adding `"space": space` to the ridge output dict changes that file's shape only if the ridge benchmark is re-run; the shipped `doc_eval_parallels.json` is not regenerated by this task.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest shared/tests/test_doc_eval.py shared/tests/test_procrustes_align.py -q`
Expected: all pass (existing doc_eval tests + 1 new + 11 procrustes).

- [ ] **Step 5: Commit**

```bash
git add shared/scripts/doc_eval.py shared/tests/test_doc_eval.py
git commit -m "feat(shared): doc_eval --space flag (ridge default unchanged; procrustes writes new results file)"
```

---

### Task 4: Fit the three real slots

**Files:**
- Create (run, not hand-written): `languages/{sumerian,hittite,greek}/results/procrustes_results.json` (committed); npz artifacts in gitignored dirs.

**Interfaces:**
- Consumes: Task 2's CLI. Requires on disk (all verified present): `languages/{slot}/models/fused_embeddings_1536d.npz`, `languages/{slot}/data/processed/english_anchors.json`, `shared/models/english_gemma_whitened_768d.npz`.
- Produces: the per-slot npz artifacts Task 5's Gate 2 rerun reads.

- [ ] **Step 1: Run the fitter for all three slots**

Run: `python3 shared/scripts/procrustes_align.py`
Expected: per slot, two variant lines (`full` / `stable` with n and val_cosine), a chosen-variant line with isometry rho. No SystemExit. A stable-variant rank warning is acceptable if printed.

- [ ] **Step 2: Verify artifacts and diagnostics**

Run:

```bash
python3 - <<'EOF'
import json
import numpy as np
for slot in ("sumerian", "hittite", "greek"):
    W = np.load(f"languages/{slot}/models/procrustes_W_gemma.npz")["W"]
    assert np.allclose(W.T @ W, np.eye(768), atol=1e-6), slot
    npz = np.load(f"languages/{slot}/final_output/{slot}_procrustes_gemma_vectors.npz")
    r = json.load(open(f"languages/{slot}/results/procrustes_results.json"))
    print(slot, npz["vectors"].shape, "chosen:", r["chosen_variant"],
          "rho:", round(r["isometry_rho_val"], 4),
          "swadesh:", r["swadesh_diagnostic"]["n_fit_pairs_in_swadesh207"])
EOF
```

Expected: three lines; vector counts match fused vocab sizes (sumerian 35508, hittite 31412, greek 227129), each 768-wide. Record the isometry rho values — if any is materially below 1 (< 0.95), that is projection loss: journal it in Task 5, do NOT refit.

- [ ] **Step 3: Confirm npz artifacts are gitignored, commit the results JSONs**

Run: `git status --short` — expect ONLY the three `results/procrustes_results.json` files (any npz appearing means a gitignore gap: stop and fix `.gitignore` first).

```bash
git add languages/sumerian/results/procrustes_results.json \
        languages/hittite/results/procrustes_results.json \
        languages/greek/results/procrustes_results.json
git commit -m "feat: procrustes maps fitted for sumerian/hittite/greek (variant selection + isometry diagnostics)"
```

---

### Task 5: Gate 2 rerun, verdict, and docs

**Files:**
- Create (run): `shared/results/doc_eval_parallels_procrustes.json` (committed)
- Modify: `docs/EXPERIMENT_JOURNAL.md` (new entry at top of "Recent findings")
- Modify: `docs/myth_study_plan.md` (§ "Current status of Plane A" ~line 73, § Consequence ~line 144, §7 "Procrustes remap" ~line 156)

**Interfaces:**
- Consumes: Task 3's CLI + Task 4's artifacts.
- Produces: the measured Gate 2 verdict; this task ends the project.

- [ ] **Step 1: Run Gate 2 on the procrustes space**

Run: `python3 shared/scripts/doc_eval.py parallels --space procrustes`
Expected: three pair lines (`kumarbi-theogony`, `illuyanka-typhon`, `ullikummi-typhon`) with `rank N/820`, then `MRR: ...  Saved to: .../doc_eval_parallels_procrustes.json`. Sumerian's space loads too (it's in the pool machinery) — pairs are Hittite→Greek as shipped.

- [ ] **Step 2: Apply the pre-registered verdict**

PASS requires BOTH: MRR ≥ 0.1 AND all three control ranks ≤ 205.
This is mechanical — no interpretation, no threshold adjustment, no refitting with tweaked variants. Ridge comparison numbers for the journal: MRR 0.0013; ranks 731, 781, 788 of 820.

- [ ] **Step 3: Write the journal entry**

Add at the top of "Recent findings" in `docs/EXPERIMENT_JOURNAL.md` (fill measured numbers; pick the matching verdict clause):

```markdown
- **2026-07-XX — Procrustes remap measured: Gate 2 [PASS/FAIL] on the semi-orthogonal plane.**
  Per-slot semi-orthogonal maps (W = UVt of XtY, 1536→768, no scale; variants
  full/stable-monosemous selected on val cosine — chosen: sumerian [VARIANT],
  hittite [VARIANT], greek [VARIANT]; isometry rho [VALUES]) as a parallel
  document-level plane alongside the production Ridge maps. Parallel retrieval
  (Hittite→Greek, pool 820): kumarbi [RANK], illuyanka [RANK], ullikummi [RANK],
  MRR [VALUE] (Ridge: 731/781/788, MRR 0.0013). Verdict: [PASS — Plane A
  reinstated for future cross-language cosine claims / FAIL — Plane A stays
  no-go; artifacts retained for future levers (stronger anchors)]. Spec:
  docs/superpowers/specs/2026-07-12-procrustes-remap-design.md.
```

- [ ] **Step 4: Update `docs/myth_study_plan.md`**

In all three Plane-A-status locations (~lines 73, 144, 156): replace the prospective "pending a map quality improvement (e.g., Procrustes remap…)" framing with the measured outcome — on PASS, Plane A cross-language cosine is reinstated on the procrustes space (cite `doc_eval_parallels_procrustes.json`); on FAIL, record that the Procrustes remap was measured on 2026-07-XX and did not clear Gate 2, so Plane A remains no-go and the remaining lever is a stronger anchor set. Keep edits surgical — status wording only.

- [ ] **Step 5: Run the full shared test suite, then commit**

Run: `python3 -m pytest shared/tests/ -q`
Expected: all pass.

```bash
git add shared/results/doc_eval_parallels_procrustes.json \
        docs/EXPERIMENT_JOURNAL.md docs/myth_study_plan.md
git commit -m "feat(shared): Gate 2 remeasured on procrustes plane — [PASS/FAIL] (journal + myth-plan status)"
```

---

## Out of scope

- Making new Plane A myth-study claims on PASS (follow-on project).
- Word-level suite on the procrustes map; any change to 09/09b/10 or the production Ridge artifacts.
- Akkadian/Egyptian slots (await per-text segmentation); README changes unless the verdict changes shipped claims (it doesn't — the README's Gate 2 line describes the Ridge measurement, which stands).
