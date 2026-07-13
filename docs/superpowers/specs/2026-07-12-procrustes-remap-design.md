# Procrustes Remap — Design

**Date:** 2026-07-12
**Status:** Approved
**Goal:** Fit a shrinkage-free semi-orthogonal map per slot (native fused space → whitened-Gemma English space) and remeasure Gate 2 (cross-language parallel retrieval), the pre-registered route to reinstating Plane A cross-language cosine claims.

## Background

Gate 2 (Hittite → Greek parallel retrieval) FAILED on the Ridge-aligned spaces:
MRR 0.0013; all three positive controls (Kumarbi, Illuyanka, Ullikummi →
Theogony) in the bottom 11% of the 820-document pool. Diagnosis
(journal 2026-07-09): cross-language cosines form a non-discriminative blob
(mean 0.252, std 0.024) — Ridge regularization contracts predictions toward
the mean, and each slot contracts differently, so cross-slot absolute cosine
carries no signal. Within-slot geometry survives (translation delta
ρ 0.81–0.92) because rankings tolerate uniform shrinkage.

`myth_study_plan.md` §7 pre-registers the remedy: a Procrustes remap of
native embeddings is "the most direct route to reinstating Plane A."
A semi-orthogonal map has all singular values equal to 1 — zero shrinkage
by construction.

## Scope

- **Parallel plane, not a replacement.** Ridge remains the production
  word-level map everywhere. The Procrustes map is a second exported
  artifact used only for cross-language document-level work.
- **Slots:** sumerian, hittite, greek — the three with per-text corpora
  that document-level evals consume. Akkadian/Egyptian wait on per-text
  segmentation.
- **Word-level suite is NOT run on this map.** Its word-level top-1 is
  irrelevant to the Gate 2 claim; running it invites comparison-shopping.
- Work on a `procrustes-remap` feature branch.

## Method

For each slot, with anchor pairs X (n×1536, fused native) and Y (n×768,
whitened Gemma of the gold gloss):

    W = U Vᵀ  from thin SVD of XᵀY,  W ∈ R^{1536×768},  WᵀW = I

No scale term (cosine cancels per-slot global scale). Inner products are
preserved exactly on the 768-d subspace the anchors select.

**Rejected alternatives** (recorded for the journal):
- *PCA 1536→768 + square rotation* (the `gemma_experiments.py` recipe):
  subspace chosen by source variance rather than anchor correspondence;
  two fitted objects; practically dominated by the direct map.
- *Post-hoc rotation of Ridge outputs:* cannot work in principle — the
  blob is shrinkage anisotropy already baked into the Ridge predictions,
  and orthogonal maps reorient variance collapse, they don't undo it.

## Fit discipline

Mirrors 09b's alpha-selection protocol:

1. Split anchors with shared `anchor_split.group_split` (seed 42) —
   identical train/val/test membership to the Ridge fit.
2. Build X/Y pairs the way 09b does: surface in fused vocab AND gloss in
   English vocab.
3. Fit **two variants on train only**:
   - **full** — all valid (surface, gloss) pairs, same convention as Ridge.
   - **stable** — monosemous anchors: group by surface, keep surfaces with
     exactly one distinct gloss, dedupe (surface, gloss) rows. No frequency
     filter (fused-vocab membership already implies FastText min-count).
     Report the Swadesh-207 intersection count (list hardcoded in the
     script) as a diagnostic only — pure Swadesh is too small to span a
     768-d subspace.
4. Score both on **val**: mean cosine of `x_val·W` vs the gold English
   Gemma vector. Winner refit on **train+val**. Val selection never sees
   Gate 2.
5. If the stable variant has n < 768 pairs: warn (rank-deficient
   subspace), still fit — val decides.
6. Word-level **test split is never touched**.

**Determinism:** no RNG beyond the seed-42 split; `W = UVᵀ` is sign-stable
for distinct singular values (paired sign flips cancel in the product).

## Components

**1. Fitter — `shared/scripts/procrustes_align.py`** (new)

One driver with a per-slot config table (`fused_path`, `anchor_path`,
`surface_key`) following `doc_eval.py`'s multi-slot pattern. Per-slot
artifacts:

- `languages/{slot}/final_output/{slot}_procrustes_gemma_vectors.npz` —
  full slot vocab projected through the winning W (vocab + vectors),
  naming parallel to `{slot}_aligned_gemma_vectors.npz`.
- `languages/{slot}/models/procrustes_W_gemma.npz` — the map, for
  reproducibility.
- `languages/{slot}/results/procrustes_results.json` — per-variant val
  cosine, chosen variant, anchor counts (full / stable / Swadesh
  intersection), fit diagnostics.

**2. Gate 2 rerun — `--space` flag on `shared/scripts/doc_eval.py`**

`run_parallels()` gains `--space ridge|procrustes` (default `ridge`;
existing behavior and shipped results untouched), resolving to the
corresponding `final_output` npz. Procrustes output goes to a new file
`shared/results/doc_eval_parallels_procrustes.json` — the shipped Ridge
FAIL stays on record.

## Data flow

    anchors + fused 1536d + whitened-Gemma 768d   (per slot)
            │
    group_split (seed 42)
            │
    fit W on train — variants: full / stable
            │
    select on val cosine → refit winner on train+val
            │
    project full vocab → {slot}_procrustes_gemma_vectors.npz
            │
    doc_eval --space procrustes → Gate 2 verdict
            │
    journal + myth_study_plan §7 update

## Success criteria (pre-registered, unchanged from Gate 2)

- **PASS:** as pre-registered in `myth_study_plan.md` §Gate 2 —
  "parallel-text retrieval MRR ≥ 0.1, with the positive-control pair
  ranking in the top quartile of its pool (top ~205 of 820)" — applied
  to all three measured control pairs (kumarbi-theogony,
  illuyanka-typhon, ullikummi-typhon).
- **Isometry diagnostic (not a criterion):** computed in the fitter and
  recorded in `procrustes_results.json` — Spearman ρ between the pairwise
  cosine matrices of the val-anchor source vectors before vs after
  mapping. Expected ρ ≈ 1 by construction; materially lower means
  projection loss and gets journaled.
- **On PASS:** Plane A reinstated for future claims; journal entry;
  `myth_study_plan.md` §7 status update. Making new Plane A myth-study
  claims is a follow-on project, not this one.
- **On FAIL:** journal the measured numbers; Plane A stays no-go;
  artifacts remain for future levers (e.g., stronger anchors). No
  threshold adjustment, no retry with tweaked variants.

## Error handling

Fail loudly on missing fused/anchor/cache files, dim mismatches (assert
1536 source / 768 target, as 09b asserts `EXPECTED_TARGET_DIM`), zero
valid pairs. No defensive handling beyond that.

## Testing — `shared/tests/test_procrustes_align.py`

1. `WᵀW ≈ I` on random data (orthonormal columns).
2. Planted-map recovery: `Y = X·W_true` → fitted map reproduces
   cosines ≈ 1.
3. No shrinkage: singular values of fitted W all ≈ 1 (the property Ridge
   lacks — the point of the exercise).
4. Monosemy filter: polysemous surface excluded, duplicate (surface,
   gloss) rows collapsed.
5. Val-selection picks the higher-cosine variant.
6. `doc_eval --space` filename resolution (ridge default unchanged).
