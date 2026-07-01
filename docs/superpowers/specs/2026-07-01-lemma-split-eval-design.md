# Lemma-Group Split + Validation Split — Design

**Date:** 2026-07-01
**Status:** Approved

## Problem

Two structural flaws inflate every reported top-k accuracy number in the repo:

1. **Surface-variant train/test leakage.** Anchor extraction deliberately registers
   multiple surface forms per lemma with the same English gloss (Akkadian L4 expansion,
   Greek cf+form pairs, Hittite cf+form, Sumerian cf/form-as-separate-anchors). The
   80/20 split (`train_test_split`, seed 42) is over *(surface, gloss)* pairs, so
   (šarrum, "king") can train while (šarru, "king") is tested — near-identical FastText
   subword vectors, identical gold target. Measured on the exact shipped seed-42 splits
   (same-gloss train anchor within edit distance 1 of a test surface): Akkadian 56.9%,
   Hittite 43.3%, Sumerian 32.0%, Egyptian 29.5%, Greek 65.2% (of its future test set).
   The L5 subword-inference feature amplifies this: OOV variants of test surfaces are
   synthesized into training.
2. **Alpha tuned on the test set.** `ridge_alpha_sweep.py` (and Egyptian's `--sweep`)
   evaluates all alphas on the same seed-42 test split used for final reporting. Every
   headline number is best-of-sweep on test; no validation split exists anywhere.

Related bug fixed as a side effect: Sumerian `09_align_and_evaluate.py` trains with
`alpha=100` but records `"alpha": 0.001` in the results artifact (irreproducible).

## Decisions (user-approved)

- **Split unit:** lemma-connected groups; Egyptian (no lemma source in repo) falls back
  to grouping by English gloss.
- **Split shape:** 64/16/20 train/val/test over anchor mass. Alpha selected on
  validation only; final model retrained at the chosen alpha on train+val; top-k
  reported on the untouched test set.
- **Scope:** fix all five pipelines; re-run extraction + sweep + alignment + export for
  the four shipped languages (Sumerian, Egyptian, Akkadian, Hittite). Greek's scripts
  are fixed but not run (its first alignment is a separate task). README/ROADMAP
  refresh is a separate task.

## Design

### 1. Anchor extraction — `languages/{sumerian,akkadian,hittite,greek}/scripts/06_extract_anchors.py`

Each emitted anchor record gains a `"lemmas": [<normalized cf>, ...]` field listing
every citation form that contributed that (surface, gloss) pair:

- **Akkadian:** in-record cf plus every cf whose L4 surface expansion produced this
  surface (accumulate cf per `(surface, gw)` during `extract_oracc_anchors`).
- **Hittite / Greek:** cf from the lemma record being registered.
- **Sumerian:** ePSD2-path anchors get the record's cf. ETCSL co-occurrence anchors
  have no citation form → `lemmas: [<surface>]` (singleton group).
- **Egyptian:** extraction lives outside this repo; anchor file untouched (fallback
  grouping at split time).

Extraction logic is otherwise unchanged. **Verification:** re-run extraction and diff —
the (surface, gloss) pair set, counts, confidences, and ordering must be identical to
the shipped anchor files; only the new field may differ.

### 2. Shared split module — `shared/scripts/anchor_split.py` (new)

One function used by all five pipelines:

```
group_split(anchors, *, source_key, val_size=0.16, test_size=0.20, seed=42)
  -> (train_anchors, val_anchors, test_anchors)
```

- **Grouping (union-find):** two anchors merge if they share a lemma OR share a source
  surface (a surface serving two lemmas must land on one side). Anchors with no
  `lemmas` field group by English gloss (Egyptian; guarantees no gold label spans the
  split there).
- **Assignment:** groups shuffled deterministically (seed), walked once: assign to test
  until ≥ test_size of total anchor mass, then to val until ≥ val_size, remainder
  train. Deterministic given (anchors, seed).
- **L5 OOV rule moves here:** a subword-inferred (OOV) anchor follows its group.
  Train-group OOVs are trained on; **val/test-group OOVs are dropped entirely**
  (currently they leak into training). Val/test remain in-vocab-only, as before.
  Concretely: the split is computed over all anchors first; after `build_training_data`
  marks `subword_inferred`, OOV anchors in val/test partitions are discarded.

### 3. Alignment — `09_align_and_evaluate.py`, `09b_align_gemma.py`, `ridge_alpha_sweep.py` (× 5 languages)

- All three consume the identical group split, computed over anchor records **before**
  X/Y construction, so GloVe and Gemma paths share one partition.
- `ridge_alpha_sweep.py`: selects alpha by top-1 on **validation**; grid extended down
  to 1e-4 (`[1e-4, 1e-3, 1e-2, ...]`) — Akkadian and Hittite optima were pinned at the
  old 1e-2 floor. (Flagged: technically a separate accuracy lever, bundled because the
  sweep is being rewritten anyway.)
- Final run: retrain at chosen alpha on train+val; report top-1/5/10 on test.
- Results JSON gains a `split` block — `{method: "lemma-group" | "gloss-group",
  seed, n_groups, train/val/test sizes, oov_dropped}` — and records the actual alpha
  used (fixes the Sumerian alpha-recording bug).

### 4. Production export — `10_export_production.py` (behavior unchanged)

Exports the same train+val weights the reported metrics came from. (All-anchor
retraining for production was considered and deferred — it would decouple the shipped
artifact from the reported numbers.)

### 5. Testing (TDD — tests first)

New `shared/tests/test_anchor_split.py` (shared/tests is already on the pytest
testpaths) unit tests for `anchor_split`:

- No lemma appears in two partitions; no source surface appears in two partitions.
- Gloss-fallback anchors: no gloss spans partitions.
- OOV anchors of val/test groups are dropped; OOV anchors of train groups retained.
- Deterministic under fixed seed; proportions ≈ 64/16/20 on realistic group-size
  distributions.
- Regression: re-measure the edit-distance-1 same-gloss cross-split leak rate on the
  new real splits — must be ~0 at lemma level (report the residual cross-lemma rate).

Existing 282 tests must stay green.

### 6. Rerun + reporting

- Re-run 06 for Sumerian/Akkadian/Hittite/Greek (raw/cached sources verified on disk
  first; if a source is missing, reconstruct lemmas from the cached lemma streams,
  e.g. `data/dictionaries/oracc_lemmas.json`).
- Re-run sweep + 09 + 09b + 10 for Sumerian, Egyptian, Akkadian, Hittite. 07
  (FastText) is **not** rerun — existing trained models are reused, so FastText
  nondeterminism does not touch this change.
- Corrected metrics written to `results/*.json` and `final_output/metadata.json`.
- One cross-language entry in `docs/EXPERIMENT_JOURNAL.md`: methodology change,
  before/after table per language. Headline numbers are expected to drop; that is the
  point.

## Out of scope

Hittite whitened/unwhitened bridge-anchor bug; FastText seeding; Egyptian stopword-gloss
triage; Egyptian case-normalization gap; Greek first alignment run; root README/ROADMAP
refresh. All tracked from the 2026-07-01 repo review.
