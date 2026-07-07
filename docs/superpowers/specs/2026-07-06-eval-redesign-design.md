# Eval Redesign: Honest Metric Suite + Document-Level Panel — Design

**Date:** 2026-07-06
**Status:** Approved

## Problem

The 2026-07-06 eval-integrity fix (see `2026-07-01-lemma-split-eval-design.md` and the
journal entry of the same date) removed train/test leakage — and revealed that exact-match
top-1 over the full 400k GloVe vocabulary is the wrong metric for the regime the honest
split creates. The lemma-group split sends large lemma families to train, so the test set
is dominated by rare lemmas; only 16.3% of Akkadian test glosses appear in train at all.
Measured that way, honest accuracy is ~0.1% top-1: technically true, scientifically
uninformative, and it hides the regimes where the aligned spaces demonstrably work
(≈44% top-1 on trained anchors; document-level structure untested). Two split defects
also remain: Hittite carries 10.68% residual leakage from TLHdig citation-form spelling
variants, and Egyptian's gloss-group fallback makes exact-match 0% by construction.

## Decisions (user-approved)

- **Headline:** a stratified word-level suite per slot (dictionary / interpolation /
  zero-shot regimes), each with CSLS + restricted candidate vocab + synonym credit.
  Document-level metrics reported alongside as a validation panel.
- **Rerun scope:** all five slots run once on the new suite — Sumerian, Akkadian,
  Egyptian, Hittite re-runs; Greek's first alignment.
- **Doc-level scope:** ETCSL genre benchmark AND cross-language parallel retrieval;
  plus a myth-study planning document as a deliverable.
- **Architecture:** artifact-based harness — training runs save artifacts; a shared
  eval module computes all metrics from artifacts and is re-runnable without retraining.

## Design

### 1. Word-level metric suite

Reported per slot × target space (whitened-Gemma 768d, GloVe 300d) as a fixed table:

| Regime | Evaluation set | Question answered |
|---|---|---|
| **Dictionary** (in-sample) | train anchors (fixed 1,000-anchor sample, seed 42) | how well the map memorizes known glosses |
| **Interpolation** | test anchors whose gold gloss appears as a train target | generalization across word forms |
| **Zero-shot** | test anchors whose gold gloss never appears in train | true lexicon induction |

Strata are properties of the split, computed once per slot and stored in the artifact
bundle. Test lemmas are never in train by construction (lemma-group split), so the
dictionary regime is measured on train anchors and always labeled **in-sample**.

Each cell reports top-1/5/10 under:

- **CSLS retrieval** (Conneau et al. 2018), k=10 neighborhood size — the standard
  hubness correction for cross-lingual embedding retrieval. Implemented in the harness;
  correctness tested against a brute-force reference on toy data.
- **Restricted candidate vocabulary:** the first 50,000 rows of the GloVe cache
  (GloVe files are frequency-ordered; the Gemma cache preserves GloVe's vocab order —
  the same 50k slice applies to both spaces). Gold glosses outside the top-50k are
  excluded from the stratum and counted in a reported `gold_oov_candidates` field —
  never silently dropped.
- **Two accuracy columns, always both:** `exact` (gold string match) and `syn`
  (synonym-credited: a retrieved word counts as a hit if it shares a WordNet synset
  with the gold gloss; lookup lowercased, all POS). Synonym credit never replaces
  exact — the pair travels together in every table.

Raw-cosine / full-vocab numbers remain computable via a harness flag for continuity
with prior results, but are not part of the headline table.

**Alpha selection** switches to the harness's val scorer — CSLS top-1, restricted
vocab, exact-match, over the whole validation set (no strata during selection) — so
the selected alpha optimizes the reported metric family. The `select_alpha` flow in
09/09b is otherwise unchanged (validation-only selection, train+val retrain,
untouched test set).

### 2. Split refinements (`shared/scripts/anchor_split.py`)

- **Near-surface edges (all slots):** `build_groups` gains union edges joining any two
  anchors that share an English gloss AND whose source surfaces are within edit
  distance 1. This is exactly the residual-leak metric's definition, so it drives
  same-gloss/ed≤1 cross-split leakage to ~0 everywhere and specifically absorbs
  Hittite's TLHdig cf spelling variants (`kattan`/`katta`, `lugalutti`/`lugaluttim` —
  218/251 of its leaked pairs). Implemented with a per-gloss bucket scan (glosses are
  the join key; no all-pairs blowup). On by default; `near_surface_edges=False` opt-out
  kept for diagnostics.
- **Egyptian grouping: gloss → case-folded surface.** Egyptian anchors (no `lemmas`
  field) now group by `casefold(surface)` instead of gloss, so a seen-gloss stratum
  exists (gloss grouping made zero-shot exact-match 0% by construction). The
  near-surface edges keep spelling variants same-side. The splitter API: the existing
  gloss fallback is replaced by a `fallback="gloss"|"surface_casefold"` parameter;
  Egyptian's 09/09b pass `surface_casefold`, everything else is unaffected (anchors
  with `lemmas` never hit the fallback).
- **Egyptian data fixes (bundled with its rerun):**
  - *Case-insensitive corpus lookup:* `build_training_data` falls back to a case-folded
    vocab index when the raw `egyptian_raw` key misses — recovers ~1,100 anchors (+18%),
    the `wsjr` vs `Wsjr` gap identified in the 2026-07-01 review.
  - *Stopword-gloss filter:* anchors whose gloss is a pure function word (list seeded
    from Greek's `STOP_WORDS` plus the German fragments `des`, `de`) are excluded from
    training and evaluation. 2,041 "the"-gloss anchors alone are 25% of the pool. The
    filter list and the dropped count are recorded in the results JSON.
- Split changes invalidate prior split assignments; all slots re-split at run time
  (the runs are in scope, so no stale-artifact hazard).

### 3. Artifact harness

- **Artifact bundle** (`languages/<slot>/results/eval_artifacts_<target>.npz` +
  sidecar JSON): ridge coefficients/intercept, per-anchor split assignment and stratum
  label, the projected predictions for test, val, and the train sample, candidate-vocab
  slice definition, config (alpha, seed, split params, filter counts).
- **`shared/scripts/eval_suite.py`:** pure functions over artifacts — `csls_topk()`,
  `stratify()`, `synonym_hits()`, `score_suite()` — plus a CLI
  (`python -m shared.scripts.eval_suite <slot> [--target gemma|glove] [--full-vocab]`)
  that prints the suite table and writes `results/eval_suite_<target>.json`. Adding a
  metric later costs minutes, never a retrain.
- **09/09b integration:** each run writes the artifact bundle, then calls
  `score_suite()` for its results JSON. `select_alpha` imports the harness val scorer.
  The five-language clone convention stands: Akkadian is canonical; Hittite/Greek are
  sed-clones; Sumerian/Egyptian carry their documented deltas (no FastText; PCA path).

### 4. Document-level panel (`shared/scripts/doc_eval.py`)

Document representation: SIF-weighted centroid (weight a/(a+p(w)), a=1e-3, corpus
unigram probabilities) of the document's projected word vectors, in each target space.

- **(a) ETCSL genre classification.** Compositions reconstructed from
  `etcsl_texts.json` line_ids (`c2554.A.1` → composition `c.2.5.5.4`); genre = ETCSL
  top-level catalogue class (c.1 narrative/mythic, c.2 royal, c.4 hymns/cult,
  c.5 wisdom/literature, c.6 proverbs; classes with <10 compositions dropped).
  Metric: leave-one-out nearest-centroid classification accuracy per space, with the
  **unaligned fused FastText** centroids as the sanity baseline — projection through
  Ridge should not destroy genre structure; if it does, that bounds document-level
  claims and gets reported as-is.
- **(b) Cross-language parallel retrieval.** Candidate pairs (verified against our
  corpora as implementation step one; absent pairs dropped and logged, never silently):
  Inanna's Descent (ETCSL c.1.4.1) ↔ Ištar's Descent (Akkadian); Sumerian flood
  (Ziusudra, c.1.7.4) ↔ Atrahasis / Gilgamesh XI flood (Akkadian); Kumarbi cycle
  (Hittite) ↔ Theogony (Greek); Illuyanka (Hittite) ↔ Typhonomachy (Greek);
  Gilgamesh-cycle Sumerian compositions ↔ Akkadian Gilgamesh. Metric: for each pair,
  rank of the true parallel among all cross-language documents in shared whitened-Gemma
  space; report per-pair rank and MRR. This panel validates the representation the
  myth study will use.

### 5. Runs, reporting, and the myth-study planning doc

- One run per slot: Sumerian, Akkadian re-run; Egyptian re-run (with §2 data fixes);
  Hittite re-run (post near-surface fix); **Greek first alignment** (09 + 09b + 10 +
  suite). Budget ~1–4h per slot per target on this host (Akkadian's observed cost).
- Reporting: journal entry with the full suite table per slot and doc-panel results;
  per-slot `final_output/metadata.json` updated to the new suite (schema gains a
  `metric_suite` block; old flat top-k fields retained one release for compatibility);
  root README results section replaced by the suite table with the invalidation banner
  reduced to a one-line historical note.
- **`docs/myth_study_plan.md`:** research questions (cross-civilization cosmogony
  affinity; magical-text vocabulary structure), candidate text sets per slot, method
  (centroid similarity, thematic concept fingerprints, Gemma–GloVe agreement as
  confidence), known-relatedness controls (IE triangle: Hittite/Greek/(future
  Sanskrit); Kumarbi→Theogony as positive control), and explicit go/no-go dependencies
  on this spec's outcomes (genre benchmark must beat chance by a stated margin;
  parallel retrieval MRR threshold).

### 6. Testing (TDD)

- Harness: CSLS vs brute-force reference on toy vectors; strata assignment on
  constructed splits; synonym-credit hits/misses with known WordNet pairs;
  restricted-vocab boundary (gold at row 49,999 in / 50,000 out); artifact round-trip.
- Split: ed≤1 near-surface edges merge (and don't over-merge different-gloss pairs);
  Egyptian case-fold grouping; existing seven `anchor_split` tests stay green
  (near-surface edges may legitimately change proportions-test tolerances — adjust
  tolerances, not invariants).
- Doc panel: composition reconstruction from line_ids; genre bucketing; SIF weighting
  on a toy corpus.
- Regression: leak check re-run post-split-change — target ≤1% for every slot.
  Full pytest green throughout.

## Out of scope

The myth study itself (planning doc only); Sanskrit slot; corpus expansion; alternative
mappings (orthogonal Procrustes — listed as a future lever); Egyptian anchor
retranslation beyond the stopword filter; Mayan (rejected for embeddings — corpus too
small; possible future document-level null control).
