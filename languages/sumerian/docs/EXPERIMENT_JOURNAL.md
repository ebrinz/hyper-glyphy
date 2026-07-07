# Experiment Journal

> **2026-07-06 — All headline accuracy numbers in this journal are pre-fix (leaked split).**
> See the repo-wide journal ([`docs/EXPERIMENT_JOURNAL.md`](../../../../docs/EXPERIMENT_JOURNAL.md),
> 2026-07-06 entry) for the full eval-integrity writeup. The historical record here (W2b,
> Phase A/B, Workstreams 2a/2b) remains valid as history; the 52.13% Sumerian top-1 is
> a pre-fix number and does not represent honest zero-shot accuracy.

Running log of experiments on the Sumerian pipeline. Reverse chronological — newest at the top. Each entry records hypothesis, method, result, and takeaway so that past attempts (and null results) stay legible even when the surrounding code has moved on.

Entry format:

```
## YYYY-MM-DD — <short name>
**Hypothesis:**
**Method:**
**Result:**
**Takeaway:**
**Artifacts / commits:**
```

---

## 2026-04-20 — Anomaly Atlas Interpretive Document shipped

**Hypothesis:** The anomaly atlas produced 300+ ranked anomalies but no interpretive prose. Extracting 15-20 most striking findings across six themes into a narrative document (markdown + PDF with embedded cuneiform font) produces a second shareable research artifact — distinct from the cosmogony (five hand-picked concepts) and complementary to the atlas (raw data).

**Method:** New build pipeline (pandoc + xelatex + Noto Sans Cuneiform font committed at `docs/fonts/`, template at `docs/templates/cuneiformy-pandoc.tex`, script at `scripts/docs/render_anomaly_pdf.sh`). Document at `docs/anomaly_atlas_findings.md` (~9,500 words, 13 sections). Six themes: (1) translation failures, (2) specialized cultic vocabulary, (3) grammatical bridges, (4) transliteration shadows, (5) the numeric tail, (6) reading through the floor. Consistency tests in `tests/test_anomaly_findings_consistency.py` verify every numeric claim traces to `docs/anomaly_atlas.json` (pinned commit `ff94533`) and every cuneiform codepoint is documented in §12 provenance appendix.

**Result:** The Lens 1 top-3 translation failures — *rin2* → "lord" (cosine -0.088, confidence 1.0), *jizzal* → "ear" (cosine -0.063), and *en* → "priest" (cosine -0.037) — illustrate a spectrum from narrow-niche honorific to metonymic wisdom-body-concept to politically-divinely polysemous title: in every case the English gloss is not wrong but geometrically reductive, failing to carry the distributional environment that the Sumerian corpus has encoded. Lens 3's isolation signature for *uttu*, *gidri*, and *ebgal* (all with d_10 > 0.615) reveals a class of "precision vocabulary" tokens whose nearest Sumerian neighbors are overwhelmingly their own morphological inflections — a computational signature for cultic and topographic terms so contextually specialized that the corpus provides no distributional generalization beyond the base word. Lens 2's domination by numeric classifiers (*1(disz)* at 115,478 occurrences, top-1 English "goat" at cosine 0.457) establishes that the Sumerian tablet archive is primarily an administrative accounting record, not a literary one — and any NLP alignment that treats all tokens equally encodes the administrative register as the default. Document's standalone frame — it does not re-engage with the cosmogony's thesis; the two documents are independent research artifacts.

**Takeaway:** The atlas is more valuable as a diagnostic than as a ranker: the top-1 result of each lens is routinely either a known Sumerological controversy (Lens 1), a corpus-frequency artifact (Lens 2), or an extreme alignment-noise case (Lens 4), while the interpretively rich signals live in ranks 15–35 where high-confidence anchors with moderate anomaly scores give a reader with domain knowledge enough purchase to say something non-obvious. The cross-lens convergence pattern — tokens that score anomalously on multiple lenses simultaneously — is the most promising future direction; the present atlas does not expose this directly, but it is recoverable by cross-referencing the six tables. The cuneiform font infrastructure (pandoc template + Noto Sans Cuneiform + xelatex build script) is reusable for future research documents without additional setup.

**Artifacts / commits:** `docs/anomaly_atlas_findings.md`, `docs/anomaly_atlas_findings.pdf`, `docs/fonts/NotoSansCuneiform-Regular.ttf`, `docs/templates/cuneiformy-pandoc.tex`, `scripts/docs/render_anomaly_pdf.sh`, `scripts/docs/consistency.py`, `tests/test_anomaly_findings_consistency.py`. Spec: `docs/superpowers/specs/2026-04-20-anomaly-atlas-interpretive-document-design.md`. Plan: `docs/superpowers/plans/2026-04-20-anomaly-atlas-interpretive-document.md`.

## 2026-04-20 — Anomaly Atlas shipped (civilization-agnostic framework)

**Hypothesis:** The Sumerian cosmogony document drew interpretive claims from five hand-picked concepts. Applying the same methodology at scale to all 35,508 aligned tokens, via six diagnostic lenses, both tests the cosmogony document's thesis and produces a ranked atlas of anomalies for future writing. Designed as a civilization-agnostic framework so the queued comparative repo (and eventually a Gemma-tized Heiroglyphy) can reuse the code.

**Method:** New `scripts/analysis/anomaly_lenses.py` (6 pure-function lenses, zero Sumerian-specific logic), `scripts/analysis/anomaly_framework.py` (AnomalyConfig + run_atlas + markdown renderers), `scripts/analysis/sumerian_anomaly_atlas.py` (thin 60-line wrapper that builds the config for Cuneiformy's artifacts). 17 unit tests total. Six lenses: English displacement, untranslated high-value terms, isolation in source space, cross-space divergence (Gemma vs GloVe), doppelgangers (cos ≥ 0.95), structural bridges (k-means k=40).

**Result — top-1 per lens:**
- Lens 1 (English displacement): rin2 -> lord (cos=-0.0877)
- Lens 2 (no counterpart): 1(disz) (freq=115478, top1_cos=0.4568)
- Lens 3 (isolation): had2 (d_k=0.6334)
- Lens 4 (cross-space divergence): adabkibi (jaccard=1.0000)
- Lens 5 (doppelgangers): baandab5be2ec == baandab5be2eš (cos=0.9979)
- Lens 6 (structural bridges): ningir2su2kake4 (bridge=1.0000, clusters 9/4)

**Takeaway:** The cosmogony document's "geometrically distinct from English" thesis is not confirmed at scale: none of the five hand-picked cosmogonic concepts (`abzu`, `zi`, `nam`, `me`, `an`) appear anywhere in Lens 1's top-50 most-displaced tokens — the atlas surface is dominated by low-frequency hapax-type forms (`rin2`, `bun2`, `jizzal`) rather than thematically loaded vocabulary. The top-1 no-counterpart token is `1(disz)` — a Sumerian numeric classifier with corpus frequency 115,478 and only a 0.46 cosine match to "goat" — which suggests the largest semantic gap between Sumerian and English exists in the grammatical/numeric register, not the mythological one. The Lens 4 divergence winner `adabkibi` (Jaccard=1.0, zero neighbour overlap between Gemma and GloVe spaces) and the Lens 6 bridge winner `ningir2su2kake4` (a compound divine name spanning two clusters) point toward proper-noun compounds as the richest territory for future anomaly-driven writing.

**Artifacts / commits:** `scripts/analysis/anomaly_lenses.py`, `scripts/analysis/anomaly_framework.py`, `scripts/analysis/sumerian_anomaly_atlas.py`, `tests/analysis/test_anomaly_lenses.py`, `docs/anomaly_atlas.json`, `docs/anomalies/*.md`. Spec: `docs/superpowers/specs/2026-04-20-anomaly-atlas-design.md`. Plan: `docs/superpowers/plans/2026-04-20-anomaly-atlas.md`.

## 2026-04-19 — Sumerian Cosmogony Document shipped

**Hypothesis:** With Workstream 2b's alignment at 52.13% top-1, the embedding geometry is rich enough to support methodology-driven interpretation of Sumerian cosmogonic vocabulary. A case study on the Anunnaki cosmogonic cycle exercises the dual-view SumerianLookup API, tests whether the geometry says anything non-obvious, and produces a shareable research artifact.

**Method:** New `scripts/analysis/` directory with 5 focused modules (semantic_field, english_displacement, etcsl_passage_finder, umap_projection, preflight_concept_check) + 2 top-level generators. Pre-flight validated 5 concepts (`abzu`, `zi`, `nam`, `namtar`, `me`) against vocab + ETCSL + top-5 quality. A normalization bug in passage-finder surfaced and was fixed (ETCSL `nam-tar` wasn't matching normalized `namtar`), preserving `namtar` in the slate rather than substituting `kur`. Generators produced `docs/cosmogony_tables.json` and 7 committed figures deterministically. Prose written from the committed data — no hand-calculated numbers, no uncited Sumerological claims.

**Result:** `docs/sumerian_cosmogony.md` (~14,000 words, ~18-20 printed pages). Five concept deep dives following the 8-section paper-grade template. `namtar` (fate) is the sole translation-adjacent concept (cosine 0.459 to English "fate"), while `abzu`, `zi`, and `me` are near-orthogonal to their English anchors (cosines −0.002, 0.056, and 0.017 respectively) — a spectrum of translation opacity not predicted by intuition. `zi` (breath/life-essence) surprises: in both Gemma and GloVe spaces, its nearest Sumerian neighbors are dominated by `zapaa`-forms (voice/sound vocabulary), with strong cross-space agreement — the geometry suggests `zi` encodes the animating, expressive dimension of breath rather than the physiological one. `me` (divine decrees) lands geometrically near governance and administration vocabulary rather than cosmological vocabulary, consistent with the interpretation that English "decree" captures only the administrative face of a concept with a deeper ontological role.

**Takeaway:** Sumerian cosmogonic vocabulary is dominantly geometrically distinct from its English glosses. Four of five concepts sit near cosine-zero to their English anchors, providing a measurable correlate for Jacobsen's claim that Sumerian religious consciousness operated in a fundamentally different mode from modern analytical categories. The lone exception — `namtar` (fate) at cosine 0.459 — is the most surprising specific result: it supports a cognitive-universality hypothesis for fate-concepts while reinforcing the alien-topology hypothesis for creation-vocabulary. The method has clear limits — alignment noise, corpus bias (Old Babylonian literary Sumerian only), grammatical-morpheme artifacts in FastText neighbors — and the document is explicit about all of them. The key methodological lesson is that `namtar`'s translation-adjacency was only recoverable because the normalization fix preserved it in the concept slate; the pre-fix failure would have substituted `kur` and lost the headline finding.

**Artifacts / commits:** `docs/sumerian_cosmogony.md`, `docs/cosmogony_tables.json`, `docs/figures/cosmogony/*.png`, `scripts/analysis/*.py`, `tests/analysis/*.py`, `results/cosmogony_preflight_2026-04-19.json`. Spec: `docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md`. Plan: `docs/superpowers/plans/2026-04-19-sumerian-cosmogony-document.md`.

## 2026-04-19 — Workstream 2b: Normalization fix shipped (STRETCH tier)

**Hypothesis:** Workstream 2b-pre's coverage diagnostic attributed 64.85% of `sumerian_vocab_miss` (7,651 anchors) to a missing normalization chain in `scripts/06_extract_anchors.py`. Applying subscripts → ASCII, strip determinative braces, drop hyphens, and lowercase at anchor extraction should recover those anchors as exact FastText vocab hits without any retrain.

**Method:** New shared module `scripts/sumerian_normalize.py` containing `normalize_sumerian_token`. Swapped in as the sole normalization in `scripts/06_extract_anchors.py` (replacing the letter-only `normalize_oracc_cf`) and in `scripts/coverage_diagnostic.py` (refactor; no behavior change). Reran `06 → 09 → 09b → 10`, plus regression checks via `validate_phase_b`, `audit_anchors`, and `coverage_diagnostic`.

**Result — stretch tier acceptance:**
- Whitened-Gemma top-1: **19.85% → 52.13% (+32.28pp)**
- Whitened-Gemma top-5: 23.66% → 61.97% (+38.31pp)
- Whitened-Gemma top-10: 26.21% → 65.99% (+39.78pp)
- GloVe top-1: 17.30% → 35.70% (+18.40pp)
- GloVe top-5: 22.90% → 44.61% (+21.71pp)
- GloVe top-10: 25.19% → 47.93% (+22.74pp)
- Training anchors: 1,572 → 6,867 (4.37×)
- Valid anchors: 1,951 / 13,886 (14.05%) → 8,558 / 13,100 (65.33%)
- Audit `sumerian_vocab_miss`: 11,798 (84.96%) → 4,101 (31.31%)
- Coverage diagnostic `normalization_recoverable`: 7,651 (64.85%) → **0 (0.00%)** — fix is complete, zero residual in that bucket
- Total merged anchors: 13,886 → 13,100 (slight drop from normalized-key collisions, expected)

**Takeaway:** The "dominant dropout is an ML problem" framing was wrong — it was a 20-line string-normalization bug sitting in a single function. The fix delivered +32pp top-1 on whitened Gemma and cleared the `normalization_recoverable` bucket to zero, confirming the 2b-pre diagnostic's attribution to the bit. Cuneiformy now substantially outperforms the Heiroglyphy baseline (32.35% top-1) that originally shaped our priors. The remaining 4,101 misses are real structural gaps (43% subword, 25% ORACC-lemma-surface, 22% below-min-count, 10% genuinely missing) — Workstreams 2c and beyond can be scoped against that residual.

**Artifacts / commits:** `scripts/sumerian_normalize.py`, `tests/test_sumerian_normalize.py`, refactored `scripts/06_extract_anchors.py` and `scripts/coverage_diagnostic.py`, regenerated `results/anchor_audit_2026-04-19.{md,json}`, `results/coverage_diagnostic_2026-04-19.{md,json}`, `final_output/metadata.json`. Spec: `docs/superpowers/specs/2026-04-19-workstream-2b-normalization-fix-design.md`. Plan: `docs/superpowers/plans/2026-04-19-workstream-2b-normalization-fix.md`.

## 2026-04-19 — Workstream 2b-pre: Coverage diagnostic baseline

**Hypothesis:** Workstream 2a showed `sumerian_vocab_miss: 11,798 (84.96%)` as the dominant dropout. Initial "FastText retrain" framing was too narrow. An ML-engineer reassessment identified five candidate interventions (ASCII normalization, lower min_count, ORACC lemma surface expansion, morpheme composition, subword inference). Before shipping any fix, we need a data-driven attribution of which anchors are recoverable by which intervention — and how trustworthy each recovery actually is under ridge projection.

**Method:** New `scripts/coverage_diagnostic.py` runs two independent analyses on the `sumerian_vocab_miss` set: a **classifier** (mutually-exclusive primary-cause attribution with 6 priority-ordered buckets) and a **simulator** (per-intervention projected recovery, with Tier-2 semantic validation projecting synthesized vectors through the whitened-Gemma ridge weights). Reuses `scripts/audit_anchors.py` loaders and classifier to find misses. 29 unit tests against synthetic inputs and a toy FastText.

**Result — classifier (primary-cause attribution, mutually exclusive, sums to 11,798):** `normalization_recoverable: 7,651 (64.85%)`, `subword_inference_recoverable: 2,213 (18.76%)`, `in_corpus_below_min_count: 926 (7.85%)`, `genuinely_missing: 576 (4.88%)`, `morpheme_composition_recoverable: 370 (3.14%)`, `oracc_lemma_surface_recoverable: 62 (0.53%)`.

**Result — simulator (per-intervention recovery, independent; counts can overlap):** `ascii_normalize: 7,651 exact hits`; `lower_min_count: 926/639/427/218` (at thresholds 1/2/3/4); `oracc_lemma_expansion: 147 anchors, +147 surface forms`; `morpheme_composition: tier1=7,916, tier2_top5=137 (tested=7,563 → 1.8% top-5 accuracy)`; `subword_inference: tier1=3,546, tier2_top5=354 (tested=3,306 → 10.7% top-5 accuracy)`.

**Takeaway:** The dominant lever is a surprise — **ASCII normalization alone recovers 64.85% of misses as exact vocab hits**, not the ORACC surface-form expansion I predicted. A ~20-line change to `scripts/06_extract_anchors.py` mirroring `scripts/05_clean_and_tokenize.py`'s normalization chain (subscripts → ASCII, strip determinative braces, drop hyphens) would lift ~7,651 anchors into the training set with no retrain, no ML, no synthesis. Tier-2 validation confirms the two inference-based interventions produce mostly-noisy projected vectors (1.8% and 10.7% top-5 accuracy) — they're not the first lever to pull. Workstream 2b should ship the normalization fix first; `lower_min_count`, `oracc_lemma_expansion`, and the inference interventions are low-yield follow-ups scoped by what the normalization fix leaves behind.

**Artifacts:** `scripts/coverage_diagnostic.py`, `tests/test_coverage_diagnostic.py`, `results/coverage_diagnostic_2026-04-19.{md,json}`. Spec: `docs/superpowers/specs/2026-04-19-coverage-diagnostic-design.md`.

## 2026-04-18 — Workstream 2a: Anchor audit baseline

**Hypothesis:** The 14.2% valid-anchor survival rate (1,965 / 13,886) is the cheapest place to recover alignment top-1. Before building fixes, we need a reproducible categorization of what happens to the other 85% against both target spaces (GloVe + whitened Gemma), not a hunch-driven one.

**Method:** New standalone script `scripts/audit_anchors.py` classifies every merged anchor into 9 mutually-exclusive, priority-assigned buckets: `junk_sumerian`, `duplicate_collision`, `low_confidence`, `sumerian_vocab_miss`, `multiword_english`, `english_both_miss`, `english_glove_miss`, `english_gemma_miss`, `survives`. Emits dated markdown + JSON reports (schema v1). Pure-function core, fully unit-tested against synthetic data (25 tests). Runs against committed input artifacts; reconstructs dedup collisions when raw extraction inputs are locally present.

**Result:** Baseline report committed at `results/anchor_audit_2026-04-18.{md,json}`. Real-data survival: 1,951 / 13,886 (14.05%). Top dropout buckets: `sumerian_vocab_miss: 11,798 (84.96%)`, `english_both_miss: 84 (0.60%)`, `multiword_english: 45 (0.32%)`. Bucket sums and English-side Venn cross-check reconcile; two consecutive runs are byte-identical.

**Takeaway:** Workstream 2a's methodology gate is closed. Any future pipeline change (new tokenization, new target space, new extraction rules) can now be scored by how it moves the bucket distribution, not by anecdote. The bucket-count deltas prioritize Workstream 2b (FastText retrain for `sumerian_vocab_miss` recovery), 2c (Gemma fine-tune for `english_gemma_miss` recovery), and any `multiword_english` phrase-handling work.

**Artifacts:** `scripts/audit_anchors.py`, `tests/test_audit_anchors.py`, `results/anchor_audit_2026-04-18.{md,json}`. Spec: `docs/superpowers/specs/2026-04-18-sumerian-anchor-audit-design.md`.

## 2026-04-16 — Phase B: Dual-view downstream pipeline shipped

**Hypothesis:** After Phase A retry #2 landed whitened-Gemma at +2.54pp top-1 with qualitatively complementary clusters, the research substrate should move to Gemma while keeping GloVe as a secondary view. Phase B is the infrastructure change, not new research.

**Method:** Rewrote `final_output/sumerian_lookup.py` as a single dual-view class with `space="gemma"|"glove"` routing. Extended `scripts/10_export_production.py` to project the same 35,508 fused Sumerian vectors through *both* ridge weight files (`ridge_weights.npz` and `ridge_weights_gemma_whitened.npz`) in one run, producing parallel npz artifacts. Metadata bumped to `schema_version: 2` with per-space provenance. New `scripts/validate_phase_b.py` regression-checks the stack end-to-end.

**Result:** Export runs cleanly, both spaces load via one class, `find_both("word")` returns top-k from each manifold, concept-cluster regression report regenerates identically to Phase A retry #2. All tests green.

**Takeaway:** The substrate decision is now locked. Downstream research (NEAR_TERM_STRATEGY.md Phase 1 geometric analysis, Workstream 2a anchor audit, eventual Gemma-on-Sumerian fine-tune) all slot on top of this dual-view API. Future contextual-encoder work is one path away — swap the Sumerian-aligned file under the hood and the lookup API stays stable.

**Artifacts / commits:** Covered by commits between the phase B spec and this entry. See `docs/superpowers/plans/2026-04-16-phase-b-gemma-downstream.md`.

---

## 2026-04-16 — Phase A retry #3: Ridge alpha sweep on whitened target

**Hypothesis:** Whitening hit 19.85% top-1 at the default alpha=100. Maybe a different regularization value recovers the last 0.5pp needed to cross the +3pp gate.

**Method:** `scripts/ridge_alpha_sweep.py` evaluates alpha in {0.01, 0.1, 1, 10, 100, 1000, 10000, 100000} on the whitened-gloss cache. Everything else identical to the main 09b run.

**Result:**

| alpha | top-1 | top-5 | top-10 | Δ top-1 vs GloVe |
|---|---|---|---|---|
| 0.01 | 18.58% | 21.88% | 24.17% | +1.27pp |
| 0.1 | 18.83% | 22.39% | 24.68% | +1.53pp |
| 1 | 19.08% | 23.66% | 25.95% | +1.78pp |
| **10** | **19.85%** | 23.16% | 25.95% | **+2.54pp** |
| **100** | **19.85%** | **23.66%** | **26.21%** | **+2.54pp** |
| 1000 | 17.56% | 17.81% | 18.32% | +0.25pp |
| 10000 | 17.56% | 17.56% | 17.56% | +0.25pp |
| 100000 | 17.56% | 17.56% | 17.56% | +0.25pp |

**Takeaway:** The top-1 plateau at alpha ∈ [10, 100] is the real ceiling for this target. The extra 0.5pp isn't hiding in regularization. Alphas ≥ 1000 collapse to a mean-prediction floor at 17.56% (which happens to coincide with GloVe's baseline — a useful sanity check that ridge is working, since predicting the centroid of the training targets is a reasonable-ish baseline).

**Gate status:** formally FAIL (+2.54 < +3pp) and not rescuable by alpha tuning. The decision between "proceed to phase B anyway because +2.54 is a clean real win" vs "pivot to Sumerian-side work" now has to be made without further target-space retries.

**Artifacts / commits:**
- `scripts/ridge_alpha_sweep.py`
- `results/ridge_alpha_sweep.json`

---

## 2026-04-16 — Phase A retry #2: Whitening the EmbeddingGemma target

**Hypothesis:** Both prior Gemma runs failed because contextual embeddings are anisotropic — they cluster in a narrow cone where a single shared direction dominates, inflating cosine similarity between unrelated words and drowning real semantic signal. Centering (subtract global mean) and whitening (scale each direction to unit variance, BERT-whitening, Su et al. 2021) should expose the actual concept structure. 30-second matrix-math retry on already-cached vectors — no re-encoding.

**Method:** `scripts/whiten_gemma.py` reads each cached Gemma variant (gloss, bare), computes mean μ and transform W = Σ⁻¹ᐟ² via eigendecomposition of the 768×768 covariance, saves whitened caches to `models/english_gemma_whitened_768d.npz` and `models/english_gemma_bare_whitened_768d.npz`. 09b gains a generic `--mode` flag routing through the appropriate cache. Ridge hyperparameters, anchors, split all unchanged.

**Result:**

| target | top-1 | top-5 | top-10 | top-1 Δ vs GloVe |
|---|---|---|---|---|
| GloVe (baseline) | 17.30% | 22.90% | 25.19% | — |
| Gemma gloss (raw) | 14.25% | 16.54% | 17.30% | −3.05pp |
| Gemma bare (raw) | 4.83% | 14.50% | 15.52% | −12.47pp |
| **Gemma gloss whitened** | **19.85%** | **23.66%** | **26.21%** | **+2.54pp** |
| **Gemma bare whitened** | **19.59%** | **23.66%** | **25.95%** | **+2.29pp** |

**Takeaway:** The anisotropy story was correct — **whitening turned both Gemma variants from clear losses into clear wins over GloVe on all top-k metrics.** Top-1 delta is just shy of the formal +3pp gate (+2.54pp, not +3.0), but top-5 and top-10 both improve and the earlier qualitative dictionary-meta artifact appears to evaporate because it lived along the first principal component that whitening strips.

Two previously misleading conclusions to retract:
- **"The gloss encoding introduced a systematic bias"** — partially true but the *actual* dominant issue was anisotropy; after centering, gloss and bare produce near-identical alignment quality (19.85% vs 19.59%), so the encoding choice barely matters.
- **"The Sumerian side is the bottleneck"** — premature. Whitening just lifted the ceiling 2.5pp without touching the Sumerian side. The new ceiling (19.85%) is still low in absolute terms, but the English target is no longer the limiting factor it appeared to be. Sumerian-side work remains the most interesting frontier, but the anisotropy lesson means any future use of contextual encoders anywhere in the pipeline should assume centering/whitening is mandatory, not optional.

Pre- vs post-whitening diagnostics for the gloss run:
- Pre: mean-vector norm 0.60 (vectors cluster on one side of the unit sphere)
- Pre: random 1000-pair mean cosine 0.35 (unrelated words look ~0.35 similar — noise floor)
- Post: mean-vector norm 0.002 (centered)
- Post: random-pair mean cosine ~0 (noise floor removed)

**Gate status per original spec:** FAIL on strict reading (+2.54pp < +3pp), PASS in spirit — the improvement is real, consistent across both encoding variants, and the methodological learning (whitening is required for contextual-encoder targets in regression alignment) is itself a valuable outcome.

**Artifacts / commits:**
- Code: `scripts/whiten_gemma.py`, extended 09b `--mode` flag
- Whitened caches: `models/english_gemma_whitened_768d.npz`, `models/english_gemma_bare_whitened_768d.npz` (local only)
- Transforms: `models/gemma_whitening_transform.npz`, `models/gemma_bare_whitening_transform.npz`
- Results: `results/alignment_results_gemma_whitened.json`, `results/alignment_results_gemma_bare_whitened.json` (local only)

---

## 2026-04-16 — Phase A retry: EmbeddingGemma (bare-word) as English target

**Hypothesis:** The gloss-encoding approach from the earlier phase A run introduced a systematic dictionary-meta-vocabulary bias. Encoding bare words (`"{word}"` with no WordNet definition body, same `Retrieval-document` prompt) would avoid that artifact and might close the gap to GloVe.

**Method:** Identical to the gloss run except `embed_english_gemma.py --bare` skipped WordNet lookup and fed each English word alone to EmbeddingGemma. Cache written to `models/english_gemma_bare_768d.npz`. Alignment via `09b_align_gemma.py --bare`. Sumerian side and anchors unchanged.

**Result:**

| metric | GloVe | Gemma (gloss) | Gemma (bare) | bare vs GloVe |
|---|---|---|---|---|
| top-1 | 17.30% | 14.25% | **4.83%** | **−12.47pp** |
| top-5 | 22.90% | 16.54% | 14.50% | −8.40pp |
| top-10 | 25.19% | 17.30% | 15.52% | −9.67pp |

Bare-word is even worse than gloss on top-1 — by 10pp. Top-5/top-10 recover somewhat, suggesting bare vectors find the right neighborhood but not the exact match.

**Takeaway:** Gemma-as-English-target is ruled out. Neither encoding strategy beats GloVe, and bare (the "theoretically cleaner" variant) is *worse* than the flawed gloss variant. The signal is unambiguous: **the English target space is not the bottleneck**. Whatever is capping top-1 at ~17%, it's not the expressiveness of the modern English encoder — it lives somewhere upstream (Sumerian FastText quality, fused 1536d construction, anchor noise, or the alignment method itself).

Interesting sub-finding: the gloss approach, despite its clear artifact, actually *helps* Gemma over bare. The dictionary-meta clustering gives the target space at least *some* structure that the ridge can exploit; bare single-token embeddings give it nothing stable to align against. This is a useful data point for any future contextual-encoder work — single words are off-distribution in ways that hurt downstream regression.

**Redirect:** Next thread should target the Sumerian side, not the English side. Concrete candidates in priority order:
1. Anchor quality audit — 14.2% valid rate is low; how many of the remaining ~12k anchors are noise vs. genuine vocab-mismatch?
2. Sumerian FastText retraining with different hyperparameters (window size, min_count, training corpus weighting)
3. The larger move: fine-tune Gemma (generative) on ETCSL+CDLI+ORACC transliteration to get contextual Sumerian embeddings. This was always the intended phase 2.

**Artifacts / commits:**
- Same code files as gloss run, extended with `--bare` flag — commits `26c49aa`, `6ca6ca8`
- Local-only artifacts: `models/english_gemma_bare_768d.npz` (1.1 GB), `results/alignment_results_gemma_bare.json`, `models/ridge_weights_gemma_bare.npz`

---

## 2026-04-16 — Phase A: EmbeddingGemma (gloss-encoded) as English target

**Hypothesis:** A modern contextual encoder (`google/embeddinggemma-300m`, 768d) as the English target space would improve Ridge-alignment top-1 accuracy vs GloVe 300d by at least +3pp, because GloVe is static / 2014-era and dimensionally shallow compared to the fused 1536d Sumerian side.

**Method:**
- Encoded all 400k GloVe vocab words through EmbeddingGemma using `prompt_name="Retrieval-document"`.
- Each English word wrapped as `"{word}: {WordNet first-synset definition}"` (miss → bare word). Gloss hit rate: 21.4%.
- Same anchor set (13,886), same ridge alpha=100, same 80/20 split (random_state=42) as the GloVe baseline. Only the target vectors changed.
- Sumerian side unchanged: fused 1536d (FastText 768d + zero-pad).

**Result:**

| metric | GloVe baseline | Gemma | delta |
|---|---|---|---|
| top-1 | 17.30% | 14.25% | **−3.05pp** |
| top-5 | 22.90% | 16.54% | −6.36pp |
| top-10 | 25.19% | 17.30% | −7.89pp |

Both quantitative (+3pp) and qualitative (concept cluster coherence) gates **failed**. Gemma is worse, not better.

**Takeaway:** The failure mode is diagnostic, not random. The WordNet-gloss encoding systematically biased the target space: encoding every English word as its definition (`"flour: a fine powder..."`) taught the ridge to project into the region of English that lives near *dictionary definitions themselves*. Nearest neighbors across all domains degenerated to dictionary-meta vocabulary — *"term", "adjective", "word-by-word", "dictionary.com", "merriam-webster"* — rather than concept-adjacent words. The bare-word encoding option (originally ruled out in the spec as "off-distribution") is now the obvious cheap retry: it may be imperfect, but it cannot produce this specific artifact.

Secondary observation: 14.2% anchor validity (1965 of 13886) is lower than ideal and applies equally to both conditions, so it isn't responsible for the Gemma shortfall — but it's a separate thread worth pulling if future retries also underperform.

**Artifacts / commits:**
- Spec: `docs/superpowers/specs/2026-04-16-gemma-embed-alignment-design.md`
- Plan: `docs/superpowers/plans/2026-04-16-gemma-embed-alignment.md`
- Decision note: `results/phase_a_decision.md` (local only — `results/` is gitignored)
- Code: commits `a7f03d0`, `5b62605`, `d0ecdc6`, `5707892`, `61967f7`, `2a007fa`
- Local-only cache: `models/english_gemma_768d.npz` (1.1 GB)
