# hyper-glyphy Experiment Journal

Cross-language experiment log. Reverse chronological — newest at the top.

## Recent findings (newest first)

## 2026-07-13 — Procrustes remap measured: Gate 2 FAIL on the semi-orthogonal plane.

Per-slot semi-orthogonal maps (W = UVt of XtY, 1536→768, no scale; variants
full/stable-monosemous selected on val cosine — chosen: sumerian full,
hittite full, greek full; isometry rho ~1.0 for all three, ~0.99999999999,
i.e. no projection loss — structural, not merely measured: the fused 1536d
embeddings have exact rank 768 (FastText 768d + projected component), so a
1536→768 semi-orthogonal map can capture the data subspace exactly; the FAIL is
therefore not attributable to projection loss) as a parallel document-level
plane alongside the production Ridge maps. Val cosine at the chosen variant: sumerian 0.1157,
hittite 0.0586, greek 0.1149. Fit pairs (train-only, from variant selection):
sumerian 6163, hittite 5307, greek 63787 (train+val counts are in each slot's
local `procrustes_results.json`, not tracked in git per repo convention).
Parallel retrieval (Hittite→Greek, pool 820): kumarbi 526, illuyanka 798,
ullikummi 796, MRR 0.0015 (Ridge: 731/781/788, MRR 0.0013). Verdict: FAIL —
Plane A stays no-go; artifacts retained for future levers (stronger anchors).
Spec: docs/superpowers/specs/2026-07-12-procrustes-remap-design.md.

## 2026-07-09 — Eval redesign shipped: stratified CSLS suite + document-level panel; all five slots measured (Greek first run).

The eval-redesign pipeline (spec `docs/superpowers/specs/2026-07-06-eval-redesign.md`) replaces the broken single-stratum eval with a stratified CSLS retrieval suite and adds two document-level panels. Metric definitions: CSLS (k=10 mutual-neighbor reranking) over a 50K-candidate GloVe or Gemma English vocabulary; evaluation reports exact-match top-1/5/10 and synonym-credited top-1/5/10 (credit if the retrieved word shares any WordNet synset with the gold gloss); three regimes per run: **dictionary-in-sample** (a fixed 1,000-anchor sample of the training set — memorization of known glosses; test lemmas are never in train under the lemma-group split, so this regime is measured in-sample by design), **interpolation** (test anchors whose gold gloss appears among train+val targets), **zero-shot** (test anchors whose gold gloss was never a training target). All five slots ran on the same seed-42 64/16/20 lemma-group split (surface-casefold-group for Egyptian, which has no lemma key). Results are the first honest numbers for all five slots; Greek is a first-ever run (G1–G4 scaffold completed 2026-05-11).

**Five-slot suite — top-1 exact % / syn % (50K candidates, best val-selected alpha per target)**

| Slot | Target | alpha | Dict n | Dict top-1 | Dict syn | Interp n | Interp top-1 | Zero-shot n | Zero-shot top-1 | Combined n | Combined top-1 | Combined syn |
|------|--------|-------|-------:|:----------:|:--------:|----------:|:------------:|------------:|:---------------:|-----------:|:--------------:|:------------:|
| Akkadian | GloVe | 0.1 | 962 | 48.54% | 50.73% | 437 | 0.00% | 1857 | 0.38% | 2294 | 0.31% | 1.13% |
| Akkadian | Gemma | 1e4 ¹ | 962 | 19.85% | 22.87% | 437 | 0.46% | 1857 | 0.05% | 2294 | 0.13% | 0.48% |
| Egyptian | GloVe | 1e-4 | 822 | 79.20% | 79.20% | 354 | 18.93% | 105 | 0.95% | 459 | 14.81% | 15.69% |
| Egyptian | Gemma | 10 | 822 | 42.82% | 45.13% | 354 | 20.06% | 105 | 0.00% | 459 | 15.47% | 16.12% |
| Hittite | GloVe | 1e-4 | 363 | 63.36% | 63.64% | 266 | 6.77% | 144 ² | 0.00% | 410 | 4.39% | 4.39% |
| Hittite | Gemma | 1e-4 | 363 | 79.61% | 80.17% | 266 | 10.53% | 144 ² | 0.00% | 410 | 6.83% | 8.05% |
| Sumerian | GloVe | 100 | 962 | 70.79% | 71.10% | 753 | 7.30% | 565 | 0.88% | 1318 | 4.55% | 5.01% |
| Sumerian | Gemma | 1000 | 962 | 74.32% | 75.16% | 753 | 9.43% | 565 | 0.35% | 1318 | 5.54% | 5.69% |
| Greek | GloVe | 1e-4 | 922 | 39.05% | 42.08% | 12487 | 4.06% | 3240 | 0.40% | 15727 | 3.31% | 5.67% |
| Greek | Gemma | 0.1 | 922 | 52.49% | 55.31% | 12487 | 5.33% | 3240 | 0.68% | 15727 | 4.37% | 7.42% |

Gemma beats GloVe combined in 4 of 5 slots (all except Akkadian — see caveat (a)). Dictionary stratum accuracy (39–80% top-1 at 50K candidates) is the strongest signal; interpolation is weak-to-moderate (0–20%); zero-shot is near-zero by design (0–1%), confirming that the linear map does not generalize to unseen lemma families at this corpus scale.

**Caveats (mandatory)**

(a) **Akkadian Gemma anomaly — alpha-selection noise floor.** The Gemma val sweep is flat at 0–0.13% (0–2 correct of 1,568 val items). Alpha=1e4 won by a single anchor over the entire 0.1–1000 plateau (all tied at 0.064%). That over-regularized alpha crushes the dictionary stratum (19.9% vs GloVe 48.5%), making Akkadian the only slot where Gemma underperforms GloVe. This is val-selection noise at near-zero signal, not a whitening-conditioning failure. Known mitigation: break ties toward lower alpha, or select on val top-5 or dictionary-stratum accuracy (a higher-signal metric). Grid extension beyond 1e4 would not help — the curve drops to 0% past 1e4.

(b) **Hittite candidate-vocab gap.** Of 1,320 Hittite test items, 910 (69%) have gold glosses that are OOV of the 50K GloVe candidate vocabulary; only 410 items (31%) are evaluable. The zero-shot 0.00% rate is measured on the 144 kept zero-shot items only. Root cause: German→English gloss translation via EmbeddingGemma for TLHdig yields many specialized glosses absent from GloVe-50K. Mitigation: expand candidate vocab or switch to a multilingual target.

(c) **Leak check: 0.00% all five slots.** Post-split near-surface-edge purge confirmed zero same-gloss edit-distance-1 cross-split leak for all five language slots. Hittite fixed from 10.68% (in the lemma-split-eval branch): root cause was TLHdig citation-form spelling variants (kattan/katta, lugalutti/lugaluttim), now merged before splitting via the shared anchor_split.py union-find. The five-slot 0.00% measurement was a controller-verified run of the leak-check diagnostic over the real anchor files; the splitter's merge/fallback invariants are unit-tested in `shared/tests/test_anchor_split.py` (318 pytest tests pass overall).

**Egyptian data-fix effects.** Egyptian uses surface-casefold-group split (no lemma key in the source lexicon). The pipeline also applies a stopword-gloss filter: **4,018 stopword-gloss pairs dropped**, leaving 4,152 total anchor pairs (2,836 valid / in-vocab). A casefold fallback lookup (`eg_vocab_cf`) recovers anchors whose source surface differs from the FastText vocab entry only in case. GloVe combined 14.81%, Gemma 15.47% — the highest combined accuracy of all five slots, driven by a strong dictionary stratum (79.2% GloVe, 42.8% Gemma at 50K candidates).

**Document-level panel 1 — genre leave-one-out (Sumerian ETCSL, n=338 compositions, 5 genres, majority baseline 40.83%)**

| Space | LOO accuracy | vs majority |
|-------|:------------:|:-----------:|
| gemma_aligned | 63.31% | +22.5 pp |
| glove_aligned | 60.95% | +20.1 pp |
| fused_unaligned | 68.05% | +27.2 pp |

Gate 1 criterion: ≥15 pp over majority in at least one aligned space. **PASS** (gemma_aligned +22.5 pp clears gate; projection cost from native to aligned space ≈5 pp). This validates within-language genre structure in the aligned spaces and Plane B native-space RSA for the myth study.

**Document-level panel 2 — cross-language parallel retrieval (Hittite → Greek, pool=820 Greek documents)**

| Pair | Hittite source | Rank / 820 |
|------|----------------|:----------:|
| Kumarbi (CTH 344) → Theogony | KBo 52.10+, KUB 47.56 | 731 |
| Illuyanka → Theogony | KBo 3.7, KUB 17.5 | 781 |
| Ullikummi (CTH 345) → Theogony | KBo 26.58, KBo 26.61 | 788 |
| **MRR** | | **0.0013** |

Gate 2 criterion: MRR ≥ 0.1, positive control in top quartile (≤205). **FAIL (measured, with three positive-control pairs in play).** All three pairs rank in the bottom 11% of the pool. Controller diagnostics ruled out corpus-coverage failure: Kumarbi document is 72% in-vocab (329 unique tokens); length does not predict low rank (top-50 length distribution matches the pool); mean-centering moved Illuyanka rank 781→773 (negligible). Root cause: cross-language cosines form a non-discriminative blob (entire 820-document pool sits in a ~0.18–0.24 cosine band from any Hittite query, with genre-irrelevant top hits — Lucian, Aristophanes). Conclusion: **word-level alignment does not compose into cross-slot document retrieval.** The KBo 52.10+ join successfully rescued the Kumarbi positive control (KUB 33.120 bare number absent from TLHdig; join contains the full Alalu→Anu→Kumarbi succession, 215 lines); the pair is fully measured, not dropped — and still fails.

**Go/No-Go summary and myth study pointer.** Gate 1 PASS clears within-language genre analysis and Plane B native-space RSA. Gate 2 FAIL puts Plane A cross-language cosine on hold. The myth study (`docs/myth_study_plan.md`) proceeds via **Plane B (native-space second-order RSA) as the primary plane** — Plane B never performs cross-language cosine and is unaffected by map quality. Cross-language Plane A claims are reinstated only if alignment maps are improved via Procrustes remap or a stronger anchor set. Egyptian and Akkadian require per-text corpus segmentation before joining the document-level study.

¹ Akkadian Gemma alpha=1e4 is the grid ceiling; see caveat (a).
² Hittite zero-shot n=144 is the in-vocab subset; 910/1320 (69%) of all Hittite test gold glosses are OOV of the 50K candidate vocab; see caveat (b).

- **2026-07-06 — Eval integrity: lemma-group split + validation-selected alpha. All prior headline numbers are invalidated as leakage artifacts.** A repo-wide review (2026-07-01) found two structural flaws in every slot's evaluation: (1) **surface-variant train/test leakage** — anchor extraction deliberately registers multiple surfaces per lemma with the same gloss, and the 80/20 `train_test_split` over *pairs* let (šarrum, "king") train while (šarru, "king") tested. Measured on the exact shipped seed-42 splits, the fraction of test items with a same-gloss train anchor within edit distance 1: **Akkadian 56.9%, Greek 65.2% (projected), Hittite 43.3%, Sumerian 32.0%, Egyptian 29.5%**. (2) **Alpha tuned on the test set** — every `ridge_alpha_sweep.py` selected alpha on the same split used for reporting.
  **The fix (all five slots, code complete):** shared `shared/scripts/anchor_split.py` — union-find grouping (anchors merge on shared lemma OR shared surface; gloss fallback where no lemma source exists, i.e. Egyptian), 64/16/20 train/val/test, alpha selected by top-1 on validation from a widened grid (floor 1e-4), retrain on train+val, report on the untouched test set; OOV subword-inferred anchors train-only; `ridge_alpha_sweep.py` retired (selection now inline in 09/09b); Sumerian's recorded-alpha bug fixed (trained α=100, recorded α=0.001); Sumerian ETCSL extraction made deterministic (sorted set iteration).
  **Akkadian rerun (the evidence):** GloVe top-1 **27.79% → 0.09%**, whitened-Gemma **36.43% → 0.14%** (α=0.01 both, val-selected). Three-way diagnostic confirmed this is real, not plumbing: the shipped model scores **44.4% top-1 on its own training anchors**; re-running the *new* code with the *old* random-pair split reproduces **27.59%** (vs 27.79% shipped — the old number was leakage, quantitatively); and only **16.3% of lemma-split test glosses appear anywhere in train** — the union-find sends large lemma families to train, so the honest eval is nearly pure zero-shot lexicon induction (unseen lemma, mostly unseen gloss, exact-match over 400k candidates). ~0.1% is ~400× above chance but demonstrates the linear map does not generalize to unseen lemmas at this corpus scale. By extension the flagship Sumerian 52.13% and all other headline numbers measure surface-variant memorization, not translation.
  **Post-fix leak check** (same-gloss ed≤1 across the new splits): Akkadian 0.31%, Greek 2.50%, Sumerian 3.40%, Egyptian 0.00% (by construction), **Hittite 10.68%** — investigated: no splitter bug (zero cross-split pairs share a lemma); 218/251 residual pairs are TLHdig citation-form spelling variants of the same lexeme (`kattan`/`katta`, `lugalutti`/`lugaluttim`), unstable source orthography that needs Akkadian-style cf normalization when Hittite reruns.
  **Scope decision:** Sumerian/Egyptian/Hittite reruns deferred — exact-match top-1 at 400k candidates is the wrong metric in the zero-shot regime this split creates (Egyptian's gloss-grouping makes it 0% by construction). Follow-on: eval redesign (seen/unseen-gloss strata, CSLS retrieval, restricted candidate vocab, synonym credit, Hittite cf-variant merging, document-level evaluation with ETCSL genre labels and known parallels e.g. Kumarbi→Theogony), then one rerun per slot; the aligned spaces remain valid for seen-lemma lookup (~44% top-1 train regime) and document-level comparison. Root README / per-slot docs still quote pre-fix numbers pending the docs refresh. Design: [spec](superpowers/specs/2026-07-01-lemma-split-eval-design.md), [plan](superpowers/plans/2026-07-01-lemma-split-eval.md).

- **2026-05-11 — Hittite v1 shipped: Gemma top-1 40.62%, beats Akkadian's v1.3 (36.43%) on day one.** Source corpus TLHdig from Zenodo (22k texts, CC-BY) — ORACC has near-zero Hittite. Glosses are German; translated via multilingual EmbeddingGemma encoding + NLTK English wordlist filter (no explicit dictionary). Heterograms bridged via existing Sumerian/Akkadian aligned spaces. All Akkadian-arc lessons baked in from day one (alpha sweep, train-only OOV partition, lemma-surface expansion, no speculative fetcher). Despite ~20% the corpus and ~50% the anchors of Akkadian, Hittite outperforms — plausible reasons: TLHdig morphology richer than ORACC `gw`; Hittite is IE (typologically closer to English); alpha sweep prevented late-discovery accuracy loss. See [slot journal](../languages/hittite/docs/EXPERIMENT_JOURNAL.md).

- **2026-05-11 — Akkadian v1.3 Ridge alpha sweep: +7.41pp top-1 from one constant (29.02% → 36.43%).** Inherited alpha=100 from Sumerian's spec was wrong for Akkadian's anchor pool; sweep identifies alpha=0.01 optimal for Gemma, alpha=0.001 for GloVe. Gemma top-10 now **66.51%, exceeding Sumerian's 65.99%**. Coverage gap effectively closed on both targets; remaining top-1 gap (36.43% vs Sumerian's 52.13%) is alignment-precision. Lesson: don't inherit hyperparameters across slots — always sweep alpha (20 min cost, potential +5-10pp payoff). See [slot journal](../languages/akkadian/docs/EXPERIMENT_JOURNAL.md).

- **2026-05-10 — Akkadian v1.2 second gap-closing pass: +7.36pp top-1 (21.66% → 29.02%).** Four more levers after v1.1: L4 global lemma-surface expansion (+3.73pp, the largest single lift in v1.2), L5-refined subword inference with training-only OOV partition (+1.19pp), L6a corpus expansion #2 to 62 ORACC projects / 3M tokens (+1.74pp), and L6b DCCLT bridge anchor bootstrapping (FALSIFIED, -3.55pp, reverted). Top-10 reached 57.08% (Sumerian 65.99%) — coverage gap nearly closed. Top-1 gap to Sumerian (29.02% vs 52.13%) is now an alignment-precision problem, not coverage. See [slot journal](../languages/akkadian/docs/EXPERIMENT_JOURNAL.md).

- **2026-05-10 — Akkadian v1.1 gap-closing pass: +4.91pp top-1 (16.75% → 21.66%).** Executed three improvement levers (mimation wiring, FastText min_count change, SB pretraining corpus). The dominant lever was L3 (corpus expansion); Sumerian's W2b-style normalization win did not exist for Akkadian (the per-slot coverage diagnostic was attribution-decisive). Identified remaining levers (subword inference at eval time, lemma-surface expansion) projected to add another +5-12pp. See [slot journal](../languages/akkadian/docs/EXPERIMENT_JOURNAL.md).

- **2026-05-09 — Akkadian slot v1 shipped:** Third language slot. OB-period scope, ORACC-only anchor lexicon (eBL pivoted — see slot journal). Whitened-Gemma top-1 **16.75%** (vs Sumerian 52.13%, Egyptian 32.35%). 50,636 DCCLT bridge pairs scaffolded for v2 cross-lingual experiment. The corpus is smaller (712k tokens vs 2.8M) and anchor coverage thinner (44% vs 65% valid) than Sumerian — see [`languages/akkadian/docs/EXPERIMENT_JOURNAL.md`](../languages/akkadian/docs/EXPERIMENT_JOURNAL.md) for the full writeup and identified levers (normalization audit is highest-leverage). Design: [spec](superpowers/specs/2026-05-09-akkadian-slot-design.md), [plan](superpowers/plans/2026-05-09-akkadian-slot.md).
