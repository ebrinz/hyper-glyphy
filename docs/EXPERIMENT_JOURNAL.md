# hyper-glyphy Experiment Journal

Cross-language experiment log. Reverse chronological — newest at the top.

## Recent findings (newest first)

## 2026-07-19 — Suite v2 shipped: shared gloss_filters + alpha-v2 plateau rule, all six slots re-run.

**Recipe deltas.** `shared/scripts/gloss_filters.py` is now the single source
of anchor-English selection across all six slots: `NEGATORS = {not, no,
without, never}`; `DE_NEGATORS = {nicht, kein, keine, keinen, ohne, nie,
niemals}`; `XREF_STARTERS = {see, cf, vid}`; `SCAFFOLD_WORDS` (14 words, see
module); `STOP_WORDS` = the inherited Greek stop-word set minus `not`/`no`;
a Unicode word regex. Two selection functions serve two slot families:
`first_english(gloss, eng_vocab_set, negators=NEGATORS)` for the
dictionary-join slots (Greek, Sanskrit) and `gw_is_usable(value,
negators=...)` for the value slots (Sumerian ePSD2, Akkadian,
Hittite — with `DE_NEGATORS` — and Egyptian). `MIN_HIT_RATE` 0.40 gate is
unchanged; `anchor_stats.json` is now the canonical per-slot artifact.
Alpha selection moved to **alpha-v2**: select on val top-5 CSLS, plateau
defined as within 100/n_val pp of the max, lowest alpha on the plateau wins
ties (v1 selected on val top-1 with an arbitrary tie-break). Recorded as
`alpha_selection=val_top5_csls_v2` in all 12 results configs (6 slots × 2
targets). Both designed behaviors were observed in the re-run: a
signal-driven interior max (Sanskrit Gemma) and a flat-noise plateau rescue
(Akkadian Gemma) — see below.

**Anchors v1 → v2 (guardrail: all six slots passed).**
- Sumerian 13,100 → 13,048 (−52; gw_rejected_v2 72; sources ePSD2 12,355 /
  ETCSL 693)
- Akkadian 24,415 → 24,116 (−299; gw_rejected_v2 8,367)
- Hittite 11,750 → 11,651 (−99; gw_rejected_v2 7,235 German; heterogram
  bridge restored: sux 28 / akk 39 — the /tmp vocab cache had evaporated and
  was regenerated from Sumerian's aligned vocab pkl)
- Greek 106,260 → 105,920 (−340; LSJ join hit rate 77.1%, gate passed)
- Egyptian (06-output basis) raw 8,541 → 4,568 normalized (gw_rejected_v2
  3,602; v1's 4,152 was measured post-09-stopword-filter — a different
  point, not directly comparable)
- Sanskrit 95,924 → 92,275 (−3,649, −3.8%; MW join 94.9%, gate passed)

**Per-slot suites, both targets (top-1 exact %: dict / interpolation /
zero-shot / combined; v1 combined in parens).** All results JSONs carry
`alpha_selection=val_top5_csls_v2`; syn and `gold_oov_candidates` values
live in each slot's `results/alignment_results*.json` under
`test_combined` — see the README v2 table for the gold-OOV column.
- Sumerian GloVe a=10: 77.99/7.20/0.20/4.44 (v1 comb 4.55) | Gemma a=1000:
  75.60/10.13/1.22/6.61 (v1 comb 5.54)
- Akkadian GloVe a=10: 38.47/0.00/0.24/0.18 | Gemma a=0.01:
  55.24/0.18/0.31/0.27 (v1 comb 0.13)
- Hittite GloVe a=1e-4: 67.99/8.90/0.00/5.01 (v1 comb 4.39) | Gemma a=1e-3:
  78.35/13.98/0.00/7.88 (v1 comb 6.83)
- Greek GloVe a=0.1: 37.10/4.43/0.59/3.60 (v1 comb 3.31) | Gemma a=1.0:
  50.11/6.07/0.68/4.91 (v1 comb 4.37)
- Egyptian GloVe a=1e-3: 66.29/25.15/0.00/19.07 (v1 comb 14.81) | Gemma
  a=1e-3: 70.52/26.02/0.92/19.96 (v1 comb 15.47) — biggest v2 winner (noisy
  TLA input cleaned)
- Sanskrit GloVe a=1e-4: 35.16/3.47/0.09/2.95 (v1 comb 2.22) | Gemma a=1e4:
  35.90/5.49/0.46/4.71 (v1 comb 3.83; USER-ACCEPTED TRADE 2026-07-18: real
  interior max at 1e4 trades in-sample dictionary (44.40→35.90) for
  generalization — all test strata up)

**Akkadian-Gemma acid test.** This is the pre-registered acid test for the
alpha-v2 rule. v1's Gemma dictionary accuracy was 19.9% via alpha=1e4, a
flat-noise pick on an entirely flat val sweep (single-anchor margin over the
0.1–1000 plateau). v2's plateau rule, applied to the *same* flat sweep, ties
toward the lowest alpha and picks 0.01 instead — dictionary jumps to
55.24%, combined 0.13%→0.27%. Fires as designed: same underlying signal,
correct-by-construction alpha selection, a large swing in dictionary
accuracy with no change to the eval data — v1's PATHOLOGY is FIXED, not
papered over.

**Procrustes observations (record only, not re-litigated).** New v2 val
cosines: sanskrit 0.1145 → 0.1198 (full) | sumerian 0.1157 → 0.1117 (full) |
hittite 0.0586 → 0.0666 (stable) | greek 0.1149 → 0.1163 (full). All four
remain ≤ 0.12: the v1 retire verdict was pre-registered on the v1 recipe and
stands; v2 values are reported for the record. No v2 cosine exceeds 0.12, so
the flag sentence per spec §5 is not triggered — worth noting factually that
sanskrit's 0.1198 sits closest of the four to the 0.12 band edge.

**A5 hubness diagnostic (Task 11, suite-v2 re-export, ridge plane, measured
2026-07-19).** The anti-hub hypothesis for Gate 2's failure is only
partially confirmed. There is one Greek target, not three — all three
PARALLEL_PAIRS entries match the same "Hesiod Theogony" document, queried by
three different Hittite groups. Its SIF centroid has a strikingly low L2
norm (4.02 vs. pool mean 4.69, std 0.32 — 1.8th percentile), suggesting
near-cancellation from lexical diversity, but its mean cosine to the rest of
the 820-doc Greek pool is only at the 15th percentile (0.853 vs. pool mean
0.885) — below median, not bottom-decile — and its cosine to the 3 Hittite
queries is just 0.79 std below the pool average (0.247 vs. 0.263). Ranks
recomputed in the v2 space (kumarbi-theogony 340, illuyanka-typhon 770,
ullikummi-typhon 733 of 820) are still worse than chance for 2 of 3 pairs
but noticeably better than the v1 numbers that motivated the hypothesis
(731/781/788), with kumarbi-theogony no longer bottom-decile at all.
Verdict: partial — Theogony carries a real geometric anomaly (norm outlier)
but not a clean "far from everything" anti-hub signature strong enough to
fully explain systematically-worse-than-chance ranking; some pair-specific
factor remains unaccounted for. Repro: the analysis script reconstructs the
Gate-2 pool via doc_eval's loaders (importlib on `shared/scripts/doc_eval.py`),
computes pool centroid norms + mean-cosine centrality, and ranks the targets;
~40 lines, session scratchpad `task11_hubness.py` — reconstructable from this
description against doc_eval's parallels entry point.

**A6.** `shared/scripts/doc_eval.py`'s module docstring now carries a dated
note: document-level tokenization there is raw `line.split()` + the slot
normalizer, while FastText corpora pass through each slot's
`05_clean_and_tokenize` — a known, accepted inconsistency of the parked
doc-level plane, not to be "fixed" without re-running Gates 1/2.

## 2026-07-16 — Sanskrit slot shipped: sixth language, and the pre-registered anchor-quality read-out FIRES.

**Slot summary.** DCS (Digital Corpus of Sanskrit, CC BY 4.0): 15,790 chapter
files parsed / 754,502 lines / 5,679,462 token-lemma records / 90,184 unique
lemmas, parse loss 0.000% (0 of 6,713,257 token lines). Monier-Williams
(Cologne CDSL 2020 digitization, mw.xml; CC BY-NC-SA 3.0 per mwheader.xml):
177,323 deduplicated entries. Cleaned corpus: 15,790 lines / 5,679,462 tokens
/ 381,412 unique tokens. FastText vocab (min_count=2): 195,309; fused
195,309 x 1536d. Anchors (Task 6): token-level DCS-lemma→MW join hit rate
94.9% (5,391,784 hits / 287,678 misses), 40% gate passed; gloss_no_eng
145,119; total anchors 95,924; valid at fit time 90,176.

**Disclosure.** Two deliberate deviations from the Greek/LSJ recipe, both
user-approved 2026-07-14. (1) **06's negation-gloss rule:** glosses hitting
a negator (not/no/without/never) before an in-vocab content word are
skipped entirely and fall through to the next gloss segment — rationale is
survey finding A1, driven by MW's privative compounds (e.g. `ahiṃsā`:
"not injuring anything" anchors to "harmlessness", never to "injuring").
This is the only extraction-recipe deviation from Greek/LSJ. (2) 05 is a
new thin IAST tokenizer, not a clone of the Greek ATF-cleaner — the spec's
own architecture section describes 05 as "sanskrit_normalize on FORM
stream," which the ATF machinery doesn't implement. Separately, the anchor
noise profile was kept as-is per user decision 2026-07-15: 3,379 anchors →
"see" (MW cross-references, 3.5%), 2,279 single-letter English tokens
(2.4%), and scaffold words in the frequent tail (having/relating/rarely/
belonging) — an inherited Greek-recipe noise class, kept for
cross-slot comparability rather than cleaned up specially for Sanskrit.

**Word-level suite** (Task 9; seed 42, lemma-group split with near-surface
edges, 50,000 candidates). Split: train 61,391 / val 12,490 valid / test
16,295 valid (raw 15,348/19,185); oov_train_only 3,557. Leak check: 0.00%
(0/19,185, shared gloss + surface edit distance ≤1 vs train).

| Target | alpha | Dict n | Dict top-1 | Dict syn | Interp n | Interp top-1 | Interp syn | Zero-shot n | Zero-shot top-1 | Zero-shot syn | Combined n | Combined top-1 | Combined syn |
|--------|-------|-------:|:----------:|:--------:|---------:|:-------------:|:----------:|------------:|:----------------:|:--------------:|-----------:|:---------------:|:------------:|
| GloVe 300d | 1.0 | 955 | 33.51% | 35.92% | 12,863 | 2.55% | 4.12% | 2,168 | 0.23% | 1.38% | 15,031 | 2.22% | 3.73% |
| Gemma whitened 768d | 1000 | 955 | 44.40% | 47.23% | 12,863 | 4.35% | 6.84% | 2,168 | 0.74% | 2.26% | 15,031 | 3.83% | 6.18% |

Gemma dictionary top-5/top-10: 56.75%/58.01% exact, 59.27%/60.94% syn;
combined top-5/top-10: 7.49%/9.75% exact, 11.28%/14.58% syn. GloVe combined
`gold_oov_candidates`: 1,264. Gemma beats GloVe combined: +1.62pp top-1 /
+2.14pp top-5 / +2.60pp top-10 (now 5 of 6 slots).

**Procrustes read-out** (Task 10, commit a5ecc08). Sanskrit joins the
semi-orthogonal-plane anchor-quality test as the stronger-anchors condition
(94.9% hit rate vs the other slots' thinner joins). Variants: full 0.11451
(n=57,834) / stable 0.11422 (n=53,538); chose full; n_fit 70,324; isometry
rho 1.0000; target `english_gemma_whitened_768d`. Val cosine **0.1145**
falls in the pre-registered **≤ 0.12** band (the existing slots' band or
below ⇒ anchors were never the constraint). Bands, restated from the
2026-07-13 design spec: val cosine ≥ 0.20 ⇒ anchor quality was a binding
constraint and the stronger-anchors lever stays live; ≤ 0.12 ⇒ anchors were
never the constraint; between 0.12 and 0.20 ⇒ inconclusive. Pre-registered
verdict, applied verbatim per the spec: **"anchors were never the
constraint ⇒ retire the stronger-anchors lever, and with it the last named
route to Plane A."** Reference slots: sumerian 0.1157 / greek 0.1149 /
hittite 0.0586; sanskrit 0.1145 — three-slot convergence (sumerian, greek,
sanskrit) at ~0.115 despite very different corpus and lexicon quality
suggests a structural ceiling on the semi-orthogonal plane, not anchor
quality; hittite's lower 0.0586 remains unexplained by this test.

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
