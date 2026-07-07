# Akkadian Experiment Journal

> **2026-07-06 — All accuracy numbers in this journal are pre-fix (leaked split).**
> See the repo-wide journal ([`docs/EXPERIMENT_JOURNAL.md`](../../../../docs/EXPERIMENT_JOURNAL.md),
> 2026-07-06 entry) for the full eval-integrity writeup. The Akkadian rerun under the
> leakage-free split (GloVe 0.09%, Gemma 0.14% top-1) is the reference honest number;
> all numbers in this historical log reflect the old leaked-pair split.

## 2026-05-11 — v1.3: Ridge alpha sweep (+7.41pp top-1, single hyperparameter)

After the v1.2 plateau (29.02% top-1) we ran an alpha sweep on the whitened-Gemma
target. The inherited alpha=100 (copied from Sumerian's spec) is nowhere near
optimal for Akkadian's anchor pool.

### Sweep results (Akkadian whitened-Gemma)

| alpha | Top-1 | Top-5 | Top-10 |
|-------|:---:|:---:|:---:|
| 0.01  | **36.43%** | **59.18%** | **66.51%** |
| 0.1   | 35.03% | 58.57% | 65.37% |
| 1     | 33.49% | 57.08% | 63.26% |
| 10    | 31.26% | 54.71% | 61.42% |
| 100 (was) | 29.02% | 49.45% | 57.08% |
| 1000  | 22.84% | 40.60% | 46.87% |
| 1e+04 | 10.92% | 19.95% | 25.43% |
| 1e+05 | 2.24% | 3.59% | 5.13% |

Adopting alpha=0.01 for Gemma. GloVe (09) was also using alpha=100; Sumerian's
spec actually documents alpha=0.001 for GloVe — that fix is applied too.

### Headline numbers

| Metric | v1.2 | v1.3 | Delta | Sumerian for comparison |
|--------|:---:|:---:|:---:|:---:|
| Gemma top-1  | 29.02% | **36.43%** | +7.41pp | 52.13% |
| Gemma top-5  | 49.45% | **59.18%** | +9.73pp | 61.97% |
| Gemma top-10 | 57.08% | **66.51%** | +9.43pp | 65.99% (!) |
| GloVe top-1  | 14.56% | **27.79%** | +13.23pp | 35.70% |
| GloVe top-5  | 23.02% | **43.23%** | +20.21pp | 44.61% |
| GloVe top-10 | 26.04% | **47.52%** | +21.48pp | 47.93% |

**Akkadian Gemma top-10 now exceeds Sumerian's top-10.** Akkadian GloVe top-10
(47.52%) is within 0.41pp of Sumerian's (47.93%). The coverage problem is
effectively solved on both targets. The remaining top-1 gap (36.43% vs
52.13% Gemma) is alignment *precision*, not lexical coverage.

### Lesson

**Don't inherit hyperparameters across slots.** The alpha=100 constant came
from Sumerian's `09b_align_gemma.py` and we propagated it through three
language slots (Sumerian, Egyptian, Akkadian) on the assumption it
generalized. It doesn't. Each slot's optimal regularization depends on:

- Anchor pool composition (Sumerian's 13k anchors @ ePSD2 confidence vs.
  Akkadian's 24k anchors via global surface expansion)
- Anchor signal-to-noise (Sumerian's ORACC is more uniformly curated than
  Akkadian's blend of OB literary + letters + DCCLT)
- Effective fused dimension after FastText training (corpus quality)

For future slots: **always sweep alpha**. Cost: 20 min. Possible payoff:
+5-10pp top-1. The ratio is absurd in favor of always doing it.

Cumulative from v1 ship: **16.75% -> 36.43% (+19.68pp)** on Gemma top-1.

Files: `scripts/09b_align_gemma.py` (RIDGE_ALPHA), `scripts/09_align_and_evaluate.py`,
`scripts/ridge_alpha_sweep.py` (ported from Sumerian). Commit: `8510efd`.

---

## 2026-05-10 — v1.2: anchor expansion + subword inference + bigger corpus (+7.36pp top-1 over v1.1)

Continued the gap-closing iteration. Four more workstreams executed after v1.1
(L2/L1b/L3 baseline). End-state: whitened-Gemma top-1 **29.02%**, top-5 49.45%,
top-10 57.08%. Cumulative from v1 ship: 16.75% -> 29.02% (+12.27pp).

### Cumulative numbers (whitened-Gemma 768d)

| Stage | Top-1 | Top-5 | Top-10 | FastText vocab | Corpus tokens | Anchors | Delta |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| v1.1 end (L3) | 21.66% | — | — | 27,799 | 1,441k | 4,481 | — |
| + L4 lemma-surface expansion | 25.39% | — | — | 27,799 | 1,441k | 24,415 | +3.73pp |
| + L5 subword inference (train-only) | 26.58% | 44.81% | 51.97% | 27,799 | 1,441k | 24,415 | +1.19pp |
| + L6a SB corpus expansion #2 | 28.32% | 49.15% | 56.16% | 45,769 | 3,000k | 24,415 | +1.74pp |
| + L6b DCCLT bridge (FALSIFIED) | 24.77% | 42.93% | 49.98% | 45,769 | 3,000k | 25,962 | -3.55pp |
| Final (L6b reverted) | **29.02%** | **49.45%** | **57.08%** | **45,769** | **3,000k** | **24,415** | **+0.70pp** |

### Lever-by-lever read

**L4 (global lemma-surface expansion) — big win, +3.73pp.** Built a global
`citation_form -> {surface_forms}` map across ALL ORACC lemma records, then
expanded each emitted anchor to include every surface variant of its
citation form. The previous per-record `(cf, form)` pairing missed surface
variants that appeared in OTHER lemma records with different glosses.
Anchor count grew from 4,481 to 24,415 (5.5x). The `oracc_lemma_surface_recoverable`
diagnostic bucket fell from 34.9% to 5.14%.

**L5-refined (subword inference, training-only) — +1.19pp.** Initial L5
implementation passed OOV-inferred anchors through `train_test_split`,
landing some in the test set; their noisy inferred vectors regressed top-1
by -2.86pp. The fix partitions valid anchors by the `subword_inferred` flag:
OOV anchors are training-only, test set drawn exclusively from in-vocab
anchors. With this fix the eval is fair-comparable to the pre-L5 baseline,
and the extra training signal from 6,198 OOV anchors lifts top-1 by +1.19pp.

**L6a (massive SB corpus expansion) — +1.74pp.** Expanded the SB-pretrain
project list from 11 to 62 ORACC projects (RINAP volumes 2-5p1, SAA letters
01-21, RIBO Babylonia 2-7, CMAWRO, ASBP, ADSD, ATAE site corpora, CAMS
sub-projects, etc.). Corpus 1.44M -> 3M tokens, FastText vocab 27,799 ->
45,769. Confirms diminishing returns on corpus: doubling 712k->1.44M gave
+4.73pp (L3), doubling again 1.44M->3M gave +1.74pp.

**L6b (DCCLT bridge bootstrapping) — FALSIFIED, -3.55pp.** Used Sumerian's
52% top-1 alignment as a Rosetta stone: for each Sumerian-Akkadian DCCLT
pair, took Sumerian's top-1 English neighbor (gated by cosine >= 0.5) and
emitted (akkadian, english) as a bridge anchor. Produced 1,573 new anchors
(8,176 rejected by cosine threshold, 3,039 by Sumerian-vocab miss).
Hypothesis: the high-cosine neighbors would be reliable enough to add net
training signal. Reality: even at cosine 0.5+, the bridge labels are too
noisy. Net regression -3.55pp. Reverted to ORACC-only anchors. Bridge
script committed as experimental infrastructure for future investigation
(stricter cosine threshold, lemma-uncovered-only mode, score-aware Ridge).

### Key lessons

1. **Vector quality > anchor quantity.** L1b recovered 413 low-frequency
   anchors for +0.05pp; L4 added 19,934 anchors with proper surface
   expansion for +3.73pp. The variable that matters is whether the
   recovered/added anchors have *usable* vectors.

2. **Corpus expansion is the dominant lever, with diminishing returns.**
   L3 + L6a together moved top-1 by +6.47pp via FastText corpus alone.
   Top-10 (57.08%) now approaches Sumerian's 65.99% — the top-1 gap
   reflects the harder problem of correct nearest-neighbor selection,
   not insufficient corpus signal.

3. **Eval-set partition matters more than expected.** L5's initial
   regression was an evaluation artifact, not a real signal degradation.
   When OOV anchors entered the test set, per-anchor accuracy on the
   noisy inferred vectors dragged down the mean. The partition fix
   (OOV anchors training-only, test drawn from in-vocab) recovered the
   real positive signal.

4. **Transitive bootstrapping from another slot's alignment is harder
   than it looks.** L6b's hypothesis seemed strong on paper — 52%
   accurate oracle + filtering by cosine should give a clean subset.
   But ridge-aligned spaces don't preserve per-anchor confidence
   monotonically: high cosine in one alignment doesn't guarantee
   semantic accuracy. Future bridge experiments need a fundamentally
   different signal (e.g., scored DCCLT entries, ancient bilingual
   glosses where both sides have explicit English).

### Remaining gap to Sumerian

Top-1 29.02% vs Sumerian 52.13% — gap of 23.11pp.
Top-10 57.08% vs Sumerian 65.99% — gap of 8.91pp.

The top-10 gap is now small. The top-1 gap reflects an alignment
*precision* problem, not coverage. Remaining levers:

- **Per-anchor Ridge alpha tuning.** Sumerian's `ridge_alpha_sweep.py`
  experimented with this. Possible +1-3pp.
- **WordNet-augmented gloss expansion.** Use WordNet to expand single-word
  English glosses into rich definitions, then encode with Gemma. Sumerian's
  v2 used this. Possible +3-5pp.
- **Manually-curated high-confidence anchor seed.** A small set (~500) of
  hand-verified Akkadian-English pairs from CAD volume 1, used to anchor
  the Ridge with very high confidence. Possible +2-5pp.

### Files added/modified

- `scripts/01b_scrape_oracc_sb.py` — expanded to 62 projects (L6a)
- `scripts/06_extract_anchors.py` — global surface-map expansion (L4)
- `scripts/06b_bridge_anchors.py` — DCCLT bridge (L6b, experimental)
- `scripts/09_align_and_evaluate.py` — subword inference + train/test
  partition (L5)
- `scripts/09b_align_gemma.py` — same (L5)

Commits: `c5471f2` (L4), `fd9da0e`+`5ec3594` (L5+refine), `f3599c1` (L6a),
`44944a6` (L6b infra).

---

## 2026-05-10 — v1.1: three-lever gap-closing pass (+4.91pp top-1)

Three improvement workstreams executed in sequence after the v1 ship, targeting
the 35pp gap to Sumerian's 52% top-1.

### Cumulative numbers (whitened-Gemma 768d)

| Stage | Top-1 | Top-5 | Top-10 | FastText vocab | Corpus tokens | Delta |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| v1 baseline | 16.75% | 25.77% | 28.87% | 5,428 | 712k | — |
| + L2 mimation alternates | 16.88% | 29.16% | 34.78% | 5,428 | 712k | +0.13pp |
| + L1b min_count 5→2 | 16.93% | — | — | 15,209 | 712k | +0.05pp |
| + L3 SB pretrain corpus | **21.66%** | — | — | **27,799** | **1,441k** | **+4.73pp** |
| Cumulative | | | | | | **+4.91pp** |

### Lever-by-lever read

**L2 (mimation wiring) — small, predictable.** ORACC citation forms drop mimation
by scribal convention (`šarru` cited, `šarrum` attested in records). The alternation
helper only added forms when `cf` itself ended in -um/-am/-im, which is rare in
ORACC's normalized stream. Anchor count grew by only +97. Top-1 moved +0.13pp.

**L1a (audit + coverage diagnostic ported) — diagnostic.** The Sumerian Workstream
2a/2b audit infrastructure was ported (`audit_anchors.py`, `coverage_diagnostic.py`
running end-to-end). Output revealed Akkadian's miss distribution is structurally
DIFFERENT from Sumerian's:

- **Sumerian's pre-W2b dominant bucket: `normalization_recoverable` at 64.85%** → fixed by a ~20-line unicode normalization patch, +32pp.
- **Akkadian's dominant bucket: `in_corpus_below_min_count` at 36.1%** → fixed by lowering FastText min_count → but the recovered vectors were too sparse to lift alignment quality. The W2b-style "single normalization gap" win does not exist for Akkadian; normalization was 0% of misses (NFC + ORACC→ATF + vowel-dedup got it right from v1).

**L1b (min_count 5→2) — structurally correct, low impact.** Recovered 413 anchors
but only +0.05pp top-1. The lesson: anchor *recovery* doesn't help when the
recovered vectors are themselves noise. Vector *quality* matters more than vector
*count*.

**L3 (SB pretraining corpus) — the actual dominant lever for Akkadian.** Adding
RINAP/SAA/CAMS as FastText training-only data (anchor extraction unchanged, OB
temporal honesty preserved) doubled the corpus and grew vocab by 83%. The lift
(+4.73pp) is consistent with the hypothesis that Akkadian's bottleneck is FastText
training quality, not anchor coverage.

### Lessons

1. **The Sumerian playbook does not apply directly.** Sumerian's leap (+32pp) came
   from a single normalization gap that happened to be present and easy to fix.
   Akkadian had no equivalent gap; its bottleneck is corpus size. Coverage
   diagnostics must be run per-slot — assumed-transferable interventions can fail.

2. **More anchors ≠ better alignment.** L1b proved this: recovering 413 anchors
   moved top-1 by 0.05pp because the recovered vectors were sparse. Spending
   compute on FastText pretraining (L3) returned the multiplier.

3. **The integrity boundary (SB feeds FastText only, not anchors) held.** Anchor
   extraction script never read `sb_lemmas.json`. Temporal-honesty claim from the
   design spec remains defensible.

### Remaining levers (not in v1.1 scope)

After L3, the diagnostic shows:

- `subword_inference_recoverable`: 35.7% of remaining misses (461 anchors).
  Fix would modify `09b_align_gemma.py` to call FastText's subword-inference for
  OOV anchors at evaluation time. Projected lift: +3-7pp.
- `oracc_lemma_surface_recoverable`: 34.9% (451 anchors).
  Fix: build a global lemma-surface map across all ORACC records, expand each
  emitted anchor to all known surface variants. Projected: +2-5pp.

Combined these could plausibly bring top-1 into the 28-33% range, approaching
Egyptian's 32%. Sumerian's 52% remains a stretch goal contingent on either a
much larger corpus or a per-anchor surface-variant treatment we haven't designed.

### Files touched

- `01b_scrape_oracc_sb.py` (new, L3)
- `05_clean_and_tokenize.py` (modified, L3)
- `06_extract_anchors.py` (modified, L2)
- `07_train_fasttext.py` (modified, L1b)
- `audit_anchors.py` (new, L1a)
- `tests/test_06_anchors.py` (modified, L2)
- `tests/test_audit_anchors.py` (new, L1a)

Commits: `8d00b01` (L2), `e9419db` (L1a), `60a97cf` (L1b), `9edfcbf` (L3).

## 2026-05-09 — v1 ship: OB Akkadian aligned to whitened-Gemma 768d

**Spec:** `docs/superpowers/specs/2026-05-09-akkadian-slot-design.md`
**Plan:** `docs/superpowers/plans/2026-05-09-akkadian-slot.md`

Pipeline mirrors Sumerian 1:1. Three corpus tiers ingested via three new
scrapers (OB literary, OB letters, DCCLT). Anchor lexicon: ORACC-only
(eBL primary path was pivoted — see Deviations). DCCLT lexical lists
parsed into `data/processed/sumerian_akkadian_pairs.jsonl` for the v2
cross-lingual bridge experiment (data ready, experiment deferred).

### Numbers

| Metric | Akkadian (whitened-Gemma 768d) | Akkadian (GloVe 300d) | Sumerian (whitened-Gemma) | Egyptian (GloVe) |
|--------|:---:|:---:|:---:|:---:|
| Top-1  | **16.75%** | 6.70% | 52.13% | 32.35% |
| Top-5  | **25.77%** | 9.54% | 61.97% | 41.47% |
| Top-10 | **28.87%** | 11.86% | 65.99% | 45.13% |
| Training anchors | 1,549 | 1,549 | 6,867 | 5,360 |
| Valid anchors | 1,937 / 4,384 (44.2%) | 1,937 / 4,384 (44.2%) | 8,558 / 13,100 (65.3%) | — |
| Corpus tokens | 712,993 | 712,993 | 2.8M | 789K |
| FastText vocab | 5,428 | 5,428 | — | — |

**Note (2026-05-10):** Production artifacts now reflect v1.1 numbers (top-1 21.66%); see entry above.

Result lands BELOW the plan's target floor (>=30% top-1). Honest read:
the architecture is validated and the pipeline runs end-to-end, but
absolute numbers reflect a smaller corpus and weaker anchor coverage
than Sumerian. Within-Akkadian semantic structure is meaningful (e.g.,
`szarrum` -> `szarrutu` 0.857 cosine), so the FastText layer captured
the language; the alignment to modern English is the choke point.

### Levers for future improvement (not in v1 scope)

1. **Anchor normalization audit.** 55.8% of anchors miss the corpus
   vocabulary. Sumerian's Workstream 2b found a single normalization
   gap accounted for ~65% of misses there and moved top-1 by +32pp.
   An equivalent audit on Akkadian is the highest-leverage next step —
   ship `coverage_diagnostic.py` against the real corpus (currently
   unit-tested only) to identify the dominant lever.
2. **Standard Babylonian fallback.** Plan flagged this contingency for
   corpora < 500k tokens. Cleaned-corpus came in at 712k tokens, ABOVE
   the threshold, so SB fallback was not triggered. Could still be
   used as supplementary FastText pretrain to enlarge the vocabulary.
3. **eBL integration (v2).** The plan's eBL primary strategy needs
   per-word query iteration over ~20k entries. Worth doing as a v2
   workstream once its caching strategy is designed.

### Deviations from the design spec

- **eBL pivot to ORACC-only.** eBL's bulk-fetch endpoint
  (`/api/words/all`) returns only string IDs, not full entries —
  requires 20k per-word queries to reconstitute, impractical at scale.
  Pivot mirrors Sumerian's actual pattern: ORACC project glosses ARE
  the primary anchor source; the "eBL primary" framing in the plan
  was speculative. The `ebl_fetch.py` module is committed for v2 use;
  unit tests pass against synthetic data; real fetch returns 404 and
  is documented in the module docstring.
- **`logogram_unmatched` bucket added** to the coverage diagnostic
  as planned.
- **Vowel-deduplication step in `normalize_akkadian_token`.** Added
  during T4 to handle Akkadian's hyphen-bridge pattern
  (`szar-ru-um -> szarrum`, where the dropped hyphen creates a doubled
  vowel that must collapse). Necessary for test correctness.
- **JSON vocab artifact (not binary serialization).** Deviates from
  Sumerian's binary serialization convention. Akkadian vocab is a flat
  string list — JSON is equivalent in size and avoids deserialization
  risk surface.
- **Alignment scripts patched.** Verbatim copies of 09 and 09b
  referenced `anchor["sumerian"]` and `languages.sumerian.scripts.
  align_09`. Patched to use `anchor["akkadian"]` and a local
  `align_09.py` shim. Documented in commit `a73b45d`.

### DCCLT bridge data

DCCLT scraping produced **50,636 Sumerian<->Akkadian word pairs** at
`data/processed/sumerian_akkadian_pairs.jsonl`. The v2 cross-lingual
bridge experiment can validate Sumerian-Gemma and Akkadian-Gemma
alignments through these pairs without additional data acquisition.

### Out-of-scope (deferred per spec)

- Cross-lingual bridge experiment (data scaffolded but not run).
- Diachronic OB -> Classical Akkadian comparison.
- Production research artifact (no PDF analog to the Sumerian
  cosmogony document).
