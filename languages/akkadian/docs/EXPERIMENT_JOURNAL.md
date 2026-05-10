# Akkadian Experiment Journal

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
