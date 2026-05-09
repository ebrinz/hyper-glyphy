# Akkadian Experiment Journal

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
