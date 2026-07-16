# Suite v2 Canonical Cleanup — Design

**Date:** 2026-07-16
**Status:** Approved
**Goal:** Promote the measured anchor-quality fixes into the canonical recipe
(shared gloss-filter module), fix alpha selection's noise-regime failure mode,
re-run all six slots as **suite v2** with the v1 table archived, and close out
the number-neutral survey items (A3 disclosure, A5 hubness diagnostic, A6
documentation) — so the seventh slot (Coptic) clones a clean canonical.

## Background

The 2026-07-14 analysis survey and the Sanskrit slot build measured four noise
classes in the anchor-extraction recipe shared by all slots: negated glosses
harvested as antonym anchors (A1, fixed in Sanskrit's 06 only), cross-reference
glosses anchored to "see" (3.5% of Sanskrit anchors), single-letter English
matches (2.4%), and scaffold words harvested from gloss prose. Separately,
alpha selection on val top-1 CSLS was observed (Akkadian-Gemma, journal
2026-07-09ff) picking α=10⁴ off a flat-noise plateau by a one-anchor margin,
crushing its dictionary stratum. These fixes change published numbers, so they
ship together as a versioned re-run of all six slots.

The suite-v1 verdicts are not re-litigated: the Sanskrit Procrustes read-out
(journal 2026-07-16) was a pre-registered experiment on the v1 recipe and its
retire verdict stands as recorded.

## Scope

- New `shared/scripts/gloss_filters.py` + tests; all six slots' 06 scripts
  refactored to import it (source-specific joins stay per-slot).
- Alpha-selection change in canonical Akkadian 09/09b, sed-refreshed to clones.
- Full re-run, all six slots, both targets, plus per-slot Procrustes refit and
  production re-export; HF artifact mirror updated incrementally.
- README suite table v2 (v1 archived); journal entry; A3 disclosure column.
- A5 read-only Gate-2 hubness diagnostic (journal-only); A6 documentation note.

**Out of scope (recorded):** doc_eval code changes (Gate 1/2 stay as measured
on v1; A6 is documentation-only); MIN_OCCURRENCES tuning; Coptic slot (next
project, clones the v2 canonical); myth-study reruns; any new pre-registered
verdict on the retired stronger-anchors lever.

## Architecture

### 1. `shared/scripts/gloss_filters.py` — single source of anchor-English selection

```python
NEGATORS = {"not", "no", "without", "never"}
XREF_STARTERS = {"see", "cf", "vid"}   # gloss is a cross-reference, not a meaning
# ("q.v." never surfaces as a token under the word regex; the 02 parsers'
#  noise-segment filters already drop bare "q.v" gloss segments)
SCAFFOLD_WORDS = {                             # gloss prose, never a meaning
    "having", "relating", "belonging", "rarely", "who", "whose", "being",
    "one's", "various", "especially", "particularly", "chiefly",
    "generally", "usually",
}
STOP_WORDS = { ... }        # moves here verbatim from the slot 06s ("not"/"no" excluded)
MIN_HIT_RATE = 0.40

def first_english(gloss: str, eng_vocab: set[str]) -> str | None:
    """First in-vocab content word of `gloss`, or None if the gloss is
    unusable. Rules, in scan order over content words:
      - negator encountered before a match      -> None (whole gloss rejected)
      - first content word in XREF_STARTERS     -> None (cross-reference gloss)
      - word in STOP_WORDS or SCAFFOLD_WORDS    -> skip, continue scanning
      - single-letter word                      -> skip, continue scanning
      - word (or hyphen-joined form) in vocab   -> return it
    Callers fall through to the entry's next gloss on None (existing
    glosses[1:5] convention, unchanged)."""

def hit_rate_stats(hits: int, misses: int, gloss_no_eng: int, anchors: int) -> dict:
    """Canonical anchor_stats payload: {*_hits, *_misses, token_hit_rate,
    gloss_no_eng, anchors}. Persisted by every 06 before any gate check."""

def check_hit_rate_gate(stats: dict, source_name: str) -> None:
    """SystemExit if token_hit_rate < MIN_HIT_RATE (stats already persisted)."""
```

Notes: high-frequency verbs that are genuine glosses ("go", "act", "make")
are deliberately NOT scaffold words. Single-letter handling is skip-and-
continue (a later real word can still anchor the gloss), unlike negation/xref
which reject the whole gloss.

### 2. Per-slot 06 refactor

Each `languages/<slot>/scripts/06_extract_anchors.py` (and equivalent) drops
its local STOP_WORDS/`_load_gloss_first_english` and calls
`gloss_filters.first_english`; join logic, sources, schemas, and
`MIN_OCCURRENCES = 5` are unchanged. Every 06 now persists `anchor_stats.json`
and runs the gate (Sanskrit behavior promoted). Sanskrit's own 06 is
refactored to import the module; its negation unit tests move to the shared
test file, its source-specific tests stay.

### 3. Alpha selection v2 — canonical 09/09b + clones

In `select_alpha` (canonical `languages/akkadian/scripts/09_align_and_evaluate.py`
and `09b_align_gemma.py`):
- score each alpha by **val top-5 CSLS** (candidate pool and CSLS parameters
  unchanged);
- treat scores within one anchor's worth of the maximum (< 1/n_val) as a
  plateau and pick the **lowest alpha** on it;
- record `"alpha_selection": "val_top5_csls_v2"` and the full sweep in the
  results JSON.
Clones (hittite, greek, sanskrit; sumerian/egyptian variants with their
documented deltas) are refreshed by the established sed-clone + diff-gate
pattern. Suite scoring itself (eval_suite.py) is untouched — v2 changes what
alpha is picked, never how test metrics are computed.

### 4. Re-run orchestration

- **Before any re-run:** tag the HF artifact repo revision as `suite-v1`.
- Per slot, sequential, detached: 06 → **anchor-count guardrail** → 09 → 09b
  (whitened) → 10 export → `procrustes_align.py --slot <slot>` → incremental
  `hf upload` of the slot's refreshed dirs.
- **Guardrail:** if a slot's v2 anchor count drops more than 15% vs v1
  (v1 counts recorded in the plan), stop and surface before alignment compute.
  Hittite is the expected fragile case.
- Slot order: **sanskrit first** (best-known v1 baseline, validates the
  module refactor end-to-end), then **akkadian** (the alpha-fix acid test:
  its Gemma dictionary stratum should recover toward the GloVe-pick level),
  then sumerian, hittite, greek, egyptian.

### 5. Reporting

- README: new v2 suite table (same columns plus a `gold OOV` disclosure
  column per A3); the v1 table moves below it, dated and labeled, following
  the existing archived-table precedent.
- Journal entry: v2 recipe deltas (module, filters, alpha rule), per-slot
  before/after numbers, new Procrustes val cosines **reported against the
  v1 bands as observation only** — wording fixed in advance: "the v1 retire
  verdict was pre-registered on the v1 recipe and stands; v2 values are
  reported for the record." If any v2 val cosine exceeds 0.12, the journal
  flags it as a question for a future, separately pre-registered decision —
  it does not reopen the verdict by itself.
- A3: `gold_oov_candidates` surfaced in every published accuracy table with
  one caption sentence stating accuracies are conditioned on golds present
  in the 50k candidate vocabulary.

### 6. Diagnostics (number-neutral)

- **A5 hubness diagnostic:** one read-only script/analysis over the existing
  Gate-2 artifacts: Greek doc-pool centroid norm distribution, mean-cosine
  centrality (hub/anti-hub ranking), and where Theogony/Typhon docs sit in
  it; tests whether "true parallels systematically bottom-11%" is an
  anti-hub artifact. Output: journal note only; no benchmark change. Not
  committed to results (repro commands in the journal note).
- **A6:** docstring note in `shared/scripts/doc_eval.py` + one journal
  sentence documenting that doc-eval tokenizes raw lines (`split()` +
  normalizer) while FastText corpora pass through each slot's 05 — a known,
  accepted inconsistency of the parked doc-level plane.

## Error handling

Repo conventions: fail loudly. The 06 gate exits after persisting stats; the
guardrail check is a stop-and-surface, not a skip; sed-diff gates must be
byte-empty; alpha sweeps print in full to run logs.

## Testing

- `shared/tests/test_gloss_filters.py`: negation fall-through and whole-gloss
  rejection, xref rejection ("see kṛṣṇa" → None), scaffold skip-and-continue
  ("having horns" → "horns"), single-letter skip ("c blade" → "blade"),
  stop-word behavior parity with v1 for plain glosses, stats/gate math.
- Per-slot 06 tests updated to the import; Sanskrit negation tests relocated.
- Sed-diff gates for 09/09b clone refresh; full pytest green per commit.
- End-to-end validation = the Sanskrit re-run (first slot) compared line-by-
  line against its v1 numbers before the remaining five launch.

## Docs & workflow

Branch `suite-v2`; spec → plan (superpowers:writing-plans) → subagent-driven
execution. HF mirror updates are incremental per slot; `suite-v1` tag pinned
first. Root README, journal, and memory updated at ship.
