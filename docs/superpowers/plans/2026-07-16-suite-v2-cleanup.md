# Suite v2 Canonical Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the measured anchor-noise fixes into a shared `gloss_filters` module consumed by all six slots, fix alpha selection (val top-5 CSLS, low-alpha plateau ties), re-run all six slots as suite v2 with the v1 table archived, and land the number-neutral items (A3 disclosure, A5 hubness diagnostic, A6 note).

**Architecture:** New `shared/scripts/gloss_filters.py` with two entry points — `first_english()` for the dictionary-join slots (greek, sanskrit) and `gw_is_usable()` for the value-based slots (sumerian ePSD2, akkadian, hittite with German negators, egyptian) — plus stats/gate helpers. Alpha-v2 lands once in canonical Akkadian 09 (09b shares it via the `align_09` shim) and propagates to hittite/greek/sanskrit by sed-refresh; sumerian/egyptian variants get the same hunks in place. Re-runs are sequential detached jobs per slot with an anchor-count guardrail between 06 and the alignment compute.

**Tech Stack:** Python 3.12, numpy, scikit-learn Ridge, gensim, pytest; `hf` CLI for artifact mirroring.

**Spec:** `docs/superpowers/specs/2026-07-16-suite-v2-cleanup-design.md` (incl. 2026-07-16 two-family amendment)

## Global Constraints

- Branch `suite-v2`. All commands from repo root. Full `pytest -q` green before every commit. Data/models/results are gitignored — never committed.
- Shared constants (exact values): `NEGATORS = {"not", "no", "without", "never"}`; `DE_NEGATORS = {"nicht", "kein", "keine", "keinen", "ohne", "nie", "niemals"}`; `XREF_STARTERS = {"see", "cf", "vid"}`; `SCAFFOLD_WORDS = {"having", "relating", "belonging", "rarely", "who", "whose", "one's", "various", "especially", "particularly", "chiefly", "generally", "usually", "being"}`; `MIN_HIT_RATE = 0.40`. STOP_WORDS = the Greek 06 set verbatim MINUS "not"/"no" (they move to NEGATORS).
- `MIN_OCCURRENCES = 5` and all join/source logic per slot unchanged. Suite scoring (eval_suite metrics) unchanged — v2 changes anchor content and alpha choice only.
- Alpha-v2 rule: select on **val top-5 CSLS exact**; scores within `100.0/n_val` percentage points of the max form a plateau; pick the **lowest** alpha on the plateau. Results JSON records `"alpha_selection": "val_top5_csls_v2"` and the full sweep (top1 + top5 per alpha).
- Anchor-count guardrail (fail if v2 total anchors < 85% of v1): sumerian 13,100 → min 11,135; akkadian 24,415 → 20,753; hittite 11,750 → 9,988; greek 106,260 → 90,321; egyptian 4,152 → 3,529; sanskrit 95,924 → 81,536. Below the floor = stop-and-surface, never proceed to 09.
- Re-run order: sanskrit (bellwether) → akkadian (alpha acid test) → sumerian → hittite → greek → egyptian. Each slot: 06 → guardrail → 09 → 09b `--mode whitened` → 10 → `procrustes_align.py --slot <slot>` (egyptian: no procrustes — it is not in SLOTS; sumerian/egyptian 09b flags per their READMEs). Runs detached (`nohup`), logs under `languages/<slot>/logs/`.
- HF mirror: tag `suite-v1` BEFORE the first re-run; after each slot completes, incremental `hf upload` of that slot's refreshed dirs (models, data/processed, results, final_output) with the same include/exclude conventions as the initial mirror (allowlist `fasttext_* fused_* ridge_* *procrustes*` for models; `--exclude "glove*"` for non-sumerian processed).
- The v1 Procrustes retire verdict is NOT re-litigated. v2 val cosines are reported against the v1 bands as observation only, wording per spec §5.
- Sed-refresh gates: regenerated clones must diff-empty against their derivation; `python -m py_compile` clean.

---

### Task 1: `shared/scripts/gloss_filters.py` + tests

**Files:**
- Create: `shared/scripts/gloss_filters.py`
- Test: `shared/tests/test_gloss_filters.py`

**Interfaces:**
- Produces: `first_english(gloss: str, eng_vocab_set: set[str], negators: frozenset = NEGATORS) -> str | None`; `gw_is_usable(value: str, negators: frozenset = NEGATORS) -> bool`; `hit_rate_stats(hits: int, misses: int, gloss_no_eng: int, anchors: int) -> dict` (keys: `hits, misses, token_hit_rate, gloss_no_eng, anchors`); `check_hit_rate_gate(stats: dict, source_name: str) -> None` (SystemExit below gate); constants `NEGATORS, DE_NEGATORS, XREF_STARTERS, SCAFFOLD_WORDS, STOP_WORDS, MIN_HIT_RATE`.
- Consumed by: Tasks 2–6 (all six 06 scripts).

- [ ] **Step 1: Write the failing tests**

`shared/tests/test_gloss_filters.py`:

```python
import pytest

from shared.scripts.gloss_filters import (
    DE_NEGATORS,
    MIN_HIT_RATE,
    NEGATORS,
    STOP_WORDS,
    check_hit_rate_gate,
    first_english,
    gw_is_usable,
    hit_rate_stats,
)

VOCAB = {"injuring", "harmlessness", "horns", "blade", "sea", "earring",
         "king", "compare", "cow"}


# --- first_english (dictionary-join slots) ---

def test_negated_gloss_rejected_entirely():
    # MW ahiṃsā: harvesting "injuring" would manufacture an antonym anchor
    assert first_english("not injuring anything", VOCAB) is None


def test_caller_falls_through_via_none():
    glosses = ["not injuring anything", "harmlessness"]
    picked = next((w for g in glosses if (w := first_english(g, VOCAB))), None)
    assert picked == "harmlessness"


def test_xref_gloss_rejected():
    assert first_english("see kṛṣṇa", VOCAB) is None
    assert first_english("cf the sea", VOCAB) is None


def test_xref_only_when_first_content_word():
    # "compare" is a genuine first gloss word here, not an xref marker
    assert first_english("compare the sea", VOCAB) == "compare"


def test_scaffold_words_skipped_not_harvested():
    assert first_english("having horns", VOCAB) == "horns"
    assert first_english("relating to the sea", VOCAB) == "sea"


def test_single_letter_skipped():
    assert first_english("c blade", VOCAB) == "blade"


def test_stop_words_still_skipped_and_not_no_are_negators():
    assert first_english("the sea", VOCAB) == "sea"
    assert "not" not in STOP_WORDS and "no" not in STOP_WORDS
    assert "not" in NEGATORS and "no" in NEGATORS


def test_hyphen_join_preserved():
    assert first_english("an ear-ring", VOCAB) == "earring"


def test_empty_and_all_scaffold():
    assert first_english("", VOCAB) is None
    assert first_english("having various", VOCAB) is None


# --- gw_is_usable (value-based slots) ---

def test_gw_plain_word_usable():
    assert gw_is_usable("king")
    assert gw_is_usable("to go")           # stop-word skipped, "go" is content


def test_gw_negation_led_rejected():
    assert not gw_is_usable("not injuring")
    assert not gw_is_usable("without form")


def test_gw_german_negators():
    assert not gw_is_usable("nicht verletzen", negators=DE_NEGATORS)
    assert not gw_is_usable("ohne Form", negators=DE_NEGATORS)
    assert gw_is_usable("König", negators=DE_NEGATORS)


def test_gw_xref_and_junk_rejected():
    assert not gw_is_usable("see previous")
    assert not gw_is_usable("c")
    assert not gw_is_usable("having")
    assert not gw_is_usable("")


# --- stats + gate ---

def test_hit_rate_stats_shape():
    s = hit_rate_stats(hits=3, misses=7, gloss_no_eng=1, anchors=2)
    assert s == {"hits": 3, "misses": 7, "token_hit_rate": 0.3,
                 "gloss_no_eng": 1, "anchors": 2}


def test_gate_fires_below_threshold():
    s = hit_rate_stats(hits=3, misses=7, gloss_no_eng=0, anchors=1)
    assert s["token_hit_rate"] < MIN_HIT_RATE
    with pytest.raises(SystemExit):
        check_hit_rate_gate(s, "MW")


def test_gate_passes_above_threshold():
    s = hit_rate_stats(hits=9, misses=1, gloss_no_eng=0, anchors=5)
    check_hit_rate_gate(s, "MW")  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest shared/tests/test_gloss_filters.py -v`
Expected: FAIL with ImportError (module does not exist).

- [ ] **Step 3: Write the implementation**

`shared/scripts/gloss_filters.py`:

```python
"""
Canonical anchor-English gloss filters — suite v2.

Single source of truth for the noise classes measured in the 2026-07-14
survey and the Sanskrit slot build (journal 2026-07-16): negated glosses
harvested as antonym anchors, cross-reference glosses anchored to "see",
single-letter matches, and scaffold words harvested from gloss prose.

Two entry points, one per slot family:
  - first_english(): dictionary-join slots (greek/LSJ, sanskrit/MW) that scan
    gloss PROSE for the first usable in-vocab content word.
  - gw_is_usable(): value-based slots (sumerian ePSD2, akkadian, hittite,
    egyptian) whose English arrives as a short gw/english VALUE; they keep
    their own junk filters and vocab handling, and add these checks on top.
    Hittite passes DE_NEGATORS (its glosses are German; a negated German
    gloss embeds near its antonym through the translation step).

Also carries the anchor-stats payload and the 40% join-rate gate promoted
from the Sanskrit slot (PGM lesson) — the gate applies to dictionary-join
slots only; value slots have no equivalent join rate.
"""
from __future__ import annotations

import re

NEGATORS = frozenset({"not", "no", "without", "never"})
DE_NEGATORS = frozenset({"nicht", "kein", "keine", "keinen", "ohne", "nie",
                         "niemals"})
# A gloss whose FIRST content word is one of these is a cross-reference,
# not a meaning. ("q.v." never surfaces as a token under _WORD_RE; bare
# "q.v" segments are dropped by the 02 parsers' noise filters.)
XREF_STARTERS = frozenset({"see", "cf", "vid"})
# Gloss prose that is never a meaning. High-frequency verbs that are genuine
# glosses ("go", "act", "make", "kind") are deliberately NOT listed.
SCAFFOLD_WORDS = frozenset({
    "having", "relating", "belonging", "rarely", "who", "whose", "one's",
    "various", "especially", "particularly", "chiefly", "generally",
    "usually", "being",
})
# The Greek 06 set verbatim, minus "not"/"no" (those are NEGATORS: a negator
# must invalidate the gloss, not be skipped over).
STOP_WORDS = frozenset({
    "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with",
    "be", "is", "are", "was", "were", "as", "or", "and",
    "but", "if", "so", "do", "did", "have", "has", "had", "from", "into",
    "out", "up", "down", "over", "under", "between", "during", "before",
    "after", "above", "below", "any", "some", "all", "each", "every",
    "one", "two", "three", "four", "five",
})
MIN_HIT_RATE = 0.40

# Unicode-aware (unlike the v1 per-slot ASCII regex): German glosses carry
# umlauts/ß ("König", "töten") and must tokenize as whole words, not be
# split at the first non-ASCII char (which made every such gw look
# single-letter-first and get wrongly rejected). \w is Unicode in py3;
# digits and underscore excluded.
_WORD_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['\-])*")


def first_english(gloss, eng_vocab_set, negators=NEGATORS):
    """First usable in-vocab content word of `gloss`, or None.

    Rules, in scan order over the gloss's words:
      - negator encountered before a match  -> None (whole gloss rejected)
      - stop/scaffold word                  -> skip, continue
      - first content word is an xref marker-> None (cross-reference gloss)
      - single-letter word                  -> skip, continue
      - word (or hyphen-joined form) in eng_vocab_set -> return it
    Callers fall through to the entry's next gloss on None.
    """
    if not gloss:
        return None
    seen_content = False
    for word in _WORD_RE.findall(gloss.lower()):
        if word in negators:
            return None
        if word in STOP_WORDS or word in SCAFFOLD_WORDS:
            continue
        if not seen_content and word in XREF_STARTERS:
            return None
        seen_content = True
        if len(word) == 1:
            continue
        if word in eng_vocab_set:
            return word
        if "-" in word:
            joined = word.replace("-", "")
            if joined in eng_vocab_set:
                return joined
    return None


def gw_is_usable(value, negators=NEGATORS):
    """Whether a short gw/english value can serve as anchor English.

    Rejects negation-led, xref-led, single-letter-first, and
    stop/scaffold-only values. No embedding-vocab check — that stays at fit
    time, as today. Verdict is based on the first content word.
    """
    if not value:
        return False
    for word in _WORD_RE.findall(value.lower()):
        if word in negators:
            return False
        if word in STOP_WORDS or word in SCAFFOLD_WORDS:
            continue
        if word in XREF_STARTERS:
            return False
        return len(word) > 1
    return False


def hit_rate_stats(hits, misses, gloss_no_eng, anchors):
    """Canonical anchor_stats.json payload for dictionary-join slots."""
    return {
        "hits": hits,
        "misses": misses,
        "token_hit_rate": hits / max(1, hits + misses),
        "gloss_no_eng": gloss_no_eng,
        "anchors": anchors,
    }


def check_hit_rate_gate(stats, source_name):
    """SystemExit if the join hit rate is below MIN_HIT_RATE. Call AFTER
    persisting anchors + stats so the evidence survives the stop."""
    if stats["token_hit_rate"] < MIN_HIT_RATE:
        raise SystemExit(
            f"{source_name} join hit rate {stats['token_hit_rate']:.1%} is "
            f"below the {MIN_HIT_RATE:.0%} gate (PGM lesson). Inspect lemma "
            "normalization / lexicon parse before any FastText compute."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest shared/tests/test_gloss_filters.py -v`
Expected: 16 passed.

- [ ] **Step 5: Full pytest + commit**

```bash
pytest -q
git add shared/scripts/gloss_filters.py shared/tests/test_gloss_filters.py
git commit -m "feat(shared): gloss_filters module — negation/xref/scaffold rules, stats + gate (suite v2)"
```

---

### Task 2: Greek 06 → shared filters + stats/gate

**Files:**
- Modify: `languages/greek/scripts/06_extract_anchors.py`
- Modify: `languages/greek/tests/test_06_anchors.py`

**Interfaces:**
- Consumes: Task 1's `first_english`, `hit_rate_stats`, `check_hit_rate_gate`.
- Produces: Greek `extract_anchors(lemmas, lsj_index, eng_vocab_set, min_occurrences=5) -> tuple[list[dict], dict]` (was single list) and `data/processed/anchor_stats.json` on run.

- [ ] **Step 1: Update the test file (failing first)**

Replace the body of `languages/greek/tests/test_06_anchors.py` with:

```python
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parent.parent / "scripts" / "06_extract_anchors.py"
_spec = spec_from_file_location("grc_anchors_06", str(_path))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_anchors_carry_lemmas():
    lemmas = [{"cf": "θάλασσα", "form": "θαλάσσης"}] * 5
    lsj_index = {"θαλασσα": {"lemma_norm": "θαλασσα", "gloss_first": "the sea",
                             "glosses": ["the sea"]}}
    anchors, stats = _mod.extract_anchors(lemmas, lsj_index, {"sea"},
                                          min_occurrences=5)
    assert anchors
    by_surface = {a["greek"]: a for a in anchors}
    assert by_surface["θαλασσα"]["lemmas"] == ["θαλασσα"]
    assert by_surface["θαλασσησ"]["lemmas"] == ["θαλασσα"]
    assert stats["token_hit_rate"] == 1.0


def test_negated_lsj_gloss_falls_through():
    # LSJ "not to be injured, inviolable" must not anchor to "injured"
    lemmas = [{"cf": "ἄτρωτος", "form": "ἄτρωτος"}] * 5
    lsj_index = {"ατρωτοσ": {"lemma_norm": "ατρωτοσ",
                             "gloss_first": "not to be injured",
                             "glosses": ["not to be injured", "inviolable"]}}
    anchors, _ = _mod.extract_anchors(lemmas, lsj_index,
                                      {"injured", "inviolable"},
                                      min_occurrences=5)
    assert anchors
    assert all(a["english"] == "inviolable" for a in anchors)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest languages/greek/tests/test_06_anchors.py -v`
Expected: FAIL (tuple unpacking / "injured" harvested).

- [ ] **Step 3: Refactor the script**

In `languages/greek/scripts/06_extract_anchors.py`:

(a) Delete the local `STOP_WORDS` constant (lines ~44–51), the local
`_WORD_RE`, and the whole `_load_gloss_first_english` function (~56–75).
Add to the shared-imports block (after the `greek_normalize` import):

```python
from shared.scripts.gloss_filters import (  # noqa: E402
    check_hit_rate_gate,
    first_english,
    hit_rate_stats,
)
```

(b) In `extract_anchors`, replace both `_load_gloss_first_english(eng_vocab_set, ...)`
call sites with `first_english(..., eng_vocab_set=eng_vocab_set)` — exactly:

```python
        english = first_english(lsj.get("gloss_first", ""), eng_vocab_set)
        if not english:
            for g in lsj.get("glosses", [])[1:5]:
                english = first_english(g, eng_vocab_set)
                if english:
                    break
```

(c) Change the tail of `extract_anchors` (after the anchors list is built and
sorted) to return a tuple, mirroring Sanskrit:

```python
    anchors = sorted(anchors, key=lambda a: a["confidence"], reverse=True)
    stats = hit_rate_stats(hits=lsj_hits, misses=lsj_misses,
                           gloss_no_eng=gloss_no_eng, anchors=len(anchors))
    return anchors, stats
```

(d) In `main()`, replace the extraction/save block with:

```python
    anchors, stats = extract_anchors(lemmas, lsj_index, eng_vocab_set)
    output_path = DATA_PROCESSED / "english_anchors.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(anchors, f, ensure_ascii=False, indent=2)
    stats_path = DATA_PROCESSED / "anchor_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTotal anchors: {len(anchors)}")
    print(f"Token-level LSJ join hit rate: {stats['token_hit_rate']:.1%}")
    print(f"Saved to: {output_path} (+ {stats_path.name})")

    check_hit_rate_gate(stats, "LSJ")
```

(e) Update the module docstring's step 4 to mention the v2 filters (negation,
cross-reference, scaffold, single-letter — shared `gloss_filters`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest languages/greek/tests/ -v` then `python -m py_compile languages/greek/scripts/06_extract_anchors.py`
Expected: all pass, clean compile.

- [ ] **Step 5: Full pytest + commit**

```bash
pytest -q
git add languages/greek/scripts/06_extract_anchors.py languages/greek/tests/test_06_anchors.py
git commit -m "feat(greek): 06 on shared gloss_filters — negation/xref/scaffold + stats/gate (suite v2)"
```

---

### Task 3: Sanskrit 06 → shared filters

**Files:**
- Modify: `languages/sanskrit/scripts/06_extract_anchors.py`
- Modify: `languages/sanskrit/tests/test_06_anchors.py`

**Interfaces:**
- Consumes: Task 1 (`first_english`, `hit_rate_stats`, `check_hit_rate_gate`).
- Produces: same `(anchors, stats)` shape as v1 but stats keys become the
  canonical `hits/misses/...` (were `mw_hits/mw_misses/...`).

- [ ] **Step 1: Update tests (failing first)**

In `languages/sanskrit/tests/test_06_anchors.py`:
- DELETE `test_negated_gloss_skipped_not_harvested` and
  `test_all_glosses_negated_drops_anchor` (covered by
  `shared/tests/test_gloss_filters.py` now); keep a single slot-level
  regression, and rename stats keys. Replace the file's test functions with:

```python
def _entry(norm, glosses):
    return {"lemma_norm": norm, "gloss_first": glosses[0], "glosses": glosses}


def test_anchors_carry_lemmas_and_surfaces():
    lemmas = [{"cf": "Deva", "form": "devāḥ"}] * 5
    mw_index = {"deva": _entry("deva", ["heavenly", "divine"])}
    anchors, stats = _mod.extract_anchors(lemmas, mw_index, {"heavenly"},
                                          min_occurrences=5)
    by_surface = {a["sanskrit"]: a for a in anchors}
    assert by_surface["deva"]["lemmas"] == ["deva"]
    assert by_surface["devāḥ"]["lemmas"] == ["deva"]
    assert by_surface["deva"]["english"] == "heavenly"
    assert stats["token_hit_rate"] == 1.0


def test_negated_gloss_regression():
    # end-to-end guard that the slot actually routes through shared filters
    lemmas = [{"cf": "ahiṃsā", "form": "ahiṃsā"}] * 5
    mw_index = {"ahiṃsā": _entry("ahiṃsā",
                                 ["not injuring anything", "harmlessness"])}
    anchors, _ = _mod.extract_anchors(lemmas, mw_index,
                                      {"injuring", "harmlessness"},
                                      min_occurrences=5)
    assert anchors and all(a["english"] == "harmlessness" for a in anchors)


def test_hit_rate_in_stats():
    lemmas = [{"cf": "deva", "form": "deva"}] * 3 + \
             [{"cf": "nope", "form": "nope"}] * 7
    mw_index = {"deva": _entry("deva", ["heavenly"])}
    _, stats = _mod.extract_anchors(lemmas, mw_index, {"heavenly"},
                                    min_occurrences=1)
    assert stats["hits"] == 3 and stats["misses"] == 7
    assert abs(stats["token_hit_rate"] - 0.30) < 1e-9
```

(keep the file's existing import/loader header unchanged)

- [ ] **Step 2: Run to verify failure**

Run: `pytest languages/sanskrit/tests/test_06_anchors.py -v`
Expected: FAIL on stats key names (`hits` vs `mw_hits`).

- [ ] **Step 3: Refactor the script**

In `languages/sanskrit/scripts/06_extract_anchors.py`:
- Delete local `MIN_HIT_RATE`, `NEGATORS`, `STOP_WORDS`, `_WORD_RE`, and
  `_load_gloss_first_english`; import from shared instead:

```python
from shared.scripts.gloss_filters import (  # noqa: E402
    check_hit_rate_gate,
    first_english,
    hit_rate_stats,
)
```

- Replace both gloss-selection call sites with `first_english(g, eng_vocab_set)`
  (same shape as Task 2 step 3b).
- Replace the hand-built stats dict with
  `stats = hit_rate_stats(hits=mw_hits, misses=mw_misses, gloss_no_eng=gloss_no_eng, anchors=len(anchors))`
  (local counter names stay `mw_hits`/`mw_misses`).
- Replace the inline gate block in `main()` with `check_hit_rate_gate(stats, "MW")`
  (keep it after both json.dump calls).
- Update the docstring's deviation paragraph: the negation rule is now
  canonical (shared `gloss_filters`, suite v2) rather than a Sanskrit-only
  deviation; xref/scaffold/single-letter filters added.

- [ ] **Step 4: Run tests, compile check**

Run: `pytest languages/sanskrit/tests/ -v && python -m py_compile languages/sanskrit/scripts/06_extract_anchors.py`
Expected: all pass.

- [ ] **Step 5: Full pytest + commit**

```bash
pytest -q
git add languages/sanskrit/scripts/06_extract_anchors.py languages/sanskrit/tests/test_06_anchors.py
git commit -m "refactor(sanskrit): 06 on shared gloss_filters (negation rule promoted to canonical)"
```

---

### Task 4: Sumerian + Akkadian 06 → `gw_is_usable` + stats file

**Files:**
- Modify: `languages/sumerian/scripts/06_extract_anchors.py`
- Modify: `languages/akkadian/scripts/06_extract_anchors.py`

**Interfaces:**
- Consumes: Task 1 (`gw_is_usable`).
- Produces: both scripts write `data/processed/anchor_stats.json` =
  `{"anchors": <int>, "gw_rejected_v2": <int>, "source_counts": {<source>: <int>}}`.

- [ ] **Step 1: Sumerian — add the filter to the ePSD2 source only**

In `languages/sumerian/scripts/06_extract_anchors.py`:
- Add import: `from shared.scripts.gloss_filters import gw_is_usable` (into the
  existing sys.path-shimmed import block).
- In `extract_epsd2_anchors()`, immediately after the existing junk checks on
  `gw` (the `junk_english`/length/digit/`~` block), add:

```python
            if not gw_is_usable(gw):
                gw_rejected += 1
                continue
```

  with `gw_rejected = 0` initialized at the top of the function and returned
  alongside (change the function to return `(anchors, gw_rejected)`; update
  its caller accordingly).
- The co-occurrence source (`extract_cooccurrence_anchors`) is NOT touched
  (statistical source, own stop-word set — spec §2).
- In `main()`, after the merged anchors are written, also write:

```python
    stats = {
        "anchors": len(merged),
        "gw_rejected_v2": gw_rejected,
        "source_counts": source_counts,
    }
    with open(DATA_PROCESSED / "anchor_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"anchor_stats.json written: {stats}")
```

  where `source_counts` is the per-source dict the script already prints
  (build it from the existing per-source split variables).

- [ ] **Step 2: Akkadian — extend `_filter_gloss`**

In `languages/akkadian/scripts/06_extract_anchors.py`:
- Add the same import.
- In `_filter_gloss(gw)` add, after the existing junk checks and before the
  final return of the cleaned gloss:

```python
    if not gw_is_usable(gw):
        return None
```

- In `main()`, write `anchor_stats.json` with the same shape as Sumerian's
  (`anchors` = final count; `gw_rejected_v2`: add a module-level counter
  incremented inside `_filter_gloss` when `gw_is_usable` rejects;
  `source_counts` = `{"oracc": <count>}`).

- [ ] **Step 3: Compile + smoke the filters (no full runs)**

```bash
python -m py_compile languages/sumerian/scripts/06_extract_anchors.py \
                     languages/akkadian/scripts/06_extract_anchors.py
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from shared.scripts.gloss_filters import gw_is_usable
assert gw_is_usable("king") and not gw_is_usable("see previous")
print("smoke OK")
EOF
```

- [ ] **Step 4: Full pytest + commit**

```bash
pytest -q
git add languages/sumerian/scripts/06_extract_anchors.py languages/akkadian/scripts/06_extract_anchors.py
git commit -m "feat(sumerian,akkadian): gw_is_usable v2 filter + anchor_stats.json"
```

---

### Task 5: Hittite 06 → German negation rejection + stats file

**Files:**
- Modify: `languages/hittite/scripts/06_extract_anchors.py`

**Interfaces:**
- Consumes: Task 1 (`gw_is_usable`, `DE_NEGATORS`).
- Produces: `data/processed/anchor_stats.json` (same shape as Task 4).

- [ ] **Step 1: Add the filter BEFORE the translation step**

In `languages/hittite/scripts/06_extract_anchors.py`:
- Import: `from shared.scripts.gloss_filters import DE_NEGATORS, gw_is_usable`.
- In `_filter_gloss(gw)` (the German-side filter), add after the existing
  `JUNK_GLOSSES`/length/digit checks:

```python
    if not gw_is_usable(gw, negators=DE_NEGATORS):
        return None
```

  Rationale (add as a one-line comment): a negated German gloss ("nicht
  verletzen") embeds near its antonym through the Gemma translation step —
  the same A1 antonym-anchor bug, one language removed.
- Count rejections (module-level `gw_rejected_v2` counter as in Task 4) and
  write `anchor_stats.json` in `main()` with
  `source_counts = {"german_gloss": <primary count>, "heterogram": <bridge count>}`
  (both counts are already printed by the script — reuse those variables).
- The heterogram-bridge source is NOT filtered (it has no gloss text).

- [ ] **Step 2: Compile + focused check**

```bash
python -m py_compile languages/hittite/scripts/06_extract_anchors.py
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from shared.scripts.gloss_filters import gw_is_usable, DE_NEGATORS
assert not gw_is_usable("nicht verletzen", negators=DE_NEGATORS)
assert gw_is_usable("König", negators=DE_NEGATORS)
print("smoke OK")
EOF
```

- [ ] **Step 3: Full pytest + commit**

```bash
pytest -q
git add languages/hittite/scripts/06_extract_anchors.py
git commit -m "feat(hittite): reject negated German glosses pre-translation (A1 via embedding) + anchor_stats"
```

---

### Task 6: Egyptian 06 → `gw_is_usable` on the english field + stats file

**Files:**
- Modify: `languages/egyptian/scripts/06_extract_anchors.py`

**Interfaces:**
- Consumes: Task 1 (`gw_is_usable`).
- Produces: `data/processed/anchor_stats.json` (Task 4 shape,
  `source_counts = {"tla_ramses": <count>}`).

- [ ] **Step 1: Add the filter in `normalize_anchors`**

- Import `gw_is_usable` (same import block pattern).
- In `normalize_anchors()`, after the existing `_JUNK_ENGLISH`/length/digit
  checks on the `english` value, add:

```python
        if not gw_is_usable(english):
            gw_rejected += 1
            continue
```

  (counter initialized/returned as in Task 4; NOTE: egyptian's existing
  `len<=1` check stays — it is subsumed but harmless.)
- Write `anchor_stats.json` in `main()`.
- Do NOT touch `09_align_and_evaluate.py`'s `filter_stopword_glosses`
  (documented variant delta, German extras — spec §2).

- [ ] **Step 2: Compile check**

```bash
python -m py_compile languages/egyptian/scripts/06_extract_anchors.py
```

- [ ] **Step 3: Full pytest + commit**

```bash
pytest -q
git add languages/egyptian/scripts/06_extract_anchors.py
git commit -m "feat(egyptian): gw_is_usable v2 filter on english field + anchor_stats"
```

---

### Task 7: Alpha-v2 — `val_topk_csls` + canonical `select_alpha` + clone refresh

**Files:**
- Modify: `shared/scripts/eval_suite.py`
- Modify: `languages/akkadian/scripts/09_align_and_evaluate.py` (canonical)
- Regenerate: `languages/{hittite,greek,sanskrit}/scripts/{09_align_and_evaluate,09b_align_gemma}.py`
- Modify in place: `languages/sumerian/scripts/09_align_and_evaluate.py`, `languages/egyptian/scripts/09_align_and_evaluate.py` (variants — same hunks)
- Test: `shared/tests/test_eval_suite.py` (append one test)

**Interfaces:**
- Consumes: existing `score_regime(Q, golds, cand_vectors, cand_vocab, query_pool, ks)` in eval_suite.
- Produces: `val_topk_csls(Y_pred_val, val_golds, cand_vectors, cand_vocab) -> tuple[float, float]` (top1_exact, top5_exact, percentages); `select_alpha` returning lowest-alpha-on-plateau winner; results JSONs carry `"alpha_selection": "val_top5_csls_v2"` and sweep records with both `val_top1_csls_exact` and `val_top5_csls_exact`.

- [ ] **Step 1: Add the scorer + failing test**

Append to `shared/tests/test_eval_suite.py` (match the file's existing fixture
style for building small candidate matrices — read the file first and reuse
its helpers):

```python
def test_val_topk_returns_top1_and_top5():
    # identical setup to the existing val_top1_csls test in this file;
    # top1 must equal the existing scorer, top5 >= top1.
    ...  # reuse the file's existing small-matrix fixture verbatim
```

Concretely: copy the arrange block of the existing `val_top1_csls` test in
that file, call `val_topk_csls(...)`, and assert
`topk[0] == val_top1_csls(<same args>)` and `topk[1] >= topk[0]`.

Then implement in `shared/scripts/eval_suite.py`, directly below
`val_top1_csls`:

```python
def val_topk_csls(Y_pred_val, val_golds, cand_vectors, cand_vocab):
    """Val top-1 and top-5 CSLS exact (%, restricted candidates). The alpha
    selector (suite v2) selects on top-5 — ~5x the signal of top-1 on weak
    slots — and records both."""
    r = score_regime(Y_pred_val, val_golds, cand_vectors, cand_vocab,
                     query_pool=Y_pred_val, ks=(1, 5))
    return r["top1"]["exact"], r["top5"]["exact"]
```

Run: `pytest shared/tests/test_eval_suite.py -v` — new test passes, all old pass.

- [ ] **Step 2: Rewrite canonical `select_alpha`**

In `languages/akkadian/scripts/09_align_and_evaluate.py`, add `val_topk_csls`
to the existing `shared.scripts.eval_suite` import list, and replace the
whole `select_alpha` function with:

```python
def select_alpha(
    X_train, Y_train, X_val, val_english, eng_vocab_list, eng_vectors,
    alphas, predict_transform=None,
):
    """Pick the Ridge alpha by val top-5 CSLS; plateau ties -> lowest alpha.

    Suite v2 rule: scores within one anchor's worth (100/n_val percentage
    points) of the max form a plateau, and the LOWEST alpha on it wins —
    less regularization preserves the dictionary stratum. Fixes the v1
    failure mode where a flat-noise sweep picked alpha=1e4 by a one-anchor
    margin (journal 2026-07-09, Akkadian-Gemma).

    predict_transform: optional callable applied to raw predictions before
    evaluation (Egyptian's PCA path lifts 256d back to 768d with it).
    Returns (best_alpha, sweep_records).
    """
    sweep = []
    for alpha in tqdm(alphas, desc="alpha sweep", file=sys.stderr,
                      disable=not sys.stderr.isatty()):
        model = train_ridge(X_train, Y_train, alpha=alpha)
        Y_pred = model.predict(X_val)
        if predict_transform is not None:
            Y_pred = predict_transform(Y_pred)
        top1, top5 = val_topk_csls(
            Y_pred, val_english, eng_vectors[:CAND_SIZE], eng_vocab_list[:CAND_SIZE]
        )
        sweep.append({"alpha": alpha, "val_top1_csls_exact": top1,
                      "val_top5_csls_exact": top5})
        print(f"  alpha={alpha:<10g} val top5 (CSLS/50k)={top5:.2f}%  top1={top1:.2f}%")
    best_top5 = max(r["val_top5_csls_exact"] for r in sweep)
    plateau_eps = 100.0 / max(1, len(val_english))  # one anchor, in pp
    best_alpha = min(r["alpha"] for r in sweep
                     if r["val_top5_csls_exact"] >= best_top5 - plateau_eps)
    return best_alpha, sweep
```

Also add `"alpha_selection": "val_top5_csls_v2",` to the results `config`
dict, on the line directly after its `"alpha":` entry.

- [ ] **Step 3: Sed-refresh the three full clones + diff gates**

```bash
for lang in hittite greek sanskrit; do
  Lang=$(python3 -c "print('$lang'.capitalize())")
  for f in 09_align_and_evaluate.py 09b_align_gemma.py; do
    sed -e "s/akkadian/$lang/g" -e "s/Akkadian/$Lang/g" \
      languages/akkadian/scripts/$f > languages/$lang/scripts/$f
    diff <(sed -e "s/akkadian/$lang/g" -e "s/Akkadian/$Lang/g" \
      languages/akkadian/scripts/$f) languages/$lang/scripts/$f
  done
done
```

Expected: all diffs empty. CAUTION — this refresh overwrites the sanskrit
09/09b that were greek-derived (`sumerian` legacy strings became `sanskrit`);
after the refresh run:

```bash
grep -n 'fasttext' languages/sanskrit/scripts/09_align_and_evaluate.py languages/sanskrit/scripts/09b_align_gemma.py
grep -rn 'sumerian\|greek' languages/{hittite,greek,sanskrit}/scripts/09*.py | grep -v '^\S*greek/scripts/09[^:]*:.*greek'
```

The akkadian canonical loads `fasttext_akkadian.model`→ sed → per-slot names;
verify sanskrit's shows `fasttext_sanskrit.model` and hittite/greek show
their v1-era filenames (`fasttext_hittite` / `fasttext_sumerian` — greek's
legacy artifact name). If greek's regenerated 09 says `fasttext_greek` but
its trained artifact on disk is `fasttext_sumerian.model`, STOP and surface:
the executor must symlink `fasttext_greek.model -> fasttext_sumerian.model`
(plus `.vec` and the two `.npy` sidecars… symlinks named
`fasttext_greek.model.wv.vectors_ngrams.npy` etc.) OR restore the filename
line — ask the controller; do not silently rename 654MB artifacts.

- [ ] **Step 4: Apply the same hunks to the two variants**

In `languages/sumerian/scripts/09_align_and_evaluate.py` and
`languages/egyptian/scripts/09_align_and_evaluate.py`: replace their
`select_alpha` bodies with the Step-2 function verbatim (egyptian keeps its
`predict_transform` usage — the signature is unchanged) and add the
`"alpha_selection"` config key. Verify the function bodies match canonical:

```bash
python - <<'EOF'
import ast, sys
def fn_src(path, name):
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.dump(node)
    sys.exit(f"{name} not in {path}")
canon = fn_src("languages/akkadian/scripts/09_align_and_evaluate.py", "select_alpha")
for slot in ("sumerian", "egyptian"):
    other = fn_src(f"languages/{slot}/scripts/09_align_and_evaluate.py", "select_alpha")
    assert other == canon, f"{slot} select_alpha diverges from canonical"
print("variant select_alpha bodies match canonical")
EOF
```

- [ ] **Step 5: Compile all, full pytest, commit**

```bash
python -m py_compile languages/*/scripts/09*.py
pytest -q
git add shared/scripts/eval_suite.py shared/tests/test_eval_suite.py languages/*/scripts/09_align_and_evaluate.py languages/*/scripts/09b_align_gemma.py
git commit -m "feat(shared,slots): alpha-v2 selection — val top-5 CSLS, low-alpha plateau ties"
```

---

### Task 8: HF `suite-v1` tag + Sanskrit re-run (bellwether)

**Files:** none committed (runs + verification; logs under `languages/sanskrit/logs/`)

- [ ] **Step 1: Pin v1 on the HF mirror**

```bash
python -c "from huggingface_hub import create_tag; create_tag('ebrinz/hyper-glyphy-artifacts', tag='suite-v1', repo_type='dataset'); print('tagged')"
```

Precondition: the initial mirror upload must have completed (`ALL DONE
FAILED=0` in its log). If it has not, STOP and surface.

- [ ] **Step 2: v2 anchors + guardrail**

```bash
python languages/sanskrit/scripts/06_extract_anchors.py
python - <<'EOF'
import json
n = len(json.load(open("languages/sanskrit/data/processed/english_anchors.json")))
assert n >= 81536, f"guardrail: sanskrit v2 anchors {n} < 85% of v1 (95,924)"
print(f"sanskrit v2 anchors: {n} (v1: 95,924) — guardrail OK")
print(json.load(open("languages/sanskrit/data/processed/anchor_stats.json")))
EOF
```

Record the count and the delta vs v1. Expected: a few % drop ("see"/single-
letter/scaffold anchors removed).

- [ ] **Step 3: Alignments + export + procrustes (detached)**

```bash
mkdir -p languages/sanskrit/logs
nohup bash -c '
python languages/sanskrit/scripts/09_align_and_evaluate.py &&
python languages/sanskrit/scripts/09b_align_gemma.py --mode whitened &&
python languages/sanskrit/scripts/10_export_production.py &&
python shared/scripts/procrustes_align.py --slot sanskrit
' > languages/sanskrit/logs/v2_rerun.log 2>&1 &
```

Poll the log until all four stages print. Then record from the results JSONs:
selected alphas (+ full sweeps), the whole metric_suite per target,
`gold_oov_candidates`, and the new procrustes val cosine. Compare line-by-line
against v1 (v1 numbers: GloVe α=1.0 dict 33.51 / interp 2.55 / zs 0.23 /
combined 2.22; Gemma α=1000 dict 44.40 / interp 4.35 / zs 0.74 / combined
3.83; procrustes 0.1145). Sanity expectations: dictionary stratum flat-to-up;
no stratum collapses by more than a few points; alpha picks on healthy
interior maxima. Anything outside that: STOP and surface before launching the
other five slots.

- [ ] **Step 4: Mirror the refreshed slot**

```bash
hf upload ebrinz/hyper-glyphy-artifacts languages/sanskrit/models languages/sanskrit/models --repo-type dataset --include "fasttext_*" "fused_*" "ridge_*" "*procrustes*" --commit-message "suite-v2: sanskrit models"
hf upload ebrinz/hyper-glyphy-artifacts languages/sanskrit/data/processed languages/sanskrit/data/processed --repo-type dataset --exclude "glove*" --commit-message "suite-v2: sanskrit processed"
hf upload ebrinz/hyper-glyphy-artifacts languages/sanskrit/results languages/sanskrit/results --repo-type dataset --commit-message "suite-v2: sanskrit results"
hf upload ebrinz/hyper-glyphy-artifacts languages/sanskrit/final_output languages/sanskrit/final_output --repo-type dataset --commit-message "suite-v2: sanskrit exports"
```

(No FastText retrain happened — models upload is a no-op via dedup; included
for uniformity. NOTE: 10_export overwrites final_output and 06 overwrites
processed, so these dirs DID change.)

- [ ] **Step 5: Ledger note (no git commit — results are gitignored)**

Record all measured numbers in the SDD progress ledger for Task 12's docs.

---

### Task 9: Akkadian re-run (alpha acid test)

**Files:** none committed (runs; logs under `languages/akkadian/logs/`)

- [ ] **Step 1: v2 anchors + guardrail**

```bash
python languages/akkadian/scripts/06_extract_anchors.py
python - <<'EOF'
import json
n = len(json.load(open("languages/akkadian/data/processed/english_anchors.json")))
assert n >= 20753, f"guardrail: akkadian v2 anchors {n} < 85% of v1 (24,415)"
print(f"akkadian v2 anchors: {n} (v1: 24,415) — guardrail OK")
EOF
```

- [ ] **Step 2: Alignments + export + procrustes? — NO procrustes (akkadian is not in SLOTS)**

```bash
mkdir -p languages/akkadian/logs
nohup bash -c '
python languages/akkadian/scripts/09_align_and_evaluate.py &&
python languages/akkadian/scripts/09b_align_gemma.py --mode whitened &&
python languages/akkadian/scripts/10_export_production.py
' > languages/akkadian/logs/v2_rerun.log 2>&1 &
```

- [ ] **Step 3: The acid test**

From `alignment_results_gemma_whitened.json`: the v1 pathology was alpha=1e4
(flat-noise pick) crushing the dictionary stratum to 19.9% (GloVe pick got
48.5%). v2 success criterion: the Gemma alpha lands at or below the GloVe-
side alpha region AND the Gemma dictionary stratum recovers to within a few
points of the GloVe-side dictionary number. Record the full sweep table in
the ledger. If the sweep is still flat noise and the plateau rule picks the
lowest alpha, that IS the designed behavior — record it as such.

- [ ] **Step 4: Mirror the refreshed slot (same four uploads as Task 8 Step 4, akkadian paths)**

---

### Task 10: Remaining four re-runs (sumerian → hittite → greek → egyptian)

**Files:** none committed (runs; logs per slot)

- [ ] **Step 1: Per slot, sequentially — anchors + guardrail first**

Guardrail floors: sumerian ≥ 11,135 (v1 13,100); hittite ≥ 9,988 (v1 11,750);
greek ≥ 90,321 (v1 106,260); egyptian ≥ 3,529 (v1 4,152). Same one-liner
pattern as Task 8 Step 2 with each slot's path and floor.

- [ ] **Step 2: Alignments + exports (detached, one slot at a time)**

Per slot the stage list is:
- sumerian: `09` → `09b` (check `languages/sumerian/README.md` for its 09b
  invocation flags — run exactly what the README documents) → `10` →
  `procrustes_align.py --slot sumerian`
- hittite: `09` → `09b --mode whitened` → `10` → `procrustes --slot hittite`
- greek: `09` → `09b --mode whitened` → `10` → `procrustes --slot greek`
- egyptian: `09` → `09b` (per its README flags) → `10` (NO procrustes)

Hittite note: its 06 depends on `german_to_english.json` translation cache
and Gemma model load — run 06 on a machine session where
`sentence-transformers` can load; if the heterogram bridge inputs
(`/tmp/sumerian_vocab_for_bridge.json`) are missing, follow the error message
(the script names its regeneration step) — do not skip the bridge source.

Greek FastText filename: whatever Task 7 Step 3 resolved (symlink or restored
line) must be in place before greek 09 runs — verify
`python -c "from gensim.models import FastText; ..."` is NOT needed; just
check the file the script opens exists:
`ls -la $(grep -o 'fasttext_[a-z]*\.model' languages/greek/scripts/09_align_and_evaluate.py | head -1 | sed 's|^|languages/greek/models/|')`

- [ ] **Step 3: Record per-slot suites + procrustes cosines in the ledger; mirror each slot to HF (Task 8 Step 4 pattern)**

- [ ] **Step 4: Cross-slot verification before docs**

All six slots re-run, leak checks 0.00% expected (spot-verify sanskrit +
akkadian with the controller's group_split + edit-distance≤1 script from the
sanskrit build), every results JSON carries `"alpha_selection":
"val_top5_csls_v2"`, procrustes results exist for sumerian/hittite/greek/
sanskrit.

---

### Task 11: A5 hubness diagnostic (journal-only)

**Files:**
- Create (scratch, NOT committed): analysis script run inline
- Modify: `docs/EXPERIMENT_JOURNAL.md` (one note, folded into Task 12's entry is fine — produce the numbers here)

- [ ] **Step 1: Run the diagnostic over the existing Gate-2 artifacts**

Inputs: the doc-eval parallels artifacts used by
`shared/results/doc_eval_parallels.json` (Ridge plane; the procrustes variant
JSON as cross-check). Reconstruct the document centroid matrix exactly the
way `shared/scripts/doc_eval.py` does (call its loader functions from a
script — do not reimplement): the Greek candidate pool (~820 docs) plus the
three Hittite query docs, in the aligned space.

Compute and record:
1. Centroid L2-norm distribution of the Greek pool (mean, std, percentiles)
   and where Theogony + the two Typhon docs sit in it.
2. Hubness: for each Greek doc, its mean cosine to all Hittite-side queries
   and to the pool itself (`mean_cos_pool`); rank Theogony/Typhon docs on
   `mean_cos_pool` (anti-hub = bottom of that ranking).
3. The key question: do the true-parallel targets sit in the anti-hub tail
   (bottom decile of `mean_cos_pool`)? If yes, Gate 2's below-chance ranks
   are (at least partly) a target-side anti-hub artifact — the queries rank
   EVERY doc above an anti-hub target. If no, the below-chance pattern
   remains unexplained; say so.

- [ ] **Step 2: Write the conclusion paragraph**

Three-to-five sentences with the measured numbers, stated either way
(artifact confirmed / not confirmed / partial). Hand the paragraph + the
repro commands to Task 12. No benchmark change, no committed script.

---

### Task 12: Docs — v2 table, A3 disclosure, A6 note, journal, ship

**Files:**
- Modify: `README.md` (suite table v2 + archived v1 + A3 caption)
- Modify: `docs/EXPERIMENT_JOURNAL.md` (suite-v2 entry)
- Modify: `shared/scripts/doc_eval.py` (docstring note only — A6)
- Modify: `languages/*/README.md` where they quote suite numbers (greek, sanskrit at minimum — grep for stale v1 cells)

- [ ] **Step 1: A6 docstring note**

Append to `shared/scripts/doc_eval.py`'s module docstring:

```
Note (2026-07, A6): document tokenization here is raw line.split() + the
slot normalizer, while FastText corpora pass through each slot's
05_clean_and_tokenize. A known, accepted inconsistency of this parked
doc-level plane — do not "fix" it without re-running Gates 1/2.
```

- [ ] **Step 2: README v2 table**

- Rename the current suite table's heading to include `(suite v1, archived
  2026-07-16 — pre-gloss-filter anchors, val-top-1 alpha)` and move it BELOW
  a new v2 table (same columns PLUS one `gold OOV` column populated from each
  results JSON's `test_combined.gold_oov_candidates` over `n`).
- Caption sentence under the v2 table (A3, verbatim): "Accuracies are
  conditioned on gold glosses present in the 50k-candidate English
  vocabulary; the gold-OOV column counts test items excluded by that
  restriction."
- Fill every v2 cell from the Task 8–10 ledger records. No cell may carry a
  v1 number.

- [ ] **Step 3: Journal entry (dated, one entry for the whole pass)**

In order: (1) recipe deltas — gloss_filters module (constants verbatim),
two slot families, alpha-v2 rule; (2) per-slot v1→v2 anchor counts and the
guardrail outcomes; (3) per-slot suite tables both targets with alphas and
sweeps summarized — call out the Akkadian-Gemma acid-test outcome
explicitly; (4) new procrustes val cosines for sumerian/hittite/greek/
sanskrit with the fixed wording: "the v1 retire verdict was pre-registered
on the v1 recipe and stands; v2 values are reported for the record" — and,
if any value exceeds 0.12, the flag sentence per spec §5; (5) the A5
paragraph from Task 11; (6) the A6 note's existence.

- [ ] **Step 4: Full pytest + commit + ship checklist**

```bash
pytest -q
git add README.md docs/EXPERIMENT_JOURNAL.md shared/scripts/doc_eval.py languages/*/README.md languages/*/final_output/metadata.json
git commit -m "docs: suite v2 — tables, journal, A3 disclosure, A6 note"
```

(metadata.json files: 10_export refreshed them during Tasks 8–10; they are
git-tracked and must ship with the docs that quote them.)

Then: final whole-branch review → merge/push per
superpowers:finishing-a-development-branch; HF mirror already updated
incrementally; memory update (suite v2 shipped, v1 tag, per-slot deltas).

---

## Self-Review Record

- **Spec coverage:** module + two families (T1–T6, amendment honored), alpha-v2 (T7), suite-v1 HF tag + ordered re-runs + guardrails + incremental mirror (T8–T10), verdict-preserving reporting + A3 + v1 archive (T12), A5 (T11), A6 (T12 Step 1). Egyptian 09 filter untouched (spec §2); sumerian co-occurrence untouched (spec §2); MIN_OCCURRENCES unchanged.
- **Known risks made explicit:** greek FastText legacy filename after the akkadian-canonical sed-refresh (T7 Step 3 STOP-and-surface); hittite 06's translation-cache/bridge dependencies (T10); mirror precondition on the initial upload (T8 Step 1).
- **Placeholder scan:** T7 Step 1's test says "reuse the file's existing fixture verbatim" — deliberate: the fixture exists in the repo and copying it here risks drift; the step names exactly which test to copy from and what to assert. No other TBDs.
- **Type consistency:** `first_english(gloss, eng_vocab_set, negators=NEGATORS)` and `(anchors, stats)` tuple used consistently across T1/T2/T3; `gw_is_usable(value, negators=...)` across T4/T5/T6; `val_topk_csls` returns `(top1, top5)` consumed positionally in T7; stats keys `hits/misses/token_hit_rate/gloss_no_eng/anchors` consistent T1/T2/T3, value-slot stats shape consistent T4/T5/T6.
