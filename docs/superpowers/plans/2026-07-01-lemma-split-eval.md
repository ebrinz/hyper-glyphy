# Lemma-Group Split + Validation Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate surface-variant train/test leakage and test-set alpha tuning from all five language alignment pipelines, then re-run the four shipped languages to produce honest metrics.

**Architecture:** A new shared `group_split()` (union-find over lemma/surface/gloss nodes) replaces `train_test_split` everywhere, producing a 64/16/20 train/val/test partition where no lemma, surface, or (fallback) gloss spans partitions. Alpha selection moves *inline* into every 09/09b script: sweep on validation, retrain train+val at the winner, report on the untouched test set. The standalone `ridge_alpha_sweep.py` scripts (which tuned on test) are deleted.

**Tech Stack:** Python 3.11+, numpy, scikit-learn Ridge, gensim FastText (existing), pytest.

**Spec:** `docs/superpowers/specs/2026-07-01-lemma-split-eval-design.md`

**⚠ One structural deviation from the spec (same methodology, different file layout):** the spec kept `ridge_alpha_sweep.py` as the alpha selector with 09/09b consuming its choice. This plan instead builds validation-based alpha selection directly into every 09/09b run (`select_alpha()` helper) and deletes `ridge_alpha_sweep.py`. Rationale: Egyptian already sweeps inline; a separate sweep file requires manually transcribing alpha into 09b (the exact drift that produced the Sumerian alpha-recording bug); and keeping a test-set-tuning script invites misuse. The methodology is exactly the spec's: val selects, train+val retrains, test reports. Doc references to `ridge_alpha_sweep.py` (e.g. Hittite README) are handled by the separate docs-refresh task.

## Global Constraints

- Split: seed **42**, `val_size=0.16`, `test_size=0.20` of anchor mass (targets, not exact).
- Alpha grid everywhere: `[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]`.
- Anchor extraction reruns must leave the (surface, english, confidence, frequency, source) content **byte-equivalent**; only the new `lemmas` field may be added. Verify with the diff snippet in each 06 task.
- OOV (subword-inferred) anchors: train-partition only. Implemented by calling `build_training_data` **without** `fasttext_model` for val/test.
- Existing helper functions (`build_training_data`, `train_ridge`, `evaluate_alignment`) keep their signatures — existing tests and `align_09*.py` shims depend on them.
- Legacy config key names in results JSONs (e.g. Akkadian's `"sumerian_vocab"`) are kept as-is; `10_export_production.py` scripts are not modified.
- All commands run from the repo root `/Users/crashy/Development/hyper-glyphy`.
- Runtime note: each 09/09b run now evaluates 10 alphas on val (~1–2 min per alpha for cdist against the 400k vocab). Expect ~15–25 min per script run. `evaluate_alignment` allocates a ~(n_items × 400k) float64 matrix (~5–8 GB at 2k items) — same as today's shipped runs.

---

### Task 1: Shared group-split module

**Files:**
- Create: `shared/scripts/anchor_split.py`
- Test: `shared/tests/test_anchor_split.py`

**Interfaces:**
- Produces: `group_split(anchors: list[dict], surface_key: str, val_size=0.16, test_size=0.20, seed=42) -> tuple[list[dict], list[dict], list[dict]]` (train, val, test), and module constants `VAL_SIZE = 0.16`, `TEST_SIZE = 0.20`, `SEED = 42`. Every later task imports these.

- [ ] **Step 1: Write the failing tests**

Create `shared/tests/test_anchor_split.py`:

```python
from shared.scripts.anchor_split import group_split


def _partition_sets(train, val, test, field):
    def collect(part):
        out = set()
        for a in part:
            v = a.get(field)
            if isinstance(v, list):
                out.update(v)
            elif v is not None:
                out.add(v)
        return out
    return collect(train), collect(val), collect(test)


def _make_anchors(n_groups=300, seed_sizes=(1, 1, 1, 2, 2, 3, 5, 8)):
    """Synthetic anchors: n_groups lemmas, varying surface counts per lemma."""
    anchors = []
    for g in range(n_groups):
        size = seed_sizes[g % len(seed_sizes)]
        for s in range(size):
            anchors.append({
                "akkadian": f"lemma{g}_surf{s}",
                "english": f"gloss{g % 50}",
                "lemmas": [f"lemma{g}"],
            })
    return anchors


def test_no_lemma_spans_partitions():
    anchors = _make_anchors()
    train, val, test = group_split(anchors, surface_key="akkadian")
    tr, va, te = _partition_sets(train, val, test, "lemmas")
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_no_surface_spans_partitions():
    # Same surface registered under two different lemmas must not split.
    anchors = _make_anchors()
    anchors.append({"akkadian": "lemma0_surf0", "english": "other",
                    "lemmas": ["lemmaX"]})
    train, val, test = group_split(anchors, surface_key="akkadian")
    tr, va, te = _partition_sets(train, val, test, "akkadian")
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_shared_surface_merges_lemma_groups():
    anchors = [
        {"akkadian": "aaa", "english": "one", "lemmas": ["L1"]},
        {"akkadian": "aaa", "english": "two", "lemmas": ["L2"]},
        {"akkadian": "bbb", "english": "three", "lemmas": ["L2"]},
    ]
    train, val, test = group_split(anchors, surface_key="akkadian")
    parts = [p for p in (train, val, test) if p]
    assert len(parts) == 1 and len(parts[0]) == 3


def test_gloss_fallback_no_gloss_spans_partitions():
    # Anchors without a `lemmas` field group by their English gloss.
    anchors = [{"egyptian_raw": f"w{i}", "english": f"g{i % 40}"}
               for i in range(400)]
    train, val, test = group_split(anchors, surface_key="egyptian_raw")
    tr, va, te = _partition_sets(train, val, test, "english")
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_deterministic():
    anchors = _make_anchors()
    a = group_split(anchors, surface_key="akkadian")
    b = group_split(anchors, surface_key="akkadian")
    assert a == b


def test_partition_proportions():
    anchors = _make_anchors(n_groups=500)
    train, val, test = group_split(anchors, surface_key="akkadian")
    total = len(anchors)
    assert len(train) + len(val) + len(test) == total
    assert abs(len(test) / total - 0.20) < 0.05
    assert abs(len(val) / total - 0.16) < 0.05


def test_empty_input():
    assert group_split([], surface_key="akkadian") == ([], [], [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest shared/tests/test_anchor_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shared.scripts.anchor_split'`

- [ ] **Step 3: Write the implementation**

Create `shared/scripts/anchor_split.py`:

```python
"""
Group-aware train/val/test split for anchor lists.

Prevents surface-variant leakage: all anchors sharing a lemma or a source
surface land in the same partition, so (šarrum, "king") can never train
while (šarru, "king") is tested. Anchors with no `lemmas` field group by
their English gloss instead (Egyptian, whose extraction lives outside this
repo; guarantees no gold label spans the split there).

Assignment is largest-group-first to the partition with the greatest
remaining deficit, so oversized groups (e.g. a high-frequency gloss group)
land in train rather than blowing up the test fraction.

See: docs/superpowers/specs/2026-07-01-lemma-split-eval-design.md
"""
import random

VAL_SIZE = 0.16
TEST_SIZE = 0.20
SEED = 42


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(anchors, surface_key):
    """Return one group id per anchor via union-find.

    Node keys per anchor: ("surface", <source surface>) always, plus
    ("lemma", l) for each entry in anchor["lemmas"] when present, else
    ("gloss", anchor["english"]).
    """
    uf = _UnionFind()
    anchor_nodes = []
    for a in anchors:
        surface_node = ("surface", a[surface_key])
        lemmas = a.get("lemmas")
        if lemmas:
            others = [("lemma", l) for l in lemmas]
        else:
            others = [("gloss", a["english"])]
        for node in others:
            uf.union(surface_node, node)
        anchor_nodes.append(surface_node)
    return [uf.find(n) for n in anchor_nodes]


def group_split(anchors, surface_key, val_size=VAL_SIZE, test_size=TEST_SIZE,
                seed=SEED):
    """Split anchors into (train, val, test); no group spans partitions.

    Deterministic for a given (anchors, seed). Original anchor order is
    preserved within each partition.
    """
    if not anchors:
        return [], [], []

    group_ids = build_groups(anchors, surface_key)
    groups = {}
    for idx, gid in enumerate(group_ids):
        groups.setdefault(gid, []).append(idx)

    members = sorted(groups.values(), key=lambda m: m[0])
    rng = random.Random(seed)
    rng.shuffle(members)
    # Largest first; stable sort keeps the shuffled order among equal sizes.
    members.sort(key=len, reverse=True)

    total = len(anchors)
    targets = {
        "train": (1.0 - val_size - test_size) * total,
        "val": val_size * total,
        "test": test_size * total,
    }
    assigned = {"train": 0, "val": 0, "test": 0}
    out = {"train": [], "val": [], "test": []}
    for m in members:
        part = max(("train", "val", "test"),
                   key=lambda p: targets[p] - assigned[p])
        out[part].extend(m)
        assigned[part] += len(m)

    return (
        [anchors[i] for i in sorted(out["train"])],
        [anchors[i] for i in sorted(out["val"])],
        [anchors[i] for i in sorted(out["test"])],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest shared/tests/test_anchor_split.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add shared/scripts/anchor_split.py shared/tests/test_anchor_split.py
git commit -m "feat(shared): group-aware anchor split (lemma/surface/gloss union-find, 64/16/20)"
```

---

### Task 2: Akkadian 06 — emit `lemmas` field

**Files:**
- Modify: `languages/akkadian/scripts/06_extract_anchors.py` (function `extract_oracc_anchors`, lines 76–115)
- Test: `languages/akkadian/tests/test_06_anchors.py`

**Interfaces:**
- Produces: every anchor dict in `english_anchors.json` gains `"lemmas": [<normalized cf>, ...]` (sorted, deduped). Consumed by `group_split` in Tasks 6–10.

- [ ] **Step 1: Write the failing test**

Append to `languages/akkadian/tests/test_06_anchors.py`:

The file already has an importlib `_load()` helper at the top — use it:

```python
def test_anchors_carry_contributing_lemmas():
    mod = _load()

    lemmas = [
        {"cf": "šarrum", "form": "šarri", "gw": "king"},
    ] * 5 + [
        # Same surface form under a second citation form: both cfs recorded.
        {"cf": "šarratum", "form": "šarri", "gw": "king"},
    ] * 5
    anchors = mod.extract_oracc_anchors(lemmas, min_occurrences=5)
    by_surface = {a["akkadian"]: a for a in anchors}
    assert "lemmas" in by_surface["šarri"]
    assert set(by_surface["šarri"]["lemmas"]) >= {"šarrum", "šarratum"}
    for a in anchors:
        assert a["lemmas"] == sorted(set(a["lemmas"]))
```

(The normalizer may transform `šarrum`/`šarri` — if the dict keys miss, print `anchors` once and pin the actual normalized surfaces; the invariant is that the shared surface carries both citation forms.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest languages/akkadian/tests/test_06_anchors.py -v`
Expected: the new test FAILS with `KeyError: 'lemmas'` (or assert failure); all pre-existing tests PASS.

- [ ] **Step 3: Implement**

In `extract_oracc_anchors` in `languages/akkadian/scripts/06_extract_anchors.py`, replace the pair-counting loop and emission:

```python
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_lemmas: dict[tuple[str, str], set[str]] = {}
    for lemma in lemmas:
        gw = (lemma.get("gw") or "").strip().lower()
        if not _filter_gloss(gw):
            continue
        cf = normalize_akkadian_token((lemma.get("cf") or "").strip())
        form = normalize_akkadian_token((lemma.get("form") or "").strip())
        lemma_key = cf or form
        # In-record surfaces (existing behavior)
        surfaces: set[str] = set()
        for surface in (cf, form):
            if surface:
                surfaces.update(mimation_alternates(surface))
        # L4: expand to all globally-attested surfaces for this lemma's citation form
        if cf in cf_to_surfaces:
            surfaces.update(cf_to_surfaces[cf])
        for surface in surfaces:
            pair_counts[(surface, gw)] += 1
            pair_lemmas.setdefault((surface, gw), set()).add(lemma_key)

    anchors: list[dict] = []
    for (form_norm, gw), count in pair_counts.items():
        if count < min_occurrences:
            continue
        confidence = min(0.95, 0.5 + (count / 100))
        anchors.append({
            "akkadian": form_norm,
            "english": gw,
            "confidence": round(confidence, 4),
            "frequency": count,
            "source": "ORACC",
            "lemmas": sorted(pair_lemmas[(form_norm, gw)]),
        })
    return sorted(anchors, key=lambda a: a["confidence"], reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest languages/akkadian/tests/test_06_anchors.py -v`
Expected: all PASS (including the new one).

- [ ] **Step 5: Re-run extraction and verify pair-set equivalence**

```bash
ls languages/akkadian/data/raw/ob_literary_lemmas.json \
   languages/akkadian/data/raw/ob_letters_lemmas.json \
   languages/akkadian/data/raw/dcclt_lemmas.json   # must all exist; STOP if not
cp languages/akkadian/data/processed/english_anchors.json \
   "$SCRATCH/akkadian_anchors_old.json"   # $SCRATCH = session scratchpad dir
python languages/akkadian/scripts/06_extract_anchors.py
python3 - "$SCRATCH/akkadian_anchors_old.json" \
  languages/akkadian/data/processed/english_anchors.json <<'EOF'
import json, sys
strip = lambda a: {k: v for k, v in a.items() if k != "lemmas"}
old = sorted(map(strip, json.load(open(sys.argv[1]))), key=str)
new = sorted(map(strip, json.load(open(sys.argv[2]))), key=str)
assert old == new, f"anchor content drifted: {len(old)} vs {len(new)}"
assert all("lemmas" in a for a in json.load(open(sys.argv[2])))
print(f"OK: {len(new)} anchors identical modulo lemmas field")
EOF
```

Expected: `OK: ... anchors identical modulo lemmas field`. If content drifted, STOP — raw data has changed since the shipped run; report instead of proceeding.

- [ ] **Step 6: Commit**

```bash
git add languages/akkadian/scripts/06_extract_anchors.py \
        languages/akkadian/tests/test_06_anchors.py \
        languages/akkadian/data/processed/english_anchors.json
git commit -m "feat(akkadian): anchors carry contributing lemmas (cf) for group split"
```

---

### Task 3: Sumerian 06 — emit `lemmas` field

**Files:**
- Modify: `languages/sumerian/scripts/06_extract_anchors.py` (functions `extract_epsd2_anchors` lines 26–73, `extract_cooccurrence_anchors` lines 108–126)
- Test: `languages/sumerian/tests/test_06_anchors.py`

**Interfaces:**
- Produces: ePSD2 anchors get `"lemmas": [<cf>, ...]`; ETCSL co-occurrence anchors (no citation form exists) get `"lemmas": [<surface>]` (singleton).

- [ ] **Step 1: Write the failing test**

Append to `languages/sumerian/tests/test_06_anchors.py` (match the file's existing import style for the 06 module):

```python
def test_epsd2_anchors_carry_lemmas():
    from languages.sumerian.scripts.anchors_06 import extract_epsd2_anchors

    lemmas = [{"cf": "lugal", "form": "lugal-e", "gw": "king"}] * 5
    anchors = extract_epsd2_anchors(lemmas, min_occurrences=5)
    by_surface = {a["sumerian"]: a for a in anchors}
    # Both the cf anchor and the form anchor attribute to the cf.
    assert by_surface["lugal"]["lemmas"] == ["lugal"]
    assert by_surface["lugal-e"]["lemmas"] == ["lugal"]


def test_cooccurrence_anchors_carry_singleton_lemmas():
    from languages.sumerian.scripts.anchors_06 import extract_cooccurrence_anchors

    lines = [{"transliteration": "lugal kur", "translation": "the king of the mountain"}] * 4
    anchors = extract_cooccurrence_anchors(lines, min_cooccurrences=3, min_confidence=0.3)
    assert anchors
    for a in anchors:
        assert a["lemmas"] == [a["sumerian"]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest languages/sumerian/tests/test_06_anchors.py -v`
Expected: 2 new tests FAIL with `KeyError: 'lemmas'`; existing tests PASS.

- [ ] **Step 3: Implement**

In `extract_epsd2_anchors`, replace the counting loop and the anchor emission:

```python
    pair_counts = Counter()
    pair_lemmas: dict[tuple[str, str], set[str]] = {}
    for lemma in lemmas:
        gw = lemma.get("gw", "").strip().lower()
        if not gw:
            continue
        # Count both citation form and surface form as potential anchors
        cf = normalize_sumerian_token(lemma.get("cf", "").strip())
        form = normalize_sumerian_token(lemma.get("form", "").strip())
        lemma_key = cf or form
        if cf:
            pair_counts[(cf, gw)] += 1
            pair_lemmas.setdefault((cf, gw), set()).add(lemma_key)
        if form and form != cf:
            pair_counts[(form, gw)] += 1
            pair_lemmas.setdefault((form, gw), set()).add(lemma_key)
```

and in the emission block add the field:

```python
            anchors.append({
                "sumerian": cf,
                "english": gw,
                "confidence": round(confidence, 4),
                "frequency": count,
                "source": "ePSD2",
                "lemmas": sorted(pair_lemmas[(cf, gw)]),
            })
```

In `extract_cooccurrence_anchors`, add the singleton field to its emission:

```python
            anchors.append({
                "sumerian": sw,
                "english": best_ew,
                "confidence": round(confidence, 4),
                "frequency": total,
                "source": "ETCSL",
                "lemmas": [sw],
            })
```

(`merge_anchors` copies whole records, so `lemmas` passes through unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest languages/sumerian/tests/test_06_anchors.py -v`
Expected: all PASS.

- [ ] **Step 5: Re-run extraction and verify pair-set equivalence**

Same pattern as Task 2 Step 5, with:
- raw inputs to check: `languages/sumerian/data/raw/oracc_lemmas.json`, `languages/sumerian/data/raw/etcsl_texts.json`
- backup: `$SCRATCH/sumerian_anchors_old.json`
- run: `python languages/sumerian/scripts/06_extract_anchors.py`
- diff script identical but with `a["sumerian"]` records (the strip-lemmas comparison is field-agnostic — reuse verbatim).

Expected: `OK: 13100 anchors identical modulo lemmas field`.

- [ ] **Step 6: Commit**

```bash
git add languages/sumerian/scripts/06_extract_anchors.py \
        languages/sumerian/tests/test_06_anchors.py \
        languages/sumerian/data/processed/english_anchors.json
git commit -m "feat(sumerian): anchors carry contributing lemmas for group split"
```

---

### Task 4: Hittite 06 — emit `lemmas` field

**Files:**
- Modify: `languages/hittite/scripts/06_extract_anchors.py` (`extract_german_anchors` lines 185–215, `extract_heterogram_anchors` anchor dicts at lines 250–258 and 273–281)
- Test: create `languages/hittite/tests/test_06_anchors.py`

**Interfaces:**
- Produces: TLHdig_de anchors get `"lemmas": [<normalized cf>, ...]`; heterogram anchors get `"lemmas": [<normalized surface>]`.

- [ ] **Step 1: Write the failing test**

Create `languages/hittite/tests/test_06_anchors.py`:

```python
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parent.parent / "scripts" / "06_extract_anchors.py"
_spec = spec_from_file_location("hit_anchors_06", str(_path))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_german_anchors_carry_lemmas(monkeypatch):
    import numpy as np
    lemmas = [{"cf": "pai-", "form": "pait", "gw": "gehen"}] * 5
    monkeypatch.setattr(
        _mod, "translate_german_glosses",
        lambda glosses, vec, vocab: {g: "walking" for g in glosses},
    )
    anchors = _mod.extract_german_anchors(
        lemmas, np.zeros((1, 768), dtype=np.float32), ["walking"]
    )
    assert anchors
    for a in anchors:
        assert "lemmas" in a and a["lemmas"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest languages/hittite/tests/test_06_anchors.py -v`
Expected: FAIL with `KeyError: 'lemmas'` (or empty-lemmas assert).

- [ ] **Step 3: Implement**

In `extract_german_anchors`, replace the pair-counting loop and emission:

```python
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_lemmas: dict[tuple[str, str], set[str]] = {}
    for lemma in lemmas:
        gw = (lemma.get("gw") or "").strip()
        if gw not in translations:
            continue
        english = translations[gw].lower()
        if not english or len(english) < 2:
            continue
        cf = normalize_hittite_token((lemma.get("cf") or "").strip())
        form = normalize_hittite_token((lemma.get("form") or "").strip())
        lemma_key = cf or form
        surfaces: set[str] = set()
        if cf:
            surfaces.add(cf)
        if form and form != cf:
            surfaces.add(form)
        for surface in surfaces:
            pair_counts[(surface, english)] += 1
            pair_lemmas.setdefault((surface, english), set()).add(lemma_key)

    anchors: list[dict] = []
    for (form_norm, eng), count in pair_counts.items():
        if count < min_occurrences:
            continue
        confidence = min(0.95, 0.5 + (count / 100))
        anchors.append({
            "hittite": form_norm,
            "english": eng,
            "confidence": round(confidence, 4),
            "frequency": count,
            "source": "TLHdig_de",
            "lemmas": sorted(pair_lemmas[(form_norm, eng)]),
        })
    return sorted(anchors, key=lambda a: a["confidence"], reverse=True)
```

In `extract_heterogram_anchors`, add to the Sumerogram anchor dict (after `"bridge_source_word": key,`):

```python
                            "lemmas": [normalize_hittite_token(sumerogram)],
```

and to the Akkadogram anchor dict (after `"bridge_source_word": akkadogram,`):

```python
                        "lemmas": [akk_norm],
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest languages/hittite/tests/ -v`
Expected: all PASS.

- [ ] **Step 5: Regenerate the bridge vocab prerequisite if missing**

`extract_heterogram_anchors` needs `/tmp/sumerian_vocab_for_bridge.json`. If absent, the Sumerogram bridge silently drops and the pair-set check below will fail. Recreate it from the shipped vocab:

```bash
test -f /tmp/sumerian_vocab_for_bridge.json || python3 - <<'EOF'
import json, pickle
with open("languages/sumerian/final_output/sumerian_aligned_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("/tmp/sumerian_vocab_for_bridge.json", "w") as f:
    json.dump(list(vocab), f)
print(f"wrote {len(vocab)} words")
EOF
```

- [ ] **Step 6: Re-run extraction and verify pair-set equivalence**

Same pattern as Task 2 Step 5, with:
- backup: `$SCRATCH/hittite_anchors_old.json`
- run: `python languages/hittite/scripts/06_extract_anchors.py` (uses the cached `german_to_english.json`; no model download expected — if it starts encoding with SentenceTransformer, STOP and check the cache)
- Expected: `OK: 11750 anchors identical modulo lemmas field`

- [ ] **Step 7: Commit**

```bash
git add languages/hittite/scripts/06_extract_anchors.py \
        languages/hittite/tests/test_06_anchors.py \
        languages/hittite/data/processed/english_anchors.json
git commit -m "feat(hittite): anchors carry contributing lemmas for group split"
```

---

### Task 5: Greek 06 — emit `lemmas` field

**Files:**
- Modify: `languages/greek/scripts/06_extract_anchors.py` (`extract_anchors` lines 95–149)
- Test: create `languages/greek/tests/test_06_anchors.py`

**Interfaces:**
- Produces: anchors get `"lemmas": [<cf_norm>, ...]`.

- [ ] **Step 1: Write the failing test**

Create `languages/greek/tests/test_06_anchors.py`:

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
    anchors = _mod.extract_anchors(lemmas, lsj_index, {"sea"}, min_occurrences=5)
    assert anchors
    by_surface = {a["greek"]: a for a in anchors}
    assert by_surface["θαλασσα"]["lemmas"] == ["θαλασσα"]
    assert by_surface["θαλασσης"]["lemmas"] == ["θαλασσα"]
```

(Exact normalized forms: `normalize_greek_token` strips accents/case — if the assert keys differ, print `anchors` once and pin the actual normalized strings; the invariant under test is that both surface anchors carry the *cf* lemma.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest languages/greek/tests/test_06_anchors.py -v`
Expected: FAIL with `KeyError: 'lemmas'`.

- [ ] **Step 3: Implement**

In `extract_anchors`, add lemma tracking (after `pair_counts` declaration, and in the surface loop and emission):

```python
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_lemmas: dict[tuple[str, str], set[str]] = {}
```

surface-registration loop becomes:

```python
        for surface in surfaces:
            pair_counts[(surface, english)] += 1
            pair_lemmas.setdefault((surface, english), set()).add(cf_norm)
```

emission gains:

```python
        anchors.append({
            "greek": greek_form,
            "english": eng,
            "confidence": round(confidence, 4),
            "frequency": count,
            "source": "Diorisis+LSJ",
            "lemmas": sorted(pair_lemmas[(greek_form, eng)]),
        })
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest languages/greek/tests/ -v`
Expected: all PASS.

- [ ] **Step 5: Re-run extraction and verify pair-set equivalence**

Same pattern as Task 2 Step 5, with:
- raw inputs: `languages/greek/data/raw/greek_lemmas.json`, `languages/greek/data/dictionaries/lsj_glosses.json`
- backup: `$SCRATCH/greek_anchors_old.json`
- run: `python languages/greek/scripts/06_extract_anchors.py` (10M records; a few minutes)
- Expected: `OK: 106260 anchors identical modulo lemmas field`

- [ ] **Step 6: Commit**

```bash
git add languages/greek/scripts/06_extract_anchors.py \
        languages/greek/tests/test_06_anchors.py \
        languages/greek/data/processed/english_anchors.json
git commit -m "feat(greek): anchors carry contributing lemmas for group split"
```

---

### Task 6: Akkadian alignment rewrite (canonical) + rerun

**Files:**
- Modify: `languages/akkadian/scripts/09_align_and_evaluate.py`
- Modify: `languages/akkadian/scripts/09b_align_gemma.py`
- Modify: `languages/akkadian/scripts/align_09.py`
- Delete: `languages/akkadian/scripts/ridge_alpha_sweep.py`

**Interfaces:**
- Consumes: `group_split`, `VAL_SIZE`, `TEST_SIZE`, `SEED` from Task 1; `lemmas` field from Task 2.
- Produces: `select_alpha(X_train, Y_train, X_val, val_english, eng_vocab_list, eng_vectors, alphas, predict_transform=None) -> tuple[float, list[dict]]` in `09_align_and_evaluate.py`, re-exported through `align_09.py`. Tasks 7–10 clone/consume this. Results JSONs gain `config.split` block and `config.alpha_sweep_val`.

- [ ] **Step 1: Rewrite `09_align_and_evaluate.py`**

(a) Docstring: replace the "Pipeline:" list with:

```
Pipeline:
  1. Load fused 1536d Akkadian vectors
  2. Load GloVe 300d English vectors
  3. Load anchor pairs
  4. Lemma-group 64/16/20 train/val/test split (no lemma or surface spans partitions)
  5. Select Ridge alpha by top-1 on the validation set
  6. Retrain at the chosen alpha on train+val
  7. Evaluate Top-1/5/10 accuracy on the held-out test set

OOV anchors (FastText subword inference) are training-only; validation and
test contain in-vocab anchors exclusively.
```

(b) Imports: delete `from sklearn.model_selection import train_test_split`; after the existing `from scipy.spatial.distance import cdist` add:

```python
import sys

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.scripts.anchor_split import group_split, SEED, TEST_SIZE, VAL_SIZE
```

(c) Add module constants under the path constants:

```python
SURFACE_KEY = "akkadian"
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
```

(d) Add `select_alpha` after `train_ridge`:

```python
def select_alpha(
    X_train, Y_train, X_val, val_english, eng_vocab_list, eng_vectors,
    alphas, predict_transform=None,
):
    """Pick the Ridge alpha with the best top-1 on the validation set.

    predict_transform: optional callable applied to raw predictions before
    evaluation (Egyptian's PCA path lifts 256d back to 768d with it).
    Returns (best_alpha, sweep_records).
    """
    sweep = []
    best_alpha, best_top1 = None, -1.0
    for alpha in alphas:
        model = train_ridge(X_train, Y_train, alpha=alpha)
        Y_pred = model.predict(X_val)
        if predict_transform is not None:
            Y_pred = predict_transform(Y_pred)
        acc = evaluate_alignment(Y_pred, val_english, eng_vocab_list, eng_vectors)
        sweep.append({"alpha": alpha, "accuracy": acc})
        print(f"  alpha={alpha:<10g} val top1={acc['top1']:.2f}%")
        if acc["top1"] > best_top1:
            best_alpha, best_top1 = alpha, acc["top1"]
    return best_alpha, sweep
```

(e) In `main()`, replace everything from the `X, Y, valid_anchors = build_training_data(...)` call (line ~154) through `model = train_ridge(...)` (line ~186) with:

```python
    train_anchors, val_anchors, test_anchors = group_split(
        anchors, surface_key=SURFACE_KEY
    )
    print(
        f"Group split (seed={SEED}): {len(train_anchors)} train / "
        f"{len(val_anchors)} val / {len(test_anchors)} test raw anchors"
    )

    # OOV subword inference is training-only: val/test are built WITHOUT the
    # FastText fallback so they stay in-vocab.
    X_train, Y_train, train_valid = build_training_data(
        train_anchors, sum_vocab, sum_vectors, eng_vocab, glove_vectors,
        fasttext_model=ft_model,
    )
    X_val, Y_val, val_valid = build_training_data(
        val_anchors, sum_vocab, sum_vectors, eng_vocab, glove_vectors
    )
    X_test, Y_test, test_valid = build_training_data(
        test_anchors, sum_vocab, sum_vectors, eng_vocab, glove_vectors
    )
    n_oov_train = sum(1 for a in train_valid if a.get("subword_inferred"))
    n_valid = len(train_valid) + len(val_valid) + len(test_valid)
    print(
        f"Valid anchors: {n_valid} / {len(anchors)} — "
        f"{len(train_valid)} train ({n_oov_train} OOV-inferred) / "
        f"{len(val_valid)} val / {len(test_valid)} test"
    )

    print("Selecting alpha on validation...")
    val_english = [a["english"] for a in val_valid]
    best_alpha, sweep = select_alpha(
        X_train, Y_train, X_val, val_english, glove_vocab, glove_vectors, ALPHAS
    )
    print(f"Selected alpha={best_alpha}")

    X_fit = np.concatenate([X_train, X_val], axis=0)
    Y_fit = np.concatenate([Y_train, Y_val], axis=0)
    print(f"Retraining on train+val ({len(X_fit)}) at alpha={best_alpha}...")
    model = train_ridge(X_fit, Y_fit, alpha=best_alpha)
```

(f) Replace `test_english = [a["english"] for a in anchors_test]` with `test_english = [a["english"] for a in test_valid]`.

(g) Replace the `full_results` config block with:

```python
    full_results = {
        "accuracy": results,
        "config": {
            "alignment": "Ridge",
            "alpha": best_alpha,
            "alpha_sweep_val": sweep,
            "train_size": len(X_fit),
            "test_size": len(X_test),
            "valid_anchors": n_valid,
            "total_anchors": len(anchors),
            "sumerian_vocab": len(sum_vocab),
            "fused_dim": int(sum_vectors.shape[1]),
            "glove_dim": int(glove_vectors.shape[1]),
            "split": {
                "method": "lemma-group",
                "seed": SEED,
                "val_size": VAL_SIZE,
                "test_size": TEST_SIZE,
                "raw": {"train": len(train_anchors), "val": len(val_anchors),
                        "test": len(test_anchors)},
                "valid": {"train": len(train_valid), "val": len(val_valid),
                          "test": len(test_valid)},
                "oov_train_only": n_oov_train,
            },
        },
    }
```

(Keep the legacy `"sumerian_vocab"` key — `10_export_production.py` and its test are not being touched.)

Also update the `print("Training Ridge regression (alpha=0.001)...")` line and the `train_ridge(..., alpha=0.001)  # L7 ...` comment — both are gone, replaced by the block in (e).

- [ ] **Step 2: Rewrite `09b_align_gemma.py`**

Same transformation. Specifics:
- Extend the `from languages.akkadian.scripts.align_09 import (...)` to include `select_alpha`.
- Delete `from sklearn.model_selection import train_test_split`.
- Add `from shared.scripts.anchor_split import group_split, SEED, TEST_SIZE, VAL_SIZE` (the `_ROOT` sys.path block already exists in 09b).
- Replace `RIDGE_ALPHA = 0.01  # L7: ...`, `TEST_SIZE = 0.2`, `RANDOM_STATE = 42` with:

```python
SURFACE_KEY = "akkadian"
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
```

- In `main()`, replace everything from `X, Y, valid_anchors = build_training_data(...)` (line ~121) through `model = train_ridge(X_train, Y_train, alpha=RIDGE_ALPHA)` (line ~156) with the identical block from Step 1(e), except the English target variables are `eng_vocab`, `eng_vectors`, `eng_vocab_list` (so the three `build_training_data` calls pass `eng_vocab, eng_vectors` and `select_alpha` receives `eng_vocab_list, eng_vectors`).
- `test_english = [a["english"] for a in test_valid]`.
- Config block: replace `"alpha": RIDGE_ALPHA, "test_size": TEST_SIZE, "random_state": RANDOM_STATE, "train_size": len(X_train), "test_size_count": len(X_test),` with:

```python
            "alpha": best_alpha,
            "alpha_sweep_val": sweep,
            "train_size": len(X_fit),
            "test_size_count": len(X_test),
```

and add the same `"split": {...}` block as Step 1(g) inside `config`.

- [ ] **Step 3: Export `select_alpha` through the shim**

In `languages/akkadian/scripts/align_09.py` append:

```python
select_alpha = _mod.select_alpha
```

- [ ] **Step 4: Delete the test-set sweep script**

```bash
git rm languages/akkadian/scripts/ridge_alpha_sweep.py
```

- [ ] **Step 5: Sanity-check imports and existing tests**

```bash
python -c "from languages.akkadian.scripts.align_09 import select_alpha; print('ok')"
python -m pytest languages/akkadian/tests/ shared/tests/ -v
```
Expected: `ok`, all tests PASS.

- [ ] **Step 6: Run the alignment (GloVe, then whitened Gemma)**

```bash
python languages/akkadian/scripts/09_align_and_evaluate.py
python languages/akkadian/scripts/09b_align_gemma.py --mode whitened
python languages/akkadian/scripts/10_export_production.py
```

Expected: each prints the group-split sizes, the 10-alpha val sweep, selected alpha, and test top-1/5/10. Record the numbers (old Gemma top-1 was 36.43%, GloVe 27.79% — new numbers will be lower; that is expected and correct). Verify `results/alignment_results.json` and `results/alignment_results_gemma_whitened.json` contain the `split` block and the real `alpha`.

- [ ] **Step 7: Commit**

```bash
git add -A languages/akkadian
git commit -m "feat(akkadian): lemma-group split + val-selected alpha; retire test-set sweep"
```

---

### Task 7: Hittite alignment clone + rerun

**Files:**
- Modify: `languages/hittite/scripts/09_align_and_evaluate.py`, `languages/hittite/scripts/09b_align_gemma.py`, `languages/hittite/scripts/align_09.py`
- Delete: `languages/hittite/scripts/ridge_alpha_sweep.py`

Hittite's three files are byte-identical to Akkadian's except `akkadian`→`hittite` tokens (verified by diff before this plan was written). Clone the rewritten Akkadian files mechanically.

- [ ] **Step 1: Clone the rewritten scripts**

```bash
sed -e 's/akkadian/hittite/g' -e 's/Akkadian/Hittite/g' \
  languages/akkadian/scripts/09_align_and_evaluate.py \
  > languages/hittite/scripts/09_align_and_evaluate.py
sed -e 's/akkadian/hittite/g' -e 's/Akkadian/Hittite/g' \
  languages/akkadian/scripts/09b_align_gemma.py \
  > languages/hittite/scripts/09b_align_gemma.py
```

- [ ] **Step 2: Verify the clone is exact**

```bash
diff <(sed -e 's/akkadian/hittite/g' -e 's/Akkadian/Hittite/g' \
  languages/akkadian/scripts/09_align_and_evaluate.py) \
  languages/hittite/scripts/09_align_and_evaluate.py
diff <(sed -e 's/akkadian/hittite/g' -e 's/Akkadian/Hittite/g' \
  languages/akkadian/scripts/09b_align_gemma.py) \
  languages/hittite/scripts/09b_align_gemma.py
```
Expected: both diffs empty. (The substitutions only touch `akkadian`/`Akkadian` tokens — the legacy `"sumerian_vocab"` config key and the `fasttext_sumerian.model` filename are deliberately untouched, matching the pre-existing per-language delta.)

- [ ] **Step 3: Shim + delete sweep**

Append `select_alpha = _mod.select_alpha` to `languages/hittite/scripts/align_09.py`, then:

```bash
git rm languages/hittite/scripts/ridge_alpha_sweep.py
python -c "from languages.hittite.scripts.align_09 import select_alpha; print('ok')"
python -m pytest languages/hittite/tests/ -v
```
Expected: `ok`, tests PASS.

- [ ] **Step 4: Run**

```bash
python languages/hittite/scripts/09_align_and_evaluate.py
python languages/hittite/scripts/09b_align_gemma.py --mode whitened
python languages/hittite/scripts/10_export_production.py
```
Expected: split sizes, val sweep, selected alpha, test metrics (old: 40.62% Gemma / 35.40% GloVe; new lower). Record numbers.

- [ ] **Step 5: Commit**

```bash
git add -A languages/hittite
git commit -m "feat(hittite): lemma-group split + val-selected alpha; retire test-set sweep"
```

---

### Task 8: Greek alignment clone (no run)

**Files:**
- Modify: `languages/greek/scripts/09_align_and_evaluate.py`, `languages/greek/scripts/09b_align_gemma.py`, `languages/greek/scripts/align_09.py`
- Delete: `languages/greek/scripts/ridge_alpha_sweep.py`

- [ ] **Step 1: Clone + verify + shim + delete** — identical procedure to Task 7 Steps 1–3 with `hittite`→`greek`, `Hittite`→`Greek` in the commands:

```bash
sed -e 's/akkadian/greek/g' -e 's/Akkadian/Greek/g' \
  languages/akkadian/scripts/09_align_and_evaluate.py \
  > languages/greek/scripts/09_align_and_evaluate.py
sed -e 's/akkadian/greek/g' -e 's/Akkadian/Greek/g' \
  languages/akkadian/scripts/09b_align_gemma.py \
  > languages/greek/scripts/09b_align_gemma.py
# verify: same diff pattern as Task 7 Step 2, expect empty
echo 'select_alpha = _mod.select_alpha' >> languages/greek/scripts/align_09.py
git rm languages/greek/scripts/ridge_alpha_sweep.py
python -c "from languages.greek.scripts.align_09 import select_alpha; print('ok')"
python -m pytest languages/greek/tests/ -v
```

Per approved scope, Greek's alignment is NOT run (its first alignment run is a separate task).

- [ ] **Step 2: Commit**

```bash
git add -A languages/greek
git commit -m "feat(greek): lemma-group split + val-selected alpha (scripts only, first run pending)"
```

---

### Task 9: Sumerian alignment rewrite + rerun

**Files:**
- Modify: `languages/sumerian/scripts/09_align_and_evaluate.py`, `languages/sumerian/scripts/09b_align_gemma.py`, `languages/sumerian/scripts/align_09.py`
- Delete: `languages/sumerian/scripts/ridge_alpha_sweep.py`

Sumerian differs from Akkadian: **no FastText/L5 subword inference** (its `build_training_data` has no `fasttext_model` param), and 09 lacks the `_ROOT` sys.path block. Also fixes the alpha-recording bug (trains alpha=100, records 0.001).

- [ ] **Step 1: Rewrite `09_align_and_evaluate.py`**

Apply Task 6 Step 1 (a)–(g) with these deltas:
- (a) docstring: "Sumerian" wording; step 4 is "Lemma-group 64/16/20 train/val/test split"; no OOV sentence.
- (c) `SURFACE_KEY = "sumerian"`.
- (e) replacement block — no FastText, all three partitions built identically:

```python
    train_anchors, val_anchors, test_anchors = group_split(
        anchors, surface_key=SURFACE_KEY
    )
    print(
        f"Group split (seed={SEED}): {len(train_anchors)} train / "
        f"{len(val_anchors)} val / {len(test_anchors)} test raw anchors"
    )

    X_train, Y_train, train_valid = build_training_data(
        train_anchors, sum_vocab, sum_vectors, eng_vocab, glove_vectors
    )
    X_val, Y_val, val_valid = build_training_data(
        val_anchors, sum_vocab, sum_vectors, eng_vocab, glove_vectors
    )
    X_test, Y_test, test_valid = build_training_data(
        test_anchors, sum_vocab, sum_vectors, eng_vocab, glove_vectors
    )
    n_valid = len(train_valid) + len(val_valid) + len(test_valid)
    print(
        f"Valid anchors: {n_valid} / {len(anchors)} — "
        f"{len(train_valid)} train / {len(val_valid)} val / {len(test_valid)} test"
    )

    print("Selecting alpha on validation...")
    val_english = [a["english"] for a in val_valid]
    best_alpha, sweep = select_alpha(
        X_train, Y_train, X_val, val_english, glove_vocab, glove_vectors, ALPHAS
    )
    print(f"Selected alpha={best_alpha}")

    X_fit = np.concatenate([X_train, X_val], axis=0)
    Y_fit = np.concatenate([Y_train, Y_val], axis=0)
    print(f"Retraining on train+val ({len(X_fit)}) at alpha={best_alpha}...")
    model = train_ridge(X_fit, Y_fit, alpha=best_alpha)
```

- (g) config block: same as Task 6 but no `"oov_train_only"` key, and the split block's `"valid"` dict as-is. `"alpha": best_alpha` replaces the hard-coded `"alpha": 0.001` — **this closes the recorded-alpha bug**. Also fix the docstring line 10 (`alpha=0.001`) which is removed by (a).

Add `select_alpha` exactly as Task 6 Step 1(d) (it is language-independent).

- [ ] **Step 2: Rewrite `09b_align_gemma.py`**

As Task 6 Step 2, with: import from `languages.sumerian.scripts.align_09`; `SURFACE_KEY = "sumerian"`; no FastText loading (delete nothing — sumerian 09b has no FastText block); replace `RIDGE_ALPHA = 100`, `TEST_SIZE = 0.2`, `RANDOM_STATE = 42` with `SURFACE_KEY`/`ALPHAS`; replacement block is Step 1(e) above with `eng_vocab, eng_vectors` / `eng_vocab_list`.

- [ ] **Step 3: Shim, delete sweep, test**

```bash
echo 'select_alpha = _mod.select_alpha' >> languages/sumerian/scripts/align_09.py
git rm languages/sumerian/scripts/ridge_alpha_sweep.py
python -c "from languages.sumerian.scripts.align_09 import select_alpha; print('ok')"
python -m pytest languages/sumerian/tests/ -v
```
Expected: `ok`, all PASS (helper signatures unchanged).

- [ ] **Step 4: Run**

```bash
python languages/sumerian/scripts/09_align_and_evaluate.py
python languages/sumerian/scripts/09b_align_gemma.py --mode whitened
python languages/sumerian/scripts/10_export_production.py
```
Expected: new honest metrics (old: 52.13% Gemma / 35.70% GloVe; new lower). Record numbers; verify recorded `alpha` in `results/alignment_results.json` equals the selected one.

- [ ] **Step 5: Commit**

```bash
git add -A languages/sumerian
git commit -m "feat(sumerian): lemma-group split + val-selected alpha; fix recorded-alpha bug"
```

---

### Task 10: Egyptian alignment rewrite + rerun

**Files:**
- Modify: `languages/egyptian/scripts/09_align_and_evaluate.py`, `languages/egyptian/scripts/09b_align_gemma.py`, `languages/egyptian/scripts/align_09.py`

Egyptian anchors have no `lemmas` field → `group_split` groups by gloss automatically (`"method": "gloss-group"`). Surface key is `egyptian_raw`. 09b keeps its PCA-256 path; its `--sweep` flag and `SWEEP_ALPHAS` are removed (val selection is now always on).

- [ ] **Step 1: Rewrite `09_align_and_evaluate.py`**

As Task 9 Step 1 with: `SURFACE_KEY = "egyptian_raw"`; variables `eg_vocab, eg_vectors`; `"method": "gloss-group"` in the split block; keep the legacy `"egyptian_vocab"` config key; delete `RIDGE_ALPHA = 0.1`, `TEST_SIZE = 0.2`, `RANDOM_STATE = 42` constants (replaced by `SURFACE_KEY`/`ALPHAS`); add the `select_alpha` function from Task 6 Step 1(d) and the `_ROOT` sys.path block + `from shared.scripts.anchor_split import group_split, SEED, TEST_SIZE, VAL_SIZE` import (note: Egyptian 09 defines `_REPO_ROOT` already — reuse it for the sys.path insert instead of adding `_ROOT`).

- [ ] **Step 2: Rewrite `09b_align_gemma.py`**

- Imports: add `select_alpha` to the `align_09` import; delete `train_test_split` import; add the `anchor_split` import (`_REPO_ROOT` sys.path block already present).
- Delete `RIDGE_ALPHA = 1.0` and `SWEEP_ALPHAS = [...]`; delete `TEST_SIZE`/`RANDOM_STATE`; add `SURFACE_KEY = "egyptian_raw"` and the global `ALPHAS` list.
- Remove the `--sweep` argparse flag and the entire `if args.sweep:` block.
- Replace the block from `X, Y_full, valid_anchors = build_training_data(...)` through `model = train_ridge(X_train, Y_train, alpha=RIDGE_ALPHA)` with:

```python
    train_anchors, val_anchors, test_anchors = group_split(
        anchors, surface_key=SURFACE_KEY
    )
    print(
        f"Group split (seed={SEED}): {len(train_anchors)} train / "
        f"{len(val_anchors)} val / {len(test_anchors)} test raw anchors"
    )

    X_train, Yf_train, train_valid = build_training_data(
        train_anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors
    )
    X_val, Yf_val, val_valid = build_training_data(
        val_anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors
    )
    X_test, Yf_test, test_valid = build_training_data(
        test_anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors
    )
    n_valid = len(train_valid) + len(val_valid) + len(test_valid)
    print(
        f"Valid anchors: {n_valid} / {len(anchors)} — "
        f"{len(train_valid)} train / {len(val_valid)} val / {len(test_valid)} test"
    )
    print(f"Target reduced: {Yf_train.shape[1]}d -> {PCA_COMPONENTS}d")

    print("Selecting alpha on validation (predictions lifted to 768d)...")
    val_english = [a["english"] for a in val_valid]
    best_alpha, sweep = select_alpha(
        X_train, pca.transform(Yf_train), X_val, val_english,
        eng_vocab_list, eng_vectors, ALPHAS,
        predict_transform=pca.inverse_transform,
    )
    print(f"Selected alpha={best_alpha}")

    X_fit = np.concatenate([X_train, X_val], axis=0)
    Y_fit = pca.transform(np.concatenate([Yf_train, Yf_val], axis=0))
    print(f"Retraining on train+val ({len(X_fit)}) at alpha={best_alpha}...")
    model = train_ridge(X_fit, Y_fit, alpha=best_alpha)
```

- Test evaluation: `Y_pred = pca.inverse_transform(model.predict(X_test))`, `test_english = [a["english"] for a in test_valid]` (delete the old `Y_test_full = pca.inverse_transform(Y_test)` line — unused).
- Config: `"alpha": best_alpha`, `"alpha_sweep_val": sweep`, `"train_size": len(X_fit)`, `"test_size_count": len(X_test)`, `"valid_anchors": n_valid`, plus the `"split"` block with `"method": "gloss-group"` (no `oov_train_only`).

- [ ] **Step 3: Shim + tests**

```bash
echo 'select_alpha = _mod.select_alpha' >> languages/egyptian/scripts/align_09.py
python -m pytest languages/egyptian/tests/ -v
```
Expected: all PASS (`test_09_alignment.py` imports only unchanged helpers).

- [ ] **Step 4: Run**

```bash
python languages/egyptian/scripts/09_align_and_evaluate.py
python languages/egyptian/scripts/09b_align_gemma.py
python languages/egyptian/scripts/10_export_production.py
```
Expected: gloss-group split (the "the" mega-group lands in train — test will contain few/no stopword glosses), val sweep, new metrics (old: 34.57% Gemma / 33.42% GloVe; new numbers not directly comparable since the test distribution changes too). Record numbers.

- [ ] **Step 5: Commit**

```bash
git add -A languages/egyptian
git commit -m "feat(egyptian): gloss-group split + val-selected alpha (PCA path preserved)"
```

---

### Task 11: Leak regression check + full test suite

**Files:**
- Create: `$SCRATCH/leak_check.py` (scratchpad; not committed)

- [ ] **Step 1: Write the leak checker**

```python
"""Re-measure same-gloss edit-distance<=1 leakage between (train+val) and test."""
import json
import sys

sys.path.insert(0, ".")
from shared.scripts.anchor_split import group_split


def ed_le_1(a, b):
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = diff = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            diff += 1
            if diff > 1:
                return False
            j += 1
    return True


CONFIGS = [
    ("sumerian", "languages/sumerian/data/processed/english_anchors.json", "sumerian"),
    ("akkadian", "languages/akkadian/data/processed/english_anchors.json", "akkadian"),
    ("hittite", "languages/hittite/data/processed/english_anchors.json", "hittite"),
    ("greek", "languages/greek/data/processed/english_anchors.json", "greek"),
    ("egyptian", "languages/egyptian/data/processed/english_anchors_normalized.json", "egyptian_raw"),
]

for name, path, key in CONFIGS:
    anchors = json.load(open(path))
    train, val, test = group_split(anchors, surface_key=key)
    by_gloss = {}
    for a in train + val:
        by_gloss.setdefault(a["english"], []).append(a[key])
    leaked = 0
    for a in test:
        cands = by_gloss.get(a["english"], [])
        if any(ed_le_1(a[key], c) for c in cands):
            leaked += 1
    pct = leaked / len(test) * 100 if test else 0.0
    print(f"{name:<10} test={len(test):>6}  same-gloss ed<=1 leak: {leaked} ({pct:.2f}%)")
```

- [ ] **Step 2: Run it**

Run: `python3 $SCRATCH/leak_check.py`
Expected: leak rates collapse from the measured 29.5–65.2% to ~0–3% for the lemma-grouped languages (any residue is cross-lemma coincidence, not variant leakage) and **exactly 0.00%** for Egyptian (gloss grouping makes same-gloss leakage impossible). If a lemma-grouped language shows >5%, STOP and investigate `build_groups` before writing the journal. Record all five numbers for the journal entry.

- [ ] **Step 3: Full suite**

Run: `python -m pytest -q`
Expected: all tests pass (282 pre-existing + ~12 new). `test_10_export` / `test_integration` read regenerated artifacts — if one fails on a changed artifact shape, inspect whether the assertion pinned old split fields and update the assertion, not the pipeline.

---

### Task 12: Journal entry + wrap-up

**Files:**
- Modify: `docs/EXPERIMENT_JOURNAL.md` (prepend entry, matching the existing entry format in that file)

- [ ] **Step 1: Write the journal entry**

Prepend an entry dated 2026-07-01 titled **"Eval integrity: lemma-group split + validation-selected alpha (all slots)"** containing:
- The two flaws (surface-variant leakage with the measured per-language leak rates: Akkadian 56.9%, Hittite 43.3%, Sumerian 32.0%, Egyptian 29.5%, Greek 65.2%; and test-set alpha tuning).
- The fix: union-find lemma/surface groups (gloss fallback for Egyptian), 64/16/20, val-selected alpha from the widened grid (1e-4 floor), train+val retrain, test-only reporting; OOV anchors train-only; `ridge_alpha_sweep.py` retired; Sumerian recorded-alpha bug fixed.
- Before/after table — fill with the numbers recorded in Tasks 6, 7, 9, 10 (old values: Sumerian 52.13/35.70, Akkadian 36.43/27.79, Hittite 40.62/35.40, Egyptian 34.57/33.42 — Gemma/GloVe top-1) plus selected alphas and post-fix leak-check rates from Task 11.
- Explicit note that new numbers are NOT comparable to previously published ones (old numbers measured variant memorization) and that per-language READMEs/root README still quote pre-fix numbers pending the docs-refresh task.

- [ ] **Step 2: Final verification + commit**

```bash
python -m pytest -q   # expected: all pass
git add docs/EXPERIMENT_JOURNAL.md
git commit -m "docs: journal entry — lemma-group split + val-selected alpha, corrected metrics"
git log --oneline -12   # expected: one commit per task above
```
