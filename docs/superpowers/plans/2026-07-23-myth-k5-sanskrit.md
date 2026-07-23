# Myth K=5 (Sanskrit Fourth Slot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Sanskrit as the myth study's fourth slot with a five-theme roster, re-measure the whole study on suite-v2 spaces, and read out the pre-registered K=5 ladder RSA and Indra-Vṛtra positive control.

**Architecture:** Surgical extension of `shared/scripts/myth_study.py` (slot list, combinatorial pair enumeration, a Sanskrit roster branch with pinned DCS IDs and a `vrtra` merge, extended IE-gradient, new `vrtra_control` block on the existing doc-profile/null machinery) plus one-line Sanskrit registration in `doc_eval._slot_documents`. Then one deterministic re-run and the docs.

**Tech Stack:** Python 3.12, numpy, scipy (existing myth_study deps), pytest.

**Spec:** `docs/superpowers/specs/2026-07-23-myth-k5-sanskrit-design.md`

## Global Constraints

- Branch `myth-k5`. All commands from repo root. Full `pytest -q` green before every commit (baseline 385).
- Pre-registered read-outs are FIXED (spec §Pre-registered): K=5 pair is sanskrit-sumerian (sanskrit-hittite caps at K=4); RSA positive = exhaustive p ≤ 0.05 with ρ > 0; Vṛtra control bands ≥90th pctile supports / ≤75th fails / between inconclusive — per sub-control (vs Illuyanka, vs Theogony), verbatim to the journal, no post-hoc adjustment.
- Seed and `N_NULL_DRAWS = 1000` unchanged. Existing slots' roster rules unchanged.
- Pinned Sanskrit roster (from the DCS inventory, verified 2026-07-23):
  - cosmogonic: `dcs-450-9905` (ṚV 10.129), `dcs-450-9864` (10.90), `dcs-450-9896` (10.121), `dcs-450-9978` (10.190) + merged `vrtra` = `dcs-450-10015` (1.32) + `dcs-450-10119` (1.80) + `dcs-450-10060` (2.12)
  - hymnic: `dcs-450-11102` (9.97), `dcs-450-10579` (1.164), `dcs-450-11071` (9.86), `dcs-450-9859` (10.85), `dcs-450-10697` (6.16)
  - wisdom (grouped by text_name): Jaiminīya-Upaniṣad-Brāhmaṇa, Bṛhadāraṇyakopaniṣad, Chāndogyopaniṣad, Taittirīyopaniṣad, Kaṭhopaniṣad
  - royal_control: `dcs-464-10474` (AVŚ 3.3), `dcs-464-10475` (3.4), `dcs-464-10522` (4.8), `dcs-464-10546` (4.22), `dcs-464-11445` (6.87)
  - magical (rule): 5 longest Atharvaveda (Śaunaka) chapters whose book number ∉ {14, 18} (wedding/funerary liturgy, not charms) and not in the royal set — expected members 11.3, 17.1, 12.1, 12.3 + next longest.
- Missing pinned IDs → `ValueError` listing them; pinned doc with zero in-vocab tokens → stop-and-surface (run task), never silent.
- Concept coverage verified: all 10 CONCEPTS present in `shared/models/english_gemma_whitened_768d.npz` vocab.
- `shared/results/myth_study.json` + `myth_study_roster.json` are git-TRACKED — the run task commits the refreshed files.

---

### Task 1: Sanskrit in `doc_eval._slot_documents`

**Files:**
- Modify: `shared/scripts/doc_eval.py:177-195` (`_slot_documents`)
- Test: `shared/tests/test_doc_eval_sanskrit.py` (new)

**Interfaces:**
- Produces: `_slot_documents("sanskrit") -> dict[p_number, tokens]` over `languages/sanskrit/data/raw/sanskrit_texts.json` with `normalize_sanskrit_token`.

- [ ] **Step 1: Write the failing test**

`shared/tests/test_doc_eval_sanskrit.py`:

```python
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
_spec = spec_from_file_location("doc_eval_mod", str(_ROOT / "shared" / "scripts" / "doc_eval.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_sanskrit_documents_load(tmp_path, monkeypatch):
    fixture = [
        {"p_number": "dcs-450-9905", "text_name": "Ṛgveda", "chapter": "ṚV, 10, 129",
         "lines": ["nāsad āsīt no sad āsīt tadānīm"], "source": "DCS"},
        {"p_number": "dcs-450-9864", "text_name": "Ṛgveda", "chapter": "ṚV, 10, 90",
         "lines": ["sahasraśīrṣā puruṣaḥ"], "source": "DCS"},
    ]
    p = tmp_path / "sanskrit_texts.json"
    p.write_text(json.dumps(fixture), encoding="utf-8")
    monkeypatch.setattr(_mod, "SANSKRIT_TEXTS_PATH", p)
    docs = _mod._slot_documents("sanskrit")
    assert set(docs) == {"dcs-450-9905", "dcs-450-9864"}
    # normalizer applied: lowercase IAST, diacritics preserved
    assert "nāsad" in docs["dcs-450-9905"]
    assert "sahasraśīrṣā" in docs["dcs-450-9864"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest shared/tests/test_doc_eval_sanskrit.py -v`
Expected: FAIL (`SANSKRIT_TEXTS_PATH` undefined / "sanskrit" unknown slot).

- [ ] **Step 3: Implement**

In `shared/scripts/doc_eval.py`: read the existing `_slot_documents` (hittite/greek entries ~lines 177-195) first. Add a module-level constant next to the other corpus paths:

```python
SANSKRIT_TEXTS_PATH = _ROOT / "languages" / "sanskrit" / "data" / "raw" / "sanskrit_texts.json"
```

and extend `_slot_documents`'s registry with a sanskrit entry that mirrors the hittite/greek generic path exactly (same dict shape: path + normalizer), importing the normalizer the same way the module imports the other slot normalizers:

```python
from languages.sanskrit.scripts.sanskrit_normalize import normalize_sanskrit_token
```

(match the file's existing import placement/pattern for the hittite/greek normalizers; if they are imported lazily inside the function, do the same). The sanskrit branch must read `SANSKRIT_TEXTS_PATH` (module attribute, so tests can monkeypatch) and produce `{p_number: [normalized tokens]}` exactly like hittite/greek.

- [ ] **Step 4: Run tests**

Run: `pytest shared/tests/test_doc_eval_sanskrit.py -v` then `pytest shared/tests/ -q`
Expected: new test passes; no regressions.

- [ ] **Step 5: Full pytest + commit**

```bash
pytest -q
git add shared/scripts/doc_eval.py shared/tests/test_doc_eval_sanskrit.py
git commit -m "feat(shared): sanskrit slot in doc_eval document loader"
```

---

### Task 2: Sanskrit roster branch + slot/pair extension in `myth_study.py`

**Files:**
- Modify: `shared/scripts/myth_study.py` (SLOTS `:38`, roster constants near `:50-74`, `build_roster()` `:127-264`, `slot_pairs` `:443-444`)
- Test: `shared/tests/test_myth_study_roster.py` (new)

**Interfaces:**
- Consumes: Task 1's `_slot_documents("sanskrit")`.
- Produces: `roster["sanskrit"]` with all five themes; `SANSKRIT_COSMOGONIC`, `SANSKRIT_MERGES`, `SANSKRIT_HYMNIC`, `SANSKRIT_WISDOM_TEXTS`, `SANSKRIT_ROYAL`, `SANSKRIT_MAGICAL_EXCLUDED_BOOKS` constants; `slot_pairs` = 6 combinations.

- [ ] **Step 1: Write the failing tests**

`shared/tests/test_myth_study_roster.py`:

```python
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent.parent
_spec = spec_from_file_location("myth_study_mod", str(_ROOT / "shared" / "scripts" / "myth_study.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_slots_and_pairs():
    assert _mod.SLOTS == ("sumerian", "hittite", "greek", "sanskrit")
    pairs = _mod.enumerate_slot_pairs()
    assert len(pairs) == 6
    assert ("greek", "sanskrit") in pairs or ("sanskrit", "greek") in [tuple(p) for p in pairs]


def _fixture_texts():
    docs = []
    # pinned cosmogonic + vrtra + hymnic + royal chapters (content irrelevant)
    for p in (_mod.SANSKRIT_COSMOGONIC + tuple(x for m in _mod.SANSKRIT_MERGES.values() for x in m)
              + _mod.SANSKRIT_HYMNIC + _mod.SANSKRIT_ROYAL):
        book = "3" if p.startswith("dcs-464") else "10"
        docs.append({"p_number": p, "text_name": "Ṛgveda" if p.startswith("dcs-450") else "Atharvaveda (Śaunaka)",
                     "chapter": f"X, {book}, 1", "lines": ["indraḥ vṛtram ahan"] * 3, "source": "DCS"})
    # wisdom texts, two chapters each
    for name in _mod.SANSKRIT_WISDOM_TEXTS:
        for i in (1, 2):
            docs.append({"p_number": f"dcs-999-{abs(hash(name)) % 10000}{i}", "text_name": name,
                         "chapter": f"U, {i}", "lines": ["ātmā vai idam"] * 4, "source": "DCS"})
    # AV magical candidates: two long non-royal chapters in allowed books + one excluded-book chapter
    docs.append({"p_number": "dcs-464-90001", "text_name": "Atharvaveda (Śaunaka)",
                 "chapter": "AVŚ, 11, 3", "lines": ["ucchiṣṭaḥ"] * 50, "source": "DCS"})
    docs.append({"p_number": "dcs-464-90002", "text_name": "Atharvaveda (Śaunaka)",
                 "chapter": "AVŚ, 12, 1", "lines": ["bhūmiḥ"] * 40, "source": "DCS"})
    docs.append({"p_number": "dcs-464-90003", "text_name": "Atharvaveda (Śaunaka)",
                 "chapter": "AVŚ, 18, 4", "lines": ["funerary"] * 100, "source": "DCS"})
    return docs


def test_sanskrit_roster_from_fixture(tmp_path, monkeypatch):
    p = tmp_path / "sanskrit_texts.json"
    p.write_text(json.dumps(_fixture_texts()), encoding="utf-8")
    monkeypatch.setattr(_mod, "SANSKRIT_TEXTS_PATH", p)
    roster, tokens = _mod.build_sanskrit_roster()
    themes = set(roster)
    assert themes == {"cosmogonic", "hymnic", "wisdom", "royal_control", "magical"}
    cos_ids = {d["doc_id"] for d in roster["cosmogonic"]}
    assert set(_mod.SANSKRIT_COSMOGONIC) <= cos_ids and "vrtra" in cos_ids
    # vrtra merge concatenates all member tokens
    n_member_lines = sum(3 for _ in _mod.SANSKRIT_MERGES["vrtra"])
    assert len(tokens["vrtra"]) == n_member_lines * 3  # 3 tokens per line
    # wisdom grouped by text_name → one doc per text
    assert len(roster["wisdom"]) == len(_mod.SANSKRIT_WISDOM_TEXTS)
    # magical excludes books 14/18 and royal picks
    mag_ids = {d["doc_id"] for d in roster["magical"]}
    assert "dcs-464-90003" not in mag_ids and "dcs-464-90001" in mag_ids
    assert not (mag_ids & set(_mod.SANSKRIT_ROYAL))


def test_missing_pinned_id_raises(tmp_path, monkeypatch):
    docs = _fixture_texts()
    docs = [d for d in docs if d["p_number"] != _mod.SANSKRIT_COSMOGONIC[0]]
    p = tmp_path / "sanskrit_texts.json"
    p.write_text(json.dumps(docs), encoding="utf-8")
    monkeypatch.setattr(_mod, "SANSKRIT_TEXTS_PATH", p)
    with pytest.raises(ValueError, match=_mod.SANSKRIT_COSMOGONIC[0]):
        _mod.build_sanskrit_roster()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest shared/tests/test_myth_study_roster.py -v`
Expected: FAIL (constants/functions undefined).

- [ ] **Step 3: Implement**

In `shared/scripts/myth_study.py`:

(a) `SLOTS = ("sumerian", "hittite", "greek", "sanskrit")` (line 38 area) and next to it:

```python
def enumerate_slot_pairs():
    """All unordered slot pairs, deterministic order from SLOTS."""
    return [(a, b) for i, a in enumerate(SLOTS) for b in SLOTS[i + 1:]]
```

Replace the hardcoded `slot_pairs = [...]` at ~:443 with `slot_pairs = enumerate_slot_pairs()`.

(b) Constants block (next to the other slot constants, values verbatim from Global Constraints):

```python
# --- Sanskrit roster (DCS; pinned 2026-07-23, spec 2026-07-23) ---
SANSKRIT_TEXTS_PATH = _ROOT / "languages" / "sanskrit" / "data" / "raw" / "sanskrit_texts.json"
SANSKRIT_COSMOGONIC = ("dcs-450-9905", "dcs-450-9864", "dcs-450-9896", "dcs-450-9978")
SANSKRIT_MERGES = {"vrtra": ("dcs-450-10015", "dcs-450-10119", "dcs-450-10060")}
SANSKRIT_HYMNIC = ("dcs-450-11102", "dcs-450-10579", "dcs-450-11071", "dcs-450-9859", "dcs-450-10697")
SANSKRIT_WISDOM_TEXTS = ("Jaiminīya-Upaniṣad-Brāhmaṇa", "Bṛhadāraṇyakopaniṣad",
                         "Chāndogyopaniṣad", "Taittirīyopaniṣad", "Kaṭhopaniṣad")
SANSKRIT_ROYAL = ("dcs-464-10474", "dcs-464-10475", "dcs-464-10522", "dcs-464-10546", "dcs-464-11445")
SANSKRIT_MAGICAL_EXCLUDED_BOOKS = {"14", "18"}   # AV wedding/funerary liturgy, not charms
SANSKRIT_AV_NAME = "Atharvaveda (Śaunaka)"
```

(c) `build_sanskrit_roster()` — a standalone function called from `build_roster()`'s new sanskrit branch (keeps the branch reviewable and unit-testable):

```python
def build_sanskrit_roster():
    """Sanskrit five-theme roster. Returns (roster_by_theme, tokens_by_doc).

    Chapter-level docs come straight from the raw corpus (DCS chapters are
    hymn-granular); wisdom texts are grouped by text_name; the vrtra doc is
    a HITTITE_MERGES-style concatenation. Raises ValueError naming any
    pinned ID absent from the corpus.
    """
    with open(SANSKRIT_TEXTS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    by_p = {t["p_number"]: t for t in raw}

    pinned = (SANSKRIT_COSMOGONIC + tuple(x for m in SANSKRIT_MERGES.values() for x in m)
              + SANSKRIT_HYMNIC + SANSKRIT_ROYAL)
    missing = [p for p in pinned if p not in by_p]
    if missing:
        raise ValueError(f"sanskrit roster: pinned DCS ids missing from corpus: {missing}")

    def toks(entry):
        return [normalize_sanskrit_token(w) for line in entry["lines"] for w in line.split()]

    tokens: dict[str, list[str]] = {}
    roster: dict[str, list[dict]] = {t: [] for t in
                                     ("cosmogonic", "hymnic", "wisdom", "royal_control", "magical")}

    def add(theme, doc_id, token_list, label):
        tokens[doc_id] = token_list
        roster[theme].append({"doc_id": doc_id, "label": label, "n_tokens": len(token_list)})

    for p in SANSKRIT_COSMOGONIC:
        add("cosmogonic", p, toks(by_p[p]), by_p[p]["chapter"])
    for merge_id, members in SANSKRIT_MERGES.items():
        merged = [w for m in members for w in toks(by_p[m])]
        add("cosmogonic", merge_id, merged, "+".join(by_p[m]["chapter"] for m in members))
    for p in SANSKRIT_HYMNIC:
        add("hymnic", p, toks(by_p[p]), by_p[p]["chapter"])
    for name in SANSKRIT_WISDOM_TEXTS:
        chapters = [t for t in raw if t["text_name"] == name]
        if not chapters:
            raise ValueError(f"sanskrit roster: wisdom text absent from corpus: {name}")
        merged = [w for c in chapters for w in toks(c)]
        add("wisdom", name, merged, name)
    for p in SANSKRIT_ROYAL:
        add("royal_control", p, toks(by_p[p]), by_p[p]["chapter"])

    def av_book(entry):
        parts = [s.strip() for s in entry["chapter"].split(",")]
        return parts[1] if len(parts) > 1 else ""

    candidates = [t for t in raw
                  if t["text_name"] == SANSKRIT_AV_NAME
                  and t["p_number"] not in SANSKRIT_ROYAL
                  and av_book(t) not in SANSKRIT_MAGICAL_EXCLUDED_BOOKS]
    candidates.sort(key=lambda t: (-sum(len(l.split()) for l in t["lines"]), t["p_number"]))
    for t in candidates[:5]:
        add("magical", t["p_number"], toks(t), t["chapter"])

    return roster, tokens
```

Add the import used above alongside the module's other slot-normalizer imports:
`from languages.sanskrit.scripts.sanskrit_normalize import normalize_sanskrit_token`
(match the file's existing import style; `json` is already imported).

(d) In `build_roster()` add the sanskrit branch after the greek branch: call
`build_sanskrit_roster()`, splice its `roster` into `roster["sanskrit"]` and
its tokens into `roster_tokens["sanskrit"]`, matching the structures the
other branches produce (read the greek branch first and mirror the exact
entry shapes — the fixture test pins the semantics, the greek branch pins
the shapes).

- [ ] **Step 4: Run tests**

Run: `pytest shared/tests/test_myth_study_roster.py -v && pytest shared/tests/ -q`
Expected: 3 new tests pass; existing tests unaffected.

- [ ] **Step 5: Full pytest + commit**

```bash
pytest -q
git add shared/scripts/myth_study.py shared/tests/test_myth_study_roster.py
git commit -m "feat(shared): sanskrit five-theme roster + combinatorial slot pairs in myth study"
```

---

### Task 3: IE-gradient extension + `vrtra_control` block

**Files:**
- Modify: `shared/scripts/myth_study.py` (IE block `:462-468`; new block after the positive control `:470-524`)

**Interfaces:**
- Consumes: Task 2's roster (`vrtra` doc in sanskrit cosmogonic), existing `doc_profile`, `percentile_in_null`, theme centroid machinery.
- Produces: results keys `ie_gradient.rho_sanskrit_hittite`, `.rho_sanskrit_greek`, `.rho_sanskrit_sumerian`, `.ie_pairs_mean`, `.non_ie_pairs_mean`; top-level `vrtra_control` = `{ladder, n_null, sub_controls: {vs_illuyanka: {rho, percentile, verdict}, vs_theogony: {...}}, bands}`.

- [ ] **Step 1: Extend the IE-gradient block**

Read `:462-468` first. Keep the existing three `rho_*` keys; add the three sanskrit pair keys by looking them up from the (now 6-entry) `slot_pair_rsa` results, and replace the single `hittite_greek_highest` bool with:

```python
    ie_keys = ("hittite_greek", "sanskrit_hittite", "sanskrit_greek")
    non_ie_keys = ("hittite_sumerian", "greek_sumerian", "sanskrit_sumerian")
    def _rho(k):
        return pair_rsa[k]["rho"] if k in pair_rsa and pair_rsa[k].get("rho") is not None else None
    ie_vals = [r for k in ie_keys if (r := _rho(k)) is not None]
    non_ie_vals = [r for k in non_ie_keys if (r := _rho(k)) is not None]
    ie_gradient = {
        **{f"rho_{k}": _rho(k) for k in ie_keys + non_ie_keys},
        "ie_pairs_mean": (sum(ie_vals) / len(ie_vals)) if ie_vals else None,
        "non_ie_pairs_mean": (sum(non_ie_vals) / len(non_ie_vals)) if non_ie_vals else None,
    }
```

(Adapt the key-naming to however `slot_pair_rsa` keys pairs — read `:441-461`; if keyed by tuple, build the string keys the same way the existing block does. Keep any existing keys the journal references.)

- [ ] **Step 2: Add the `vrtra_control` block**

Directly after the existing positive-control block (read `:470-524` first and mirror its structure), add:

```python
    # --- Vrtra positive control (pre-registered, spec 2026-07-23) ---
    # Profile of the merged vrtra doc in sanskrit native space, Spearman vs
    # illuyanka (hittite) and Theogony (greek) profiles, each against the
    # same-genre bootstrap null used by the Kumarbi control.
    vrtra_control = {"ladder": None, "n_null": N_NULL_DRAWS, "sub_controls": {},
                     "bands": {"supports": ">=90th pctile", "fails": "<=75th pctile",
                               "else": "inconclusive"}}
    # build profiles over the shared theme ladder of each pair, exactly as
    # the Kumarbi block does (leave-one-out for the doc's own theme):
    #   prof_vrtra_h = doc_profile(vrtra centroid, sanskrit theme centroids, ladder_sh)
    #   prof_illu    = doc_profile(illuyanka centroid, hittite theme centroids, ladder_sh)
    # rho = spearman(prof_vrtra_h, prof_illu); null = bootstrap draws of
    # non-cosmogonic sanskrit docs profiled the same way vs prof_illu.
    for name, other_slot, other_doc in (("vs_illuyanka", "hittite", "illuyanka"),
                                        ("vs_theogony", "greek", GREEK_THEOGONY)):
        ...  # concrete construction mirrors the Kumarbi block line-for-line
             # with (sanskrit, "vrtra") substituted for (hittite, kumarbi-doc)
    results["vrtra_control"] = vrtra_control
```

IMPLEMENTATION NOTE (this is the one place the plan delegates structure to
the existing code rather than printing it): the Kumarbi block `:470-524` is
the template — reproduce its exact sequence (ladder intersection, centroid
lookup, `doc_profile`, Spearman, `percentile_in_null` over
`N_NULL_DRAWS` same-genre draws, discreteness note) with the vrtra doc as
the probe and illuyanka/Theogony as the reference profiles, and attach the
verdict string per the bands: `percentile >= 90 → "supports the IE
combat-myth link"`, `percentile <= 75 → "fails, consistent with the
Kumarbi-control finding"`, else `"inconclusive"`. Record `ladder` (the
theme lists used) and `rho`/`percentile`/`verdict` per sub-control. Do NOT
alter the Kumarbi block itself.

- [ ] **Step 3: Compile + focused sanity**

```bash
python -m py_compile shared/scripts/myth_study.py
pytest shared/tests/test_myth_study_roster.py shared/tests/ -q
```

(The analysis blocks execute only in `run()` — full validation happens in Task 4's real run; this task's gate is compile + unchanged tests.)

- [ ] **Step 4: Full pytest + commit**

```bash
pytest -q
git add shared/scripts/myth_study.py
git commit -m "feat(shared): extended IE gradient + pre-registered vrtra control in myth study"
```

---

### Task 4: The K=5 re-run (suite-v2 spaces)

**Files:**
- Produces + COMMITS: `shared/results/myth_study.json`, `shared/results/myth_study_roster.json` (both git-tracked)

- [ ] **Step 1: Preconditions**

```bash
ls languages/sanskrit/models/fused_embeddings_1536d.npz \
   languages/sanskrit/final_output/sanskrit_aligned_gemma_vectors.npz \
   languages/{sumerian,hittite,greek}/models/fused_embeddings_1536d.npz
```

All must exist (they do — suite v2 outputs). If any is missing, STOP.

- [ ] **Step 2: Run**

```bash
mkdir -p shared/logs
nohup python shared/scripts/myth_study.py > shared/logs/myth_k5_run.log 2>&1 &
```

Poll the log. On completion check, in order:
1. `dropped_docs`: NO sanskrit roster doc may appear (zero-in-vocab pinned doc = stop-and-surface to the user with the doc ID and its token count).
2. `slot_pair_rsa` has 6 entries; the sanskrit-sumerian entry's ladder length (expected 5) and sanskrit-hittite's (expected 4). If the sumerian pair's ladder is < 5, STOP and surface (a theme dropped — find which via `themes_dropped_from_ladder`).
3. `vrtra_control.sub_controls` both present with rho/percentile/verdict.

- [ ] **Step 3: Record the pre-registered read-outs (verbatim bands)**

From the results JSON record into the SDD ledger: per-pair (ladder K, ρ, exhaustive p, n_perms) for all 6 pairs; the K=5 read-out sentence per spec (p ≤ 0.05 ∧ ρ > 0 ⇒ "first adequately-powered Plane-B positive"; else the null wording); both vrtra verdict strings exactly as emitted; Kumarbi-control v2 values vs the v1 journal values; translation-delta and fingerprint v1→v2 shifts.

- [ ] **Step 4: Commit the tracked results**

```bash
git add shared/results/myth_study.json shared/results/myth_study_roster.json
git commit -m "feat(shared): myth study K=5 re-run on suite-v2 spaces (sanskrit fourth slot)"
```

---

### Task 5: Journal + README bullet + memory

**Files:**
- Modify: `docs/EXPERIMENT_JOURNAL.md`, `README.md` (Recent-findings bullet only)

- [ ] **Step 1: Journal entry** (dated with the run date), in order: (1) what changed (fourth slot, five-theme roster with the pinned IDs and the magical-rule rationale incl. the books-14/18 exclusion, 6 pairs, v2 spaces); (2) the two pre-registered read-outs with bands restated and verdict sentences VERBATIM from Task 4's ledger record; (3) all six pair RSAs (ladder K, ρ, p); (4) Kumarbi control re-measured on v2 + explicit v1→v2 comparison paragraph (v1 values from the 2026-07-12 journal entries); (5) translation delta + fingerprints incl. sanskrit; (6) dropped-doc accounting.
- [ ] **Step 2: README** — one Recent-findings bullet (newest-first position), same style as the existing bullets.
- [ ] **Step 3: Full pytest + commit**

```bash
pytest -q
git add docs/EXPERIMENT_JOURNAL.md README.md
git commit -m "docs: myth K=5 journal entry + findings bullet"
```

Then: final whole-branch review → finishing-a-development-branch → memory update (new memory `myth-k5-state` + MEMORY.md line + follow-on-plan cross-link).

---

## Self-Review Record

- **Spec coverage:** code changes §1-7 → T1 (registration), T2 (slots/pairs/roster/merge/missing-ID raise), T3 (IE + vrtra control); roster §themes → T2 constants (pinned IDs verbatim from the verified inventory); pre-registered read-outs → Global Constraints + T4 Step 3 + T5; re-run semantics → T4 (tracked results committed; v1 preserved via journal + suite-v1 tag); error handling → T2 ValueError + T4 stop-and-surface gates; testing → T1/T2 test files, T3 compile gate + real-run validation.
- **Declared delegation (not a placeholder):** T3 Step 2 instructs mirroring the Kumarbi block line-for-line rather than printing ~50 lines of restructured analysis code — the template exists in-repo at a pinned line range, the substitutions are enumerated, and the output schema + verdict strings are fully specified. Same for T1's "mirror hittite/greek entry" and T2(d)'s "mirror the greek branch shapes" — in each case the fixture tests pin the semantics.
- **Type consistency:** `build_sanskrit_roster() -> (roster_by_theme, tokens_by_doc)` consumed by T2(d) splice and T2 tests; `enumerate_slot_pairs()` used in T2 test + `run()`; `SANSKRIT_*` constants referenced identically in T2 code/tests; `vrtra_control` schema in T3 matches T4 Step 2's checks and T5's journal fields.
