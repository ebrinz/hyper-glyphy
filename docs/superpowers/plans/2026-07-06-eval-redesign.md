# Eval Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace exact-match/full-vocab evaluation with a stratified CSLS suite computed from saved artifacts, refine the split (near-surface edges; Egyptian surface grouping + data fixes), add a document-level panel (ETCSL genre + cross-language parallels), run all five slots once, and ship a myth-study planning doc.

**Architecture:** A new `shared/scripts/eval_suite.py` holds all metric logic as pure functions over an artifact bundle each 09/09b run saves; training scripts call it at the end and alpha selection uses its val scorer. `shared/scripts/doc_eval.py` builds SIF-weighted document centroids from exported aligned vectors for the genre and parallels benchmarks. Akkadian stays the canonical alignment script; Hittite/Greek are sed-clones; Sumerian/Egyptian carry their documented deltas.

**Tech Stack:** Python 3.12, numpy, scikit-learn Ridge, NLTK WordNet, pytest.

**Spec:** `docs/superpowers/specs/2026-07-06-eval-redesign-design.md`

## Global Constraints

- CSLS neighborhood k=10; candidate vocabulary = first **50,000** rows of the target cache (GloVe file order; the Gemma cache preserves it). Constant `CAND_SIZE = 50000` lives in `eval_suite.py`; every consumer imports it.
- Candidate hubness term r(y) is computed against the artifact's full query pool (train sample + val + test projections); during alpha selection the pool is the val predictions themselves.
- Every accuracy cell reports `exact` AND `syn` (WordNet synset-sharing credit) — never one without the other. Gold glosses outside the 50k candidates are excluded from the stratum and counted in `gold_oov_candidates`, never silently dropped.
- Strata: `interpolation` = test anchor whose gold gloss appears among train+val gold glosses; `zero_shot` = it doesn't. Dictionary regime = fixed 1,000-anchor sample (numpy seed 42) of **non-OOV** train anchors, labeled in-sample.
- Split: near-surface union edges (same gloss AND surface edit distance ≤ 1) on by default for all slots; Egyptian fallback = case-folded surface grouping. Post-change leak check must be ≤1% per slot.
- Artifacts always store predictions in the FULL target space (768d/300d — Egyptian's PCA lifts before saving).
- Results JSONs keep the legacy `"accuracy"` key (now = combined-strata test CSLS/restricted exact top-1/5/10) so `10_export_production.py` keeps working unmodified; the new `"metric_suite"` block carries the real suite.
- Alignment runs: ~1–4 h per slot per target on this host. Run detached (`nohup ... &`) — foreground background-task limits killed runs before.
- Existing helper signatures (`build_training_data`, `train_ridge`, `evaluate_alignment`) unchanged. All commands from repo root. Data under `languages/*/data/` is gitignored — never committed. Full pytest green before every commit.

---

### Task 1: CSLS core in `eval_suite.py`

**Files:**
- Create: `shared/scripts/eval_suite.py`
- Test: `shared/tests/test_eval_suite.py`

**Interfaces:**
- Produces: `CAND_SIZE = 50000`, `K_CSLS = 10`, `csls_topk(Q, C, query_pool, k_csls=K_CSLS, topk=10, chunk=2048) -> np.ndarray` (n×topk candidate indices, best first). All later tasks import these.

- [ ] **Step 1: Write the failing tests**

Create `shared/tests/test_eval_suite.py`:

```python
import numpy as np

from shared.scripts.eval_suite import CAND_SIZE, K_CSLS, csls_topk


def _brute_csls(Q, C, pool, k):
    def norm(M):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    Qn, Cn, Pn = norm(Q), norm(C), norm(pool)
    S_pool = Pn @ Cn.T                       # p x m
    r_C = np.array([np.sort(S_pool[:, j])[-k:].mean() for j in range(C.shape[0])])
    S = Qn @ Cn.T
    return 2 * S - r_C[None, :]


def test_csls_matches_bruteforce():
    rng = np.random.RandomState(0)
    Q, C, pool = rng.randn(7, 16), rng.randn(40, 16), rng.randn(25, 16)
    idx = csls_topk(Q, C, pool, k_csls=5, topk=3)
    ref = np.argsort(-_brute_csls(Q, C, pool, 5), axis=1)[:, :3]
    assert np.array_equal(idx, ref)


def test_csls_demotes_hub():
    # Plain cosine prefers the hub; CSLS must flip the ranking.
    rng = np.random.RandomState(1)
    pool = rng.randn(50, 8) + np.array([5.0] + [0.0] * 7)   # pool crowds the hub
    hub = np.array([[5.0] + [0.0] * 7])
    niche = np.array([[0.0] * 7 + [5.0]])
    C = np.vstack([hub, niche])
    q = np.array([[3.0, 0, 0, 0, 0, 0, 0, 2.64]])           # cos_hub=.75 > cos_niche=.66

    def norm(M):
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    plain = (norm(q) @ norm(C).T)[0]
    assert plain[0] > plain[1]                # cosine prefers the hub...
    idx = csls_topk(q, C, pool, k_csls=10, topk=2)
    assert idx[0, 0] == 1                     # ...CSLS flips to the niche


def test_csls_chunking_consistent():
    rng = np.random.RandomState(2)
    Q, C, pool = rng.randn(30, 8), rng.randn(60, 8), rng.randn(20, 8)
    a = csls_topk(Q, C, pool, k_csls=4, topk=5, chunk=7)
    b = csls_topk(Q, C, pool, k_csls=4, topk=5, chunk=1000)
    assert np.array_equal(a, b)


def test_constants():
    assert CAND_SIZE == 50000 and K_CSLS == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest shared/tests/test_eval_suite.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shared.scripts.eval_suite'`

- [ ] **Step 3: Implement**

Create `shared/scripts/eval_suite.py`:

```python
"""
Stratified CSLS evaluation suite over saved alignment artifacts.

See: docs/superpowers/specs/2026-07-06-eval-redesign-design.md
"""
import numpy as np

CAND_SIZE = 50000
K_CSLS = 10


def _normalize(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def csls_topk(Q, C, query_pool, k_csls=K_CSLS, topk=10, chunk=2048):
    """Top-k candidate indices per query under CSLS retrieval.

    CSLS(x, y) = 2 cos(x, y) - r(y); the query-side hubness term is constant
    per query and cannot change ranking. r(y) = mean cosine of candidate y to
    its k_csls nearest neighbors in query_pool.
    """
    Qn, Cn, Pn = _normalize(np.asarray(Q, dtype=np.float32)), _normalize(
        np.asarray(C, dtype=np.float32)
    ), _normalize(np.asarray(query_pool, dtype=np.float32))

    m = Cn.shape[0]
    r_C = np.empty(m, dtype=np.float32)
    for j0 in range(0, m, chunk):
        S_pool = Pn @ Cn[j0 : j0 + chunk].T          # p x chunk
        k = min(k_csls, S_pool.shape[0])
        top = np.partition(S_pool, -k, axis=0)[-k:]
        r_C[j0 : j0 + chunk] = top.mean(axis=0)

    out = np.empty((Qn.shape[0], topk), dtype=np.int64)
    for i0 in range(0, Qn.shape[0], chunk):
        scores = 2 * (Qn[i0 : i0 + chunk] @ Cn.T) - r_C[None, :]
        part = np.argpartition(-scores, topk - 1, axis=1)[:, :topk]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        out[i0 : i0 + chunk] = np.take_along_axis(part, order, axis=1)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest shared/tests/test_eval_suite.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add shared/scripts/eval_suite.py shared/tests/test_eval_suite.py
git commit -m "feat(shared): CSLS top-k retrieval core for eval suite"
```

---

### Task 2: Strata, synonym credit, suite scoring

**Files:**
- Modify: `shared/scripts/eval_suite.py`
- Test: `shared/tests/test_eval_suite.py`

**Interfaces:**
- Consumes: `csls_topk` (Task 1).
- Produces: `stratify(test_golds, trainval_golds) -> list[str]` (`"interpolation"`/`"zero_shot"`); `synonym_set(word) -> set[str]`; `score_regime(Q, golds, cand_vectors, cand_vocab, query_pool, ks=(1,5,10)) -> dict`; `score_suite(artifacts, cand_vectors, cand_vocab) -> dict`; `val_top1_csls(Y_pred_val, val_golds, cand_vectors, cand_vocab) -> float`.

- [ ] **Step 1: Write the failing tests**

Append to `shared/tests/test_eval_suite.py`:

```python
from shared.scripts.eval_suite import (
    score_regime,
    stratify,
    synonym_set,
    val_top1_csls,
)


def test_stratify():
    strata = stratify(["king", "reed"], {"king", "water"})
    assert strata == ["interpolation", "zero_shot"]


def test_synonym_set_contains_wordnet_synonyms():
    s = synonym_set("king")
    assert "king" in s and "male monarch" in s


def test_synonym_set_unknown_word_is_identity():
    assert synonym_set("zzzznotaword") == {"zzzznotaword"}


def _identity_setup():
    # Candidates are unit basis vectors; query i equals candidate i exactly.
    C = np.eye(6, dtype=np.float32)
    vocab = ["king", "water", "reed", "house", "ruler", "sea"]
    return C, vocab


def test_score_regime_exact_and_syn():
    C, vocab = _identity_setup()
    Q = C[[0, 1]]                      # predict "king", "water" exactly
    r = score_regime(Q, ["ruler", "sea"], C, vocab, query_pool=C, ks=(1,))
    assert r["n"] == 2
    assert r["top1"]["exact"] == 0.0   # retrieved words differ from golds
    # "king" shares a synset with "ruler"? WordNet: king/ruler NOT synonyms;
    # but water/sea are not either — use a real pair: predict index of "king"
    r2 = score_regime(C[[0]], ["king"], C, vocab, query_pool=C, ks=(1,))
    assert r2["top1"]["exact"] == 100.0 and r2["top1"]["syn"] == 100.0


def test_score_regime_gold_oov_candidates():
    C, vocab = _identity_setup()
    r = score_regime(C[[0]], ["notinvocab"], C, vocab, query_pool=C, ks=(1,))
    assert r["n"] == 0 and r["gold_oov_candidates"] == 1


def test_val_top1_csls():
    C, vocab = _identity_setup()
    acc = val_top1_csls(C[[0, 2]], ["king", "reed"], C, vocab)
    assert acc == 100.0
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest shared/tests/test_eval_suite.py -v`
Expected: new tests FAIL with ImportError; Task 1 tests still pass.

- [ ] **Step 3: Implement**

Append to `shared/scripts/eval_suite.py`:

```python
_SYN_CACHE = {}


def synonym_set(word):
    """The word plus all WordNet lemma names sharing any synset with it."""
    if word in _SYN_CACHE:
        return _SYN_CACHE[word]
    out = {word}
    try:
        from nltk.corpus import wordnet as wn

        for syn in wn.synsets(word):
            out.update(l.name().lower().replace("_", " ") for l in syn.lemmas())
    except Exception:
        pass  # WordNet unavailable -> exact-only degrade, never a crash
    _SYN_CACHE[word] = out
    return out


def stratify(test_golds, trainval_golds):
    tv = set(trainval_golds)
    return ["interpolation" if g in tv else "zero_shot" for g in test_golds]


def score_regime(Q, golds, cand_vectors, cand_vocab, query_pool, ks=(1, 5, 10)):
    """Score one regime: CSLS retrieval over the restricted candidates.

    Items whose gold is outside cand_vocab are excluded and counted.
    """
    cand_set = set(cand_vocab)
    keep = [i for i, g in enumerate(golds) if g in cand_set]
    oov = len(golds) - len(keep)
    result = {"n": len(keep), "gold_oov_candidates": oov}
    if not keep:
        result.update({f"top{k}": {"exact": 0.0, "syn": 0.0} for k in ks})
        return result

    Qk = np.asarray(Q)[keep]
    gk = [golds[i] for i in keep]
    idx = csls_topk(Qk, cand_vectors, query_pool, topk=max(ks))
    for k in ks:
        exact = syn = 0
        for i, g in enumerate(gk):
            words = [cand_vocab[j] for j in idx[i, :k]]
            if g in words:
                exact += 1
            gsyn = synonym_set(g)
            if any(w in gsyn for w in words):
                syn += 1
        result[f"top{k}"] = {
            "exact": exact / len(gk) * 100,
            "syn": syn / len(gk) * 100,
        }
    return result


def val_top1_csls(Y_pred_val, val_golds, cand_vectors, cand_vocab):
    """Alpha-selection scorer: exact CSLS top-1 on the restricted candidates.

    Query pool = the val predictions themselves.
    """
    r = score_regime(
        Y_pred_val, val_golds, cand_vectors, cand_vocab, query_pool=Y_pred_val, ks=(1,)
    )
    return r["top1"]["exact"]


def score_suite(artifacts, cand_vectors, cand_vocab):
    """Full suite from an artifact bundle (see load_artifacts, Task 3)."""
    pool = np.vstack([artifacts["Q_train"], artifacts["Q_val"], artifacts["Q_test"]])
    strata = artifacts["meta"]["test_strata"]
    test_golds = [a["gold"] for a in artifacts["meta"]["test"]]
    Q_test = artifacts["Q_test"]

    def subset(name):
        sel = [i for i, s in enumerate(strata) if s == name]
        return Q_test[sel], [test_golds[i] for i in sel]

    suite = {
        "dictionary_in_sample": score_regime(
            artifacts["Q_train"],
            [a["gold"] for a in artifacts["meta"]["train_sample"]],
            cand_vectors, cand_vocab, pool,
        ),
    }
    for name in ("interpolation", "zero_shot"):
        Qs, gs = subset(name)
        suite[name] = score_regime(Qs, gs, cand_vectors, cand_vocab, pool)
    suite["test_combined"] = score_regime(Q_test, test_golds, cand_vectors, cand_vocab, pool)
    return suite
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest shared/tests/test_eval_suite.py -v`
Expected: all pass. (WordNet corpus must be present: if `synonym_set` tests fail with a LookupError, run `python -c "import nltk; nltk.download('wordnet')"` once and re-run.)

- [ ] **Step 5: Commit**

```bash
git add shared/scripts/eval_suite.py shared/tests/test_eval_suite.py
git commit -m "feat(shared): strata, synonym credit, and suite scoring"
```

---

### Task 3: Artifact bundle + CLI

**Files:**
- Modify: `shared/scripts/eval_suite.py`
- Test: `shared/tests/test_eval_suite.py`

**Interfaces:**
- Produces: `save_artifacts(prefix, *, coef, intercept, Q_train, Q_val, Q_test, train_sample, val, test, test_strata, config)` writing `<prefix>.npz` + `<prefix>.json`; `load_artifacts(prefix) -> dict` with keys `coef,intercept,Q_train,Q_val,Q_test,meta`; `load_candidates(config) -> (cand_vectors, cand_vocab)`; CLI `python -m shared.scripts.eval_suite <slot_dir> --target gemma|glove`.
- `train_sample`/`val`/`test` are lists of `{"surface": str, "gold": str}` dicts; `config` must include `target` (`"gemma"|"glove"`), `target_cache` (path), `alpha`, `seed`, and the split block.

- [ ] **Step 1: Write the failing test**

Append to `shared/tests/test_eval_suite.py`:

```python
def test_artifact_roundtrip(tmp_path):
    from shared.scripts.eval_suite import load_artifacts, save_artifacts

    prefix = str(tmp_path / "eval_artifacts_gemma")
    rng = np.random.RandomState(3)
    save_artifacts(
        prefix,
        coef=rng.randn(4, 8), intercept=rng.randn(4),
        Q_train=rng.randn(3, 4), Q_val=rng.randn(2, 4), Q_test=rng.randn(2, 4),
        train_sample=[{"surface": "a", "gold": "x"}] * 3,
        val=[{"surface": "b", "gold": "y"}] * 2,
        test=[{"surface": "c", "gold": "x"}, {"surface": "d", "gold": "z"}],
        test_strata=["interpolation", "zero_shot"],
        config={"target": "gemma", "target_cache": "unused", "alpha": 0.01, "seed": 42},
    )
    art = load_artifacts(prefix)
    assert art["Q_test"].shape == (2, 4)
    assert art["meta"]["test_strata"] == ["interpolation", "zero_shot"]
    assert art["meta"]["config"]["alpha"] == 0.01
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest shared/tests/test_eval_suite.py::test_artifact_roundtrip -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement**

Append to `shared/scripts/eval_suite.py`:

```python
import json
from pathlib import Path


def save_artifacts(prefix, *, coef, intercept, Q_train, Q_val, Q_test,
                   train_sample, val, test, test_strata, config):
    np.savez_compressed(
        prefix + ".npz",
        coef=coef, intercept=intercept,
        Q_train=np.asarray(Q_train, dtype=np.float32),
        Q_val=np.asarray(Q_val, dtype=np.float32),
        Q_test=np.asarray(Q_test, dtype=np.float32),
    )
    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(
            {"train_sample": train_sample, "val": val, "test": test,
             "test_strata": test_strata, "config": config},
            f, ensure_ascii=False, indent=2,
        )


def load_artifacts(prefix):
    npz = np.load(prefix + ".npz")
    with open(prefix + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    return {"coef": npz["coef"], "intercept": npz["intercept"],
            "Q_train": npz["Q_train"], "Q_val": npz["Q_val"],
            "Q_test": npz["Q_test"], "meta": meta}


def load_candidates(config):
    """First CAND_SIZE rows of the target cache named in config."""
    path = config["target_cache"]
    if path.endswith(".npz"):
        data = np.load(path)
        vecs = data["vectors"][:CAND_SIZE].astype(np.float32)
        vocab = [str(w) for w in data["vocab"][:CAND_SIZE]]
    else:  # GloVe text format
        vocab, rows = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                vocab.append(parts[0])
                rows.append(np.asarray(parts[1:], dtype=np.float32))
                if len(vocab) >= CAND_SIZE:
                    break
        vecs = np.array(rows)
    return vecs, vocab


def _print_suite(suite):
    for regime in ("dictionary_in_sample", "interpolation", "zero_shot", "test_combined"):
        r = suite[regime]
        cells = "  ".join(
            f"top{k}={r[f'top{k}']['exact']:.2f}/{r[f'top{k}']['syn']:.2f}%"
            for k in (1, 5, 10)
        )
        print(f"{regime:<22} n={r['n']:>6} oov={r['gold_oov_candidates']:>5}  {cells} (exact/syn)")


def main():
    import argparse

    p = argparse.ArgumentParser(description="Score an alignment artifact bundle.")
    p.add_argument("slot_dir", help="e.g. languages/akkadian")
    p.add_argument("--target", choices=("gemma", "glove"), default="gemma")
    p.add_argument("--cand-size", type=int, default=None,
                   help="Override candidate vocab size (continuity flag; e.g. 400000 for full-vocab)")
    args = p.parse_args()

    prefix = str(Path(args.slot_dir) / "results" / f"eval_artifacts_{args.target}")
    art = load_artifacts(prefix)
    global CAND_SIZE
    if args.cand_size:
        CAND_SIZE = args.cand_size
    cand_vectors, cand_vocab = load_candidates(art["meta"]["config"])
    suite = score_suite(art, cand_vectors, cand_vocab)
    _print_suite(suite)
    out = Path(args.slot_dir) / "results" / f"eval_suite_{args.target}.json"
    with open(out, "w") as f:
        json.dump(suite, f, indent=2)
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, full suite**

Run: `python -m pytest shared/tests/ -v` then `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add shared/scripts/eval_suite.py shared/tests/test_eval_suite.py
git commit -m "feat(shared): artifact bundle save/load and eval-suite CLI"
```

---

### Task 4: Split refinements + leak regression

**Files:**
- Modify: `shared/scripts/anchor_split.py`
- Test: `shared/tests/test_anchor_split.py`

**Interfaces:**
- Produces: `build_groups(anchors, surface_key, fallback="gloss", near_surface_edges=True)`; `group_split(..., fallback="gloss", near_surface_edges=True)`. `fallback` ∈ {`"gloss"`, `"surface_casefold"`} applies to anchors lacking `lemmas`. Existing callers keep working unchanged (defaults preserve behavior plus the new edges).

- [ ] **Step 1: Write the failing tests**

Append to `shared/tests/test_anchor_split.py`:

```python
def test_near_surface_edges_merge_same_gloss_ed1():
    anchors = [
        {"hittite": "kattan", "english": "under", "lemmas": ["kattan"]},
        {"hittite": "katta", "english": "under", "lemmas": ["katta"]},
    ]
    train, val, test = group_split(anchors, surface_key="hittite")
    parts = [p for p in (train, val, test) if p]
    assert len(parts) == 1  # ed<=1 + same gloss => one group


def test_near_surface_edges_do_not_merge_different_gloss():
    anchors = [
        {"akkadian": "abc", "english": "one", "lemmas": ["L1"]},
        {"akkadian": "abd", "english": "two", "lemmas": ["L2"]},
    ] * 50
    train, val, test = group_split(anchors, surface_key="akkadian")
    # Two groups exist; with 100 anchors they may land anywhere, but the two
    # surfaces must not be forced together: check group count via build_groups.
    from shared.scripts.anchor_split import build_groups

    gids = build_groups(anchors, surface_key="akkadian")
    assert len(set(gids)) == 2


def test_near_surface_edges_can_be_disabled():
    anchors = [
        {"hittite": "kattan", "english": "under", "lemmas": ["kattan"]},
        {"hittite": "katta", "english": "under", "lemmas": ["katta"]},
    ]
    from shared.scripts.anchor_split import build_groups

    assert len(set(build_groups(anchors, surface_key="hittite",
                                near_surface_edges=False))) == 2


def test_surface_casefold_fallback():
    anchors = [
        {"egyptian_raw": "Wsjr", "english": "osiris"},
        {"egyptian_raw": "wsjr", "english": "osiris-name"},
        {"egyptian_raw": "nTr", "english": "god"},
    ]
    from shared.scripts.anchor_split import build_groups

    gids = build_groups(anchors, surface_key="egyptian_raw",
                        fallback="surface_casefold", near_surface_edges=False)
    assert gids[0] == gids[1] and gids[0] != gids[2]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest shared/tests/test_anchor_split.py -v`
Expected: new tests FAIL (TypeError on unknown kwargs / group-count asserts); the existing 7 tests pass.

- [ ] **Step 3: Implement**

In `shared/scripts/anchor_split.py`, add module-level helper above `build_groups`:

```python
def _ed_le_1(a, b):
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
```

Replace `build_groups` with:

```python
def build_groups(anchors, surface_key, fallback="gloss", near_surface_edges=True):
    """Return one group id per anchor via union-find.

    Node keys per anchor: ("surface", <source surface>) always, plus
    ("lemma", l) for each entry in anchor["lemmas"] when present; otherwise
    the fallback node — ("gloss", english) or ("surface_cf", casefolded
    surface). With near_surface_edges, anchors sharing a gloss whose surfaces
    are within edit distance 1 also merge (kills residual spelling-variant
    leakage, e.g. TLHdig cf orthography).
    """
    if fallback not in ("gloss", "surface_casefold"):
        raise ValueError(f"unknown fallback: {fallback}")
    uf = _UnionFind()
    anchor_nodes = []
    for a in anchors:
        surface_node = ("surface", a[surface_key])
        lemmas = a.get("lemmas")
        if lemmas:
            others = [("lemma", l) for l in lemmas]
        elif fallback == "gloss":
            others = [("gloss", a["english"])]
        else:
            others = [("surface_cf", a[surface_key].casefold())]
        for node in others:
            uf.union(surface_node, node)
        anchor_nodes.append(surface_node)

    if near_surface_edges:
        by_gloss = {}
        for a in anchors:
            by_gloss.setdefault(a["english"], set()).add(a[surface_key])
        for gloss, surfaces in by_gloss.items():
            ss = sorted(surfaces)
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    if _ed_le_1(ss[i], ss[j]):
                        uf.union(("surface", ss[i]), ("surface", ss[j]))

    return [uf.find(n) for n in anchor_nodes]
```

Thread the parameters through `group_split` (signature becomes
`group_split(anchors, surface_key, val_size=VAL_SIZE, test_size=TEST_SIZE, seed=SEED, fallback="gloss", near_surface_edges=True)`;
the `build_groups` call inside passes both through). Nothing else changes.

Note: the per-gloss scan is O(g·s²) over surfaces sharing one gloss. Egyptian's
"the" gloss has ~2,041 surfaces → ~2M `_ed_le_1` calls with an early length
gate — seconds, acceptable. If the Greek run of the leak check (Step 5) takes
more than ~10 minutes, presort by length within each gloss bucket and only
compare surfaces whose lengths differ ≤1 — but measure first.

- [ ] **Step 4: Run tests**

Run: `python -m pytest shared/tests/test_anchor_split.py -v && python -m pytest -q`
Expected: all pass (11 anchor-split tests). If `test_partition_proportions` fails
tolerance from merged groups, widen its tolerance to 0.07 — the invariant tests
must NOT be weakened.

- [ ] **Step 5: Leak regression across all five slots**

Re-run the leak check (script exists at the session scratchpad from the prior
branch; recreate if absent) — it measures same-gloss ed≤1 leakage over the NEW
splits. For Egyptian, pass `fallback="surface_casefold"` in a variant run:

```bash
python3 /private/tmp/claude-501/-Users-crashy-Development-hyper-glyphy/0d562bd9-d698-4de6-84af-797d5f83883f/scratchpad/leak_check.py
```

Expected: ≤1.0% for every slot (Hittite drops from 10.68%). If any slot exceeds
1%, STOP and investigate before committing. Record the five numbers for Task 12.

- [ ] **Step 6: Commit**

```bash
git add shared/scripts/anchor_split.py shared/tests/test_anchor_split.py
git commit -m "feat(shared): near-surface union edges + surface-casefold fallback"
```

---

### Task 5: Canonical Akkadian integration

**Files:**
- Modify: `languages/akkadian/scripts/09_align_and_evaluate.py`
- Modify: `languages/akkadian/scripts/09b_align_gemma.py`
- Modify: `languages/akkadian/scripts/align_09.py`

This is the CANONICAL template for Task 6's clones — apply exactly; no stylistic liberties. No pipeline runs in this task (runs are Task 8).

- [ ] **Step 1: Rewire alpha selection in `09_align_and_evaluate.py`**

(a) Extend the shared imports block:

```python
from shared.scripts.anchor_split import group_split, SEED, TEST_SIZE, VAL_SIZE
from shared.scripts.eval_suite import (
    CAND_SIZE,
    save_artifacts,
    score_suite,
    stratify,
    val_top1_csls,
)
```

(b) In `select_alpha`, replace the body's scoring lines

```python
        acc = evaluate_alignment(Y_pred, val_english, eng_vocab_list, eng_vectors)
        sweep.append({"alpha": alpha, "accuracy": acc})
        print(f"  alpha={alpha:<10g} val top1={acc['top1']:.2f}%")
        if acc["top1"] > best_top1:
            best_alpha, best_top1 = alpha, acc["top1"]
```

with

```python
        top1 = val_top1_csls(
            Y_pred, val_english, eng_vectors[:CAND_SIZE], eng_vocab_list[:CAND_SIZE]
        )
        sweep.append({"alpha": alpha, "val_top1_csls_exact": top1})
        print(f"  alpha={alpha:<10g} val top1 (CSLS/50k)={top1:.2f}%")
        if top1 > best_top1:
            best_alpha, best_top1 = alpha, top1
```

(`select_alpha`'s signature is unchanged; `evaluate_alignment` stays defined for
the legacy path but is no longer called during selection.)

(c) In `main()` after the retrain (`model = train_ridge(X_fit, Y_fit, ...)`),
replace everything from `Y_pred = model.predict(X_test)` through the
`full_results = {...}` construction with:

```python
    # Artifact bundle: predictions in full target space + strata metadata.
    rng = np.random.RandomState(SEED)
    non_oov_train = [
        (i, a) for i, a in enumerate(train_valid) if not a.get("subword_inferred")
    ]
    sample_idx = rng.choice(
        len(non_oov_train), size=min(1000, len(non_oov_train)), replace=False
    )
    train_sample = [non_oov_train[i] for i in sample_idx]
    Q_train = model.predict(X_train[[i for i, _ in train_sample]])
    Q_val = model.predict(X_val)
    Q_test = model.predict(X_test)

    trainval_golds = {a["english"] for a in train_valid} | {
        a["english"] for a in val_valid
    }
    test_strata = stratify([a["english"] for a in test_valid], trainval_golds)

    config = {
        "target": "glove",
        "target_cache": str(glove_path),
        "alpha": best_alpha,
        "alpha_sweep_val": sweep,
        "seed": SEED,
        "candidate_vocab_size": CAND_SIZE,
        "split": {
            "method": "lemma-group",
            "seed": SEED,
            "val_size": VAL_SIZE,
            "test_size": TEST_SIZE,
            "near_surface_edges": True,
            "raw": {"train": len(train_anchors), "val": len(val_anchors),
                    "test": len(test_anchors)},
            "valid": {"train": len(train_valid), "val": len(val_valid),
                      "test": len(test_valid)},
            "oov_train_only": n_oov_train,
        },
    }
    prefix = str(RESULTS_DIR / "eval_artifacts_glove")
    save_artifacts(
        prefix,
        coef=model.coef_, intercept=model.intercept_,
        Q_train=Q_train, Q_val=Q_val, Q_test=Q_test,
        train_sample=[{"surface": a[SURFACE_KEY], "gold": a["english"]}
                      for _, a in train_sample],
        val=[{"surface": a[SURFACE_KEY], "gold": a["english"]} for a in val_valid],
        test=[{"surface": a[SURFACE_KEY], "gold": a["english"]} for a in test_valid],
        test_strata=test_strata,
        config=config,
    )
    print(f"Artifacts saved to {prefix}.npz/.json")

    cand_vectors = glove_vectors[:CAND_SIZE]
    cand_vocab = glove_vocab[:CAND_SIZE]
    from shared.scripts.eval_suite import load_artifacts

    suite = score_suite(load_artifacts(prefix), cand_vectors, cand_vocab)

    print("\n=== METRIC SUITE (CSLS, 50k candidates, exact/syn) ===")
    for regime in ("dictionary_in_sample", "interpolation", "zero_shot", "test_combined"):
        r = suite[regime]
        print(f"{regime:<22} n={r['n']:>6} "
              + "  ".join(f"top{k}={r[f'top{k}']['exact']:.2f}/{r[f'top{k}']['syn']:.2f}"
                          for k in (1, 5, 10)))

    combined = suite["test_combined"]
    full_results = {
        # Legacy key: combined-strata test CSLS/restricted EXACT top-k, so
        # 10_export_production.py keeps reading the same shape.
        "accuracy": {f"top{k}": combined[f"top{k}"]["exact"] for k in (1, 5, 10)},
        "metric_suite": suite,
        "config": config | {
            "train_size": len(X_fit),
            "test_size": len(X_test),
            "valid_anchors": n_valid,
            "total_anchors": len(anchors),
            "sumerian_vocab": len(sum_vocab),
            "fused_dim": int(sum_vectors.shape[1]),
            "glove_dim": int(glove_vectors.shape[1]),
        },
    }
```

The `X_train` indexing for `Q_train` requires the ROW indices of the sampled
anchors within `X_train` — `train_valid` rows align 1:1 with `X_train` rows
(both come from the same `build_training_data` call), so `non_oov_train`'s
enumerate index IS the row index. Keep the ridge-weights `np.savez_compressed`
block and results-JSON write as they are (results path unchanged).

- [ ] **Step 2: Same rewiring in `09b_align_gemma.py`**

Identical changes with: `"target": "gemma"`, `"target_cache": str(english_gemma_path)`,
artifact prefix `eval_artifacts_gemma`, candidates `eng_vectors[:CAND_SIZE]` /
`eng_vocab_list[:CAND_SIZE]`, and the legacy `"accuracy"` key computed the same
way (this also feeds the `baseline_glove` comparison print — keep the existing
print block, reading `results['top1']` etc. from the new `"accuracy"` dict —
assign `results = full_results["accuracy"]` before the print block to keep it
working verbatim).

Import line extends the existing `align_09` import with nothing new — the
eval_suite imports go directly at the top as in Step 1(a).

- [ ] **Step 3: Shim + verification**

```bash
echo 'val_top1_csls = _mod.val_top1_csls if hasattr(_mod, "val_top1_csls") else None' >> languages/akkadian/scripts/align_09.py
python -m py_compile languages/akkadian/scripts/09_align_and_evaluate.py languages/akkadian/scripts/09b_align_gemma.py
python -m pytest languages/akkadian/tests/ shared/tests/ -q
```
Expected: compile clean, all tests pass. (The shim line is defensive only;
09b imports `val_top1_csls` from eval_suite directly.)

- [ ] **Step 4: Smoke-run on a truncated anchor set**

Verify end-to-end plumbing WITHOUT the multi-hour run: temporarily run with a
reduced grid by env var — add at the top of `main()` in BOTH scripts:

```python
    import os
    alphas = [0.01] if os.environ.get("EVAL_SMOKE") else ALPHAS
```

and use `alphas` in the `select_alpha` call. Then:

```bash
EVAL_SMOKE=1 python languages/akkadian/scripts/09_align_and_evaluate.py
```

Expected: completes in ~10–20 min; prints the metric-suite table; writes
`results/eval_artifacts_glove.{npz,json}` and a results JSON whose `"accuracy"`
and `"metric_suite"` blocks are populated. Then confirm the CLI reproduces it:

```bash
python -m shared.scripts.eval_suite languages/akkadian --target glove
```

Expected: same numbers as the run printed.

- [ ] **Step 5: Commit**

```bash
git add languages/akkadian/scripts/
git commit -m "feat(akkadian): artifact bundle + CSLS suite integration (canonical)"
```

---

### Task 6: Clone to Hittite/Greek + Sumerian variant

**Files:**
- Modify: `languages/hittite/scripts/09_align_and_evaluate.py`, `09b_align_gemma.py`, `align_09.py`
- Modify: `languages/greek/scripts/09_align_and_evaluate.py`, `09b_align_gemma.py`, `align_09.py`
- Modify: `languages/sumerian/scripts/09_align_and_evaluate.py`, `09b_align_gemma.py`, `align_09.py`

- [ ] **Step 1: Sed-clone Hittite and Greek from canonical**

```bash
for lang in hittite greek; do
  Lang=$(python3 -c "print('$lang'.capitalize())")
  for f in 09_align_and_evaluate.py 09b_align_gemma.py; do
    sed -e "s/akkadian/$lang/g" -e "s/Akkadian/$Lang/g" \
      languages/akkadian/scripts/$f > languages/$lang/scripts/$f
    diff <(sed -e "s/akkadian/$lang/g" -e "s/Akkadian/$Lang/g" \
      languages/akkadian/scripts/$f) languages/$lang/scripts/$f
  done
done
```

Expected: all diffs empty. Append the same defensive shim line from Task 5
Step 3 to both languages' `align_09.py`.

- [ ] **Step 2: Sumerian variant**

Apply Task 5 Steps 1–2 to Sumerian's scripts with its known deltas: no
FastText/OOV anywhere, so `non_oov_train = list(enumerate(train_valid))` (no
`subword_inferred` filter) and the split config has no `oov_train_only` key;
`SURFACE_KEY = "sumerian"`; config keeps `"sumerian_vocab"` naming. Append the
shim line to its `align_09.py`.

- [ ] **Step 3: Verify**

```bash
python -m py_compile languages/{hittite,greek,sumerian}/scripts/09_align_and_evaluate.py \
  languages/{hittite,greek,sumerian}/scripts/09b_align_gemma.py
python -m pytest -q
```
Expected: compile clean; full suite passes.

- [ ] **Step 4: Commit**

```bash
git add languages/hittite/scripts/ languages/greek/scripts/ languages/sumerian/scripts/
git commit -m "feat(hittite,greek,sumerian): CSLS suite integration (clones + variant)"
```

---

### Task 7: Egyptian integration (surface split + data fixes)

**Files:**
- Modify: `languages/egyptian/scripts/09_align_and_evaluate.py`, `09b_align_gemma.py`, `align_09.py`
- Test: `languages/egyptian/tests/test_09_alignment.py`

- [ ] **Step 1: Write the failing tests**

Append to `languages/egyptian/tests/test_09_alignment.py`:

```python
def test_build_training_data_casefold_fallback():
    from languages.egyptian.scripts.align_09 import build_training_data

    anchors = [{"egyptian": "wsjr", "egyptian_raw": "wsjr", "english": "osiris"}]
    eg_vocab = {"Wsjr": 0}                     # corpus preserved capitalization
    eg_vectors = np.random.randn(1, 1536).astype(np.float32)
    eng_vocab = {"osiris": 0}
    eng_vectors = np.random.randn(1, 300).astype(np.float32)
    X, Y, valid = build_training_data(anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors)
    assert len(valid) == 1                     # recovered via casefold


def test_stopword_gloss_filter():
    from languages.egyptian.scripts.align_09 import filter_stopword_glosses

    anchors = [
        {"egyptian_raw": "a", "english": "the"},
        {"egyptian_raw": "b", "english": "des"},
        {"egyptian_raw": "c", "english": "god"},
    ]
    kept, dropped = filter_stopword_glosses(anchors)
    assert [a["english"] for a in kept] == ["god"] and dropped == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest languages/egyptian/tests/test_09_alignment.py -v`
Expected: 2 new FAIL (casefold miss → `len(valid) == 0`; ImportError for the filter).

- [ ] **Step 3: Implement in `09_align_and_evaluate.py`**

(a) In `build_training_data`, replace the lookup block

```python
        if e_word in eg_vocab and eng_word in eng_vocab:
            X_list.append(eg_vectors[eg_vocab[e_word]])
```

with a casefold fallback (build the folded index once, before the loop):

```python
    eg_vocab_cf = {}
    for w, i in eg_vocab.items():
        eg_vocab_cf.setdefault(w.casefold(), i)
```

and inside the loop:

```python
        idx = eg_vocab.get(e_word)
        if idx is None:
            idx = eg_vocab_cf.get(e_word.casefold())
        if idx is not None and eng_word in eng_vocab:
            X_list.append(eg_vectors[idx])
            Y_list.append(eng_vectors[eng_vocab[eng_word]])
            valid.append(anchor)
```

(b) Add above `build_training_data`:

```python
STOPWORD_GLOSSES = {
    "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with",
    "not", "no", "be", "is", "are", "was", "were", "as", "or", "and",
    "but", "if", "so", "do", "did", "have", "has", "had", "from", "into",
    "out", "up", "down", "over", "under", "between", "during", "before",
    "after", "above", "below", "any", "some", "all", "each", "every",
    "one", "two", "three", "four", "five", "des", "de",
}


def filter_stopword_glosses(anchors):
    """Drop anchors whose gloss is a pure function word. Returns (kept, n_dropped)."""
    kept = [a for a in anchors if a["english"] not in STOPWORD_GLOSSES]
    return kept, len(anchors) - len(kept)
```

(c) In `main()` right after loading anchors:

```python
    anchors, n_stopword_dropped = filter_stopword_glosses(anchors)
    print(f"Stopword-gloss filter: dropped {n_stopword_dropped}, kept {len(anchors)}")
```

(d) Switch the split call to surface grouping:

```python
    train_anchors, val_anchors, test_anchors = group_split(
        anchors, surface_key=SURFACE_KEY, fallback="surface_casefold"
    )
```

(e) Apply the Task 5 artifact/suite rewiring (Steps 1(b), 1(c)) with Egyptian's
names (`eg_vocab`, `eg_vectors`, `"egyptian_vocab"` legacy key, no
`oov_train_only`, `non_oov_train = list(enumerate(train_valid))`), and config
additions `"split.method": "surface-casefold-group"` and
`"stopword_glosses_dropped": n_stopword_dropped`.

- [ ] **Step 4: Implement in `09b_align_gemma.py`**

Same (c)/(d)/(e) plus the PCA delta: predictions are lifted BEFORE saving —

```python
    Q_train = pca.inverse_transform(model.predict(X_train[[i for i, _ in train_sample]]))
    Q_val = pca.inverse_transform(model.predict(X_val))
    Q_test = pca.inverse_transform(model.predict(X_test))
```

and `select_alpha` keeps `predict_transform=pca.inverse_transform` (the scorer
inside then sees 768d predictions — no other change). `filter_stopword_glosses`
is imported from `align_09`.

- [ ] **Step 5: Verify + commit**

```bash
python -m pytest languages/egyptian/tests/ -q && python -m pytest -q
python -m py_compile languages/egyptian/scripts/09_align_and_evaluate.py languages/egyptian/scripts/09b_align_gemma.py
git add languages/egyptian/
git commit -m "feat(egyptian): surface-casefold split, casefold lookup, stopword-gloss filter, suite"
```

---

### Task 8: Five-slot runs

**Files:** none modified — execution + recorded metrics only. Remove the `EVAL_SMOKE` env lines from all ten scripts FIRST (they were Task 5/6/7 scaffolding), commit that removal (`chore: drop smoke-run scaffolding`), then run.

- [ ] **Step 1: Run each slot detached, sequentially**

For each slot in akkadian, sumerian, hittite, egyptian, greek:

```bash
nohup sh -c "python languages/<slot>/scripts/09_align_and_evaluate.py && \
             python languages/<slot>/scripts/09b_align_gemma.py --mode whitened && \
             python languages/<slot>/scripts/10_export_production.py" \
  > /private/tmp/claude-501/-Users-crashy-Development-hyper-glyphy/0d562bd9-d698-4de6-84af-797d5f83883f/scratchpad/<slot>_run.log 2>&1 &
```

Egyptian's 09b takes no `--mode` flag — drop it there. Greek: this is its FIRST
run; if `languages/greek/scripts/10_export_production.py` does not exist, create
it before Greek's run as a sed-clone of Akkadian's
(`sed -e 's/akkadian/greek/g' -e 's/Akkadian/Greek/g'`), verify with
`python -m py_compile`, and commit it (`feat(greek): production export (clone)`)
— Task 10's parallels benchmark requires Greek's aligned vectors. Wait for each slot's log to end (`tail`) before starting the next;
budget 2–8 h per slot. If a run dies, capture the traceback in the report and
STOP that slot — do not silently retry.

- [ ] **Step 2: Record the suite tables**

For every completed slot × target, capture from the log (or re-print via
`python -m shared.scripts.eval_suite languages/<slot> --target <t>`): the four
regime rows (n, oov, top-1/5/10 exact/syn), selected alpha, split sizes,
stopword-drop count (Egyptian). These numbers feed Tasks 9, 10, 12.

- [ ] **Step 3: Verify artifacts + full pytest**

```bash
ls languages/*/results/eval_artifacts_*.npz | wc -l   # expected 10 (9 if Greek glove-only issue — investigate if fewer)
python -m pytest -q
```

---

### Task 9: Document-level panel — ETCSL genre benchmark

**Files:**
- Create: `shared/scripts/doc_eval.py`
- Test: `shared/tests/test_doc_eval.py`

**Interfaces:**
- Produces: `parse_etcsl_compositions(records) -> dict[str, dict]` (comp id → `{"genre": str, "tokens": list[str]}`); `sif_weights(token_counts, a=1e-3) -> dict[str, float]`; `doc_centroid(tokens, vocab, vectors, weights) -> np.ndarray | None`; `loo_nearest_centroid(doc_vecs, labels) -> float`; CLI `python -m shared.scripts.doc_eval genre`.

- [ ] **Step 1: Write the failing tests**

Create `shared/tests/test_doc_eval.py`:

```python
import numpy as np

from shared.scripts.doc_eval import (
    GENRE_CLASSES,
    doc_centroid,
    loo_nearest_centroid,
    parse_etcsl_compositions,
    sif_weights,
)


def test_parse_etcsl_compositions():
    records = [
        {"line_id": "c141.A.1", "transliteration": "lugal kur-ra", "translation": ""},
        {"line_id": "c141.A.2", "transliteration": "e2 gal", "translation": ""},
        {"line_id": "c2554.A.1", "transliteration": "lugal an-na", "translation": ""},
        {"line_id": "c341.A.1", "transliteration": "x", "translation": ""},  # class 3 -> dropped
    ]
    comps = parse_etcsl_compositions(records)
    assert set(comps) == {"c141", "c2554"}
    assert comps["c141"]["genre"] == GENRE_CLASSES["1"]
    assert comps["c2554"]["genre"] == GENRE_CLASSES["2"]
    assert "lugal" in comps["c141"]["tokens"]


def test_sif_weights_downweight_frequent():
    w = sif_weights({"the": 1000, "rare": 1})
    assert w["rare"] > w["the"]


def test_doc_centroid_skips_oov():
    vocab = {"lugal": 0}
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    c = doc_centroid(["lugal", "notavocabword"], vocab, vectors, {"lugal": 1.0})
    assert np.allclose(c, [1.0, 0.0])
    assert doc_centroid(["notavocabword"], vocab, vectors, {}) is None


def test_loo_nearest_centroid_separable():
    rng = np.random.RandomState(0)
    a = rng.randn(10, 4) + np.array([10, 0, 0, 0])
    b = rng.randn(10, 4) + np.array([0, 10, 0, 0])
    acc = loo_nearest_centroid(np.vstack([a, b]), ["A"] * 10 + ["B"] * 10)
    assert acc == 100.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest shared/tests/test_doc_eval.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

Create `shared/scripts/doc_eval.py`:

```python
"""
Document-level evaluation: SIF centroids over aligned word vectors.

Benchmarks: (a) ETCSL genre classification (leave-one-out nearest-centroid),
(b) cross-language parallel retrieval (see parallels subcommand, added later).

See: docs/superpowers/specs/2026-07-06-eval-redesign-design.md
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from languages.sumerian.scripts.sumerian_normalize import normalize_sumerian_token  # noqa: E402

GENRE_CLASSES = {"1": "narrative", "2": "royal", "4": "hymns",
                 "5": "literature", "6": "proverbs"}
MIN_COMPOSITIONS = 10
SIF_A = 1e-3
_TOKEN_RE = re.compile(r"[a-zA-ZšŠḫḪṭṬṣṢĝĜ]+\d*")

ETCSL_PATH = _ROOT / "languages" / "sumerian" / "data" / "raw" / "etcsl_texts.json"


def _tokenize_sumerian(text):
    return [t for t in (normalize_sumerian_token(m) for m in _TOKEN_RE.findall(text.lower())) if t]


def parse_etcsl_compositions(records):
    """Group ETCSL lines into compositions; genre from catalogue class digit."""
    comps = {}
    for r in records:
        line_id = r.get("line_id", "")
        comp = line_id.split(".")[0]          # 'c2554.A.1' -> 'c2554'
        if len(comp) < 2 or not comp.startswith("c"):
            continue
        genre = GENRE_CLASSES.get(comp[1])
        if genre is None:
            continue
        entry = comps.setdefault(comp, {"genre": genre, "tokens": []})
        entry["tokens"].extend(_tokenize_sumerian(r.get("transliteration", "")))
    return comps


def sif_weights(token_counts, a=SIF_A):
    total = sum(token_counts.values())
    return {w: a / (a + c / total) for w, c in token_counts.items()}


def doc_centroid(tokens, vocab, vectors, weights):
    rows, ws = [], []
    for t in tokens:
        i = vocab.get(t)
        if i is not None:
            rows.append(vectors[i])
            ws.append(weights.get(t, 1.0))
    if not rows:
        return None
    M = np.asarray(rows, dtype=np.float32)
    w = np.asarray(ws, dtype=np.float32)[:, None]
    return (M * w).sum(axis=0) / w.sum()


def loo_nearest_centroid(doc_vecs, labels):
    """Leave-one-out nearest-centroid classification accuracy (cosine)."""
    X = np.asarray(doc_vecs, dtype=np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    labels = np.asarray(labels)
    sums, counts = {}, {}
    for lab in set(labels):
        sel = labels == lab
        sums[lab] = X[sel].sum(axis=0)
        counts[lab] = int(sel.sum())
    correct = 0
    for i in range(len(X)):
        best_lab, best_sim = None, -2.0
        for lab in sums:
            s, c = sums[lab], counts[lab]
            if labels[i] == lab:
                if c == 1:
                    continue          # singleton class: no LOO centroid
                cent = (s - X[i]) / (c - 1)
            else:
                cent = s / c
            cent = cent / (np.linalg.norm(cent) + 1e-12)
            sim = float(Xn[i] @ cent)
            if sim > best_sim:
                best_lab, best_sim = lab, sim
        correct += best_lab == labels[i]
    return correct / len(X) * 100


def _load_space(npz_path):
    data = np.load(str(npz_path), allow_pickle=True)
    vectors = data["vectors"].astype(np.float32)
    vocab = {str(w): i for i, w in enumerate(data["vocab"])}
    return vocab, vectors


def run_genre():
    with open(ETCSL_PATH) as f:
        comps = parse_etcsl_compositions(json.load(f))
    by_genre = Counter(c["genre"] for c in comps.values())
    comps = {k: v for k, v in comps.items() if by_genre[v["genre"]] >= MIN_COMPOSITIONS}
    counts = Counter(t for c in comps.values() for t in c["tokens"])
    weights = sif_weights(counts)

    sfo = _ROOT / "languages" / "sumerian" / "final_output"
    spaces = {
        "gemma_aligned": sfo / "sumerian_aligned_gemma_vectors.npz",
        "glove_aligned": sfo / "sumerian_aligned_vectors.npz",
        "fused_unaligned": _ROOT / "languages" / "sumerian" / "models" / "fused_embeddings_1536d.npz",
    }
    out = {"n_compositions": len(comps),
           "genre_counts": dict(Counter(c["genre"] for c in comps.values()))}
    for name, path in spaces.items():
        if not path.exists():
            out[name] = "MISSING: " + str(path)
            continue
        vocab, vectors = _load_space(path)
        vecs, labels = [], []
        for c in comps.values():
            v = doc_centroid(c["tokens"], vocab, vectors, weights)
            if v is not None:
                vecs.append(v)
                labels.append(c["genre"])
        out[name] = {"n_docs": len(vecs),
                     "loo_accuracy": round(loo_nearest_centroid(vecs, labels), 2)}
        print(f"{name:<16} n={len(vecs)} LOO acc={out[name]['loo_accuracy']}%")
    majority = max(out["genre_counts"].values()) / out["n_compositions"] * 100
    out["majority_baseline"] = round(majority, 2)
    print(f"majority-class baseline: {majority:.2f}%")
    res = _ROOT / "languages" / "sumerian" / "results" / "doc_eval_genre.json"
    with open(res, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to: {res}")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("benchmark", choices=("genre", "parallels"))
    args = p.parse_args()
    if args.benchmark == "genre":
        run_genre()
    else:
        run_parallels()


if __name__ == "__main__":
    main()
```

(`run_parallels` is added in Task 10; define a stub `def run_parallels(): raise SystemExit("parallels: implemented in Task 10")` for now so the CLI parses.)

- [ ] **Step 4: Run tests, then the benchmark**

```bash
python -m pytest shared/tests/test_doc_eval.py -v      # expected: 4 passed
python -m shared.scripts.doc_eval genre
```

Expected: prints n_compositions (~300–400 after class filtering), LOO accuracy
per space, and the majority baseline. Aligned spaces must beat the majority
baseline for document-level claims to stand — report whatever comes out;
if `fused_unaligned` (1536d) scores far above the aligned spaces, that bounds
projection quality and goes in the journal as-is.

- [ ] **Step 5: Commit**

```bash
git add shared/scripts/doc_eval.py shared/tests/test_doc_eval.py
git commit -m "feat(shared): ETCSL genre benchmark (SIF centroids, LOO nearest-centroid)"
```

---

### Task 10: Document-level panel — cross-language parallels

**Files:**
- Modify: `shared/scripts/doc_eval.py`
- Test: `shared/tests/test_doc_eval.py`

- [ ] **Step 1: Discovery — verify candidate texts exist in our corpora**

Known data shapes: `languages/{hittite,greek}/data/raw/*_texts.json` are lists of
`{"p_number": str, "lines": [str]}`; Sumerian documents come from ETCSL comp ids;
**Akkadian lemma dumps carry no text ids** (verified: `ob_literary_lemmas.json`
fields are cf/form/gw/pos/norm/lang only) — Akkadian-side pairs are expected to
drop unless another per-text source exists (check `languages/akkadian/data/raw/`
for files with text/document ids before concluding).

Candidate manifest (search terms → drop pair if either side unresolved):

| Pair | Side A | Side B |
|---|---|---|
| Inanna ↔ Ištar Descent | ETCSL `c141` | Akkadian (expected DROP — no doc ids) |
| Flood | ETCSL `c174` (Ziusudra) | Akkadian (expected DROP) |
| Kumarbi ↔ Theogony | Hittite p_numbers matching `KUB 33.120` (CTH 344) | Greek p_number containing `Theogon` |
| Illuyanka ↔ Typhon | Hittite p_numbers matching `KBo 3.7` or `KUB 17.5` (CTH 321) | Greek p_number containing `Theogon` (Typhonomachy is inside Theogony) |

Write the discovery results (found/dropped per side, with the matched
p_numbers) into `shared/results/doc_eval_parallels_manifest.json` — create the
`shared/results/` directory. Every DROP is logged with its reason; if fewer
than 2 pairs survive, STOP and report — the benchmark needs at least 2 to rank.

- [ ] **Step 2: Write the failing test**

Append to `shared/tests/test_doc_eval.py`:

```python
def test_mrr():
    from shared.scripts.doc_eval import mean_reciprocal_rank

    # ranks are 1-based positions of the true parallel
    assert mean_reciprocal_rank([1, 2, 4]) == round((1 + 0.5 + 0.25) / 3, 4)
```

Run: `python -m pytest shared/tests/test_doc_eval.py::test_mrr -v` — expected ImportError.

- [ ] **Step 3: Implement `run_parallels`**

Replace the Task 9 stub in `doc_eval.py`:

```python
def mean_reciprocal_rank(ranks):
    return round(sum(1.0 / r for r in ranks) / len(ranks), 4)


def _slot_documents():
    """Per-slot {doc_id: tokens} for the slots with per-text corpora."""
    docs = {}
    with open(ETCSL_PATH) as f:
        comps = parse_etcsl_compositions(json.load(f))
    docs["sumerian"] = {k: v["tokens"] for k, v in comps.items()}

    from languages.hittite.scripts.hittite_normalize import normalize_hittite_token
    from languages.greek.scripts.greek_normalize import normalize_greek_token

    for slot, normalizer in (("hittite", normalize_hittite_token),
                             ("greek", normalize_greek_token)):
        path = _ROOT / "languages" / slot / "data" / "raw" / f"{slot}_texts.json"
        out = {}
        for t in json.load(open(path)):
            toks = [normalizer(w) for line in t["lines"] for w in line.split()]
            out[t["p_number"]] = [t for t in toks if t]
        docs[slot] = out
    return docs


PARALLEL_PAIRS = [
    # (slot_a, doc_matcher_a, slot_b, doc_matcher_b, label)
    ("hittite", lambda p: "KUB 33.120" in p, "greek", lambda p: "Theogon" in p, "kumarbi-theogony"),
    ("hittite", lambda p: ("KBo 3.7" in p) or ("KUB 17.5" in p), "greek",
     lambda p: "Theogon" in p, "illuyanka-typhon"),
]


def run_parallels():
    docs = _slot_documents()
    aligned = {}
    for slot in docs:
        path = _ROOT / "languages" / slot / "final_output" / f"{slot}_aligned_gemma_vectors.npz"
        if path.exists():
            aligned[slot] = _load_space(path)

    # SIF weights per slot from its own document tokens
    weights = {s: sif_weights(Counter(t for d in docs[s].values() for t in d))
               for s in docs}

    centroids = {}
    for slot in aligned:
        vocab, vectors = aligned[slot]
        centroids[slot] = {
            did: v for did, toks in docs[slot].items()
            if (v := doc_centroid(toks, vocab, vectors, weights[slot])) is not None
        }

    results, ranks = [], []
    for slot_a, match_a, slot_b, match_b, label in PARALLEL_PAIRS:
        if slot_a not in centroids or slot_b not in centroids:
            results.append({"pair": label, "status": "DROPPED: missing aligned space"})
            continue
        a_ids = [d for d in centroids[slot_a] if match_a(d)]
        b_ids = [d for d in centroids[slot_b] if match_b(d)]
        if not a_ids or not b_ids:
            results.append({"pair": label, "status":
                            f"DROPPED: unmatched (a={len(a_ids)}, b={len(b_ids)})"})
            continue
        qv = np.mean([centroids[slot_a][d] for d in a_ids], axis=0)
        tv = np.mean([centroids[slot_b][d] for d in b_ids], axis=0)
        # Rank the true target among ALL other-slot documents.
        pool_ids = list(centroids[slot_b].keys())
        pool = np.array([centroids[slot_b][d] for d in pool_ids], dtype=np.float32)
        pool_n = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-12)
        qn = qv / (np.linalg.norm(qv) + 1e-12)
        sims = pool_n @ qn
        target_sim = float((tv / (np.linalg.norm(tv) + 1e-12)) @ qn)
        rank = int((sims > target_sim).sum()) + 1
        ranks.append(rank)
        results.append({"pair": label, "rank": rank, "pool_size": len(pool_ids),
                        "matched_a": a_ids[:5], "matched_b": b_ids[:5]})
        print(f"{label:<20} rank {rank}/{len(pool_ids)}")

    out = {"pairs": results, "mrr": mean_reciprocal_rank(ranks) if ranks else None}
    res = _ROOT / "shared" / "results" / "doc_eval_parallels.json"
    res.parent.mkdir(exist_ok=True)
    with open(res, "w") as f:
        json.dump(out, f, indent=2)
    print(f"MRR: {out['mrr']}  Saved to: {res}")
```

Extend `PARALLEL_PAIRS` with any additional pairs the Step 1 discovery
confirmed (e.g., Akkadian pairs if a per-text source was found), following the
same tuple shape.

- [ ] **Step 4: Run tests + benchmark**

```bash
python -m pytest shared/tests/test_doc_eval.py -v
python -m shared.scripts.doc_eval parallels
```

Expected: per-pair ranks and MRR printed (needs Task 8's Greek/Hittite exports
on disk); any dropped pair listed with a reason. Record numbers for Task 12.

- [ ] **Step 5: Commit**

```bash
git add shared/scripts/doc_eval.py shared/tests/test_doc_eval.py
git commit -m "feat(shared): cross-language parallel-retrieval benchmark"
```

---

### Task 11: Myth-study planning document

**Files:**
- Create: `docs/myth_study_plan.md`

- [ ] **Step 1: Write the document** with exactly these sections (prose, ~1,500–2,500 words, drawing numbers from Tasks 8–10 outputs):

1. **Purpose** — cross-civilization comparison of creation myths and magical/incantation texts using the aligned semantic spaces; explicitly an interpretive study built on the validated document-level representation, not a translation claim.
2. **Research questions** — (a) do cosmogonic texts across Sumerian/Akkadian/Hittite/Greek show measurably higher mutual affinity than genre-matched controls? (b) do magical texts share a distinctive vocabulary geometry (binding, naming, apotropaic concepts)? (c) does the known Kumarbi→Theogony transmission show up as the strongest cross-family link (positive control)?
3. **Candidate text sets** — per slot, the concrete documents: Sumerian ETCSL narratives (c.1.x: Inanna's Descent c141, Ziusudra c174, Enki narratives), Hittite (Kumarbi KUB 33.120, Illuyanka KBo 3.7/KUB 17.5, rituals), Greek (Theogony, magical papyri NOT in Diorisis — note absence honestly), Egyptian (Coffin Texts already in corpus — verify), Akkadian (blocked on per-text corpus — name it as a prerequisite).
4. **Method** — SIF centroids in shared whitened-Gemma space; pairwise similarity matrices with genre-matched null distributions (bootstrap over same-genre non-parallel pairs); thematic concept fingerprints (ranked similarity of each doc centroid to a curated English concept list: water, chaos, serpent, name, fate, bind, create, mountain); Gemma–GloVe agreement as per-claim confidence.
5. **Controls** — Kumarbi→Theogony as positive; IE relatedness gradient (Hittite–Greek elevated vs isolates); random same-genre pairs as null; future Sanskrit slot strengthens the IE triangle.
6. **Go/no-go dependencies** — from this spec's outcomes: ETCSL genre LOO accuracy must beat the majority baseline by ≥15pp in at least one aligned space; parallel-retrieval MRR ≥ 0.1 with the positive-control pair ranking in the top quartile of its pool. Fill in the measured values from Tasks 9–10 and state pass/fail per criterion.
7. **Out of scope / future** — Sanskrit slot (Atharvaveda as magical comparandum), Mayan as document-level null control only, Procrustes remap.

- [ ] **Step 2: Commit**

```bash
git add docs/myth_study_plan.md
git commit -m "docs: myth-study planning document (questions, texts, method, go/no-go)"
```

---

### Task 12: Reporting — journal, metadata, README

**Files:**
- Modify: `docs/EXPERIMENT_JOURNAL.md` (prepend entry)
- Modify: `languages/*/final_output/metadata.json` (already regenerated by Task 8's exports — verify only)
- Modify: `README.md`

- [ ] **Step 1: Journal entry** — prepend to `docs/EXPERIMENT_JOURNAL.md`, matching house style, dated with the actual completion date, titled "**Eval redesign shipped: stratified CSLS suite + document-level panel; all five slots re-measured (Greek first run).**" Contents: the metric definitions in two sentences (CSLS k=10, 50k candidates, exact+syn columns, three regimes); the full per-slot suite table (from Task 8's records: rows = slot × target, columns = dictionary/interpolation/zero-shot top-1 exact/syn); split refinement results (leak-check numbers from Task 4 Step 5, Hittite's fix confirmed); Egyptian data-fix effects (anchors recovered by casefold, stopword-drop count); genre benchmark results (LOO per space vs majority baseline); parallels ranks + MRR; pointer to `docs/myth_study_plan.md` with its go/no-go verdicts.

- [ ] **Step 2: Verify metadata** — for each slot, confirm `final_output/metadata.json` reflects the new run (spot-check alpha + accuracy against the slot's results JSON). If `10_export_production.py` wrote stale-shaped metadata, fix the values by hand to match the results JSON and note it.

- [ ] **Step 3: README** — replace the Results section: the invalidation banner shrinks to one line ("Numbers before 2026-07-06 measured surface-variant memorization — see the journal."); the headline table becomes the suite (one row per slot × target, columns: dictionary top-1 exact, interpolation top-1 exact/syn, zero-shot top-1 exact/syn, genre-LOO for Sumerian), followed by two sentences explaining the three regimes. Update the "Research progress" list with the new entry (one line, linking to the journal).

- [ ] **Step 4: Final verification + commit**

```bash
python -m pytest -q          # expected: all pass
git add docs/EXPERIMENT_JOURNAL.md README.md languages/*/final_output/metadata.json
git commit -m "docs: eval-redesign results — suite tables, genre + parallels panels"
git log --oneline -14
```
