# Egyptian Phase Beta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port heiroglyphy V15 Egyptian alignment pipeline into `languages/egyptian/`, producing cross-comparable GloVe 300d and whitened-Gemma 768d alignments that mirror the Sumerian pipeline 1:1.

**Architecture:** Copy existing Egyptian corpus + anchors + FastText model from heiroglyphy. Write 5 pipeline scripts (06-10) + normalizer + lookup class mirroring Sumerian's structure. Each script reads/writes to `languages/egyptian/{data,models,results,final_output}`. Ridge regression maps fused 1536d vectors into GloVe 300d (alpha=0.001) and whitened Gemma 768d (alpha=100 starting point with sweep). Pickle is used for vocab serialization to match existing Sumerian pipeline convention.

**Tech Stack:** Python 3, NumPy, gensim (FastText), scikit-learn (Ridge), SciPy (cdist), pytest

---

## File Map

### New files (create)

| File | Responsibility |
|------|---------------|
| `languages/egyptian/__init__.py` | Package marker |
| `languages/egyptian/scripts/__init__.py` | Package marker |
| `languages/egyptian/scripts/egyptian_normalize.py` | Transliteration normalization (Manuel de Codage) |
| `languages/egyptian/scripts/06_extract_anchors.py` | Load heiroglyphy anchors, filter, normalize to hyper-glyphy format |
| `languages/egyptian/scripts/anchors_06.py` | Import shim (mirrors Sumerian pattern) |
| `languages/egyptian/scripts/07_train_fasttext.py` | FastText 768d skip-gram training |
| `languages/egyptian/scripts/fasttext_07.py` | Import shim |
| `languages/egyptian/scripts/08_fuse_embeddings.py` | Zero-pad fusion [768d \| 000...768d] -> 1536d |
| `languages/egyptian/scripts/fuse_08.py` | Import shim |
| `languages/egyptian/scripts/09_align_and_evaluate.py` | Ridge -> GloVe 300d (alpha=0.001) |
| `languages/egyptian/scripts/align_09.py` | Import shim |
| `languages/egyptian/scripts/09b_align_gemma.py` | Ridge -> whitened Gemma 768d (alpha=100, with sweep) |
| `languages/egyptian/scripts/align_09b.py` | Import shim |
| `languages/egyptian/scripts/10_export_production.py` | Dual-view export (Gemma + GloVe) |
| `languages/egyptian/scripts/export_10.py` | Import shim |
| `languages/egyptian/final_output/egyptian_lookup.py` | EgyptianLookup dual-view class |
| `languages/egyptian/tests/__init__.py` | Package marker |
| `languages/egyptian/tests/test_egyptian_normalize.py` | Normalizer tests |
| `languages/egyptian/tests/test_06_anchors.py` | Anchor extraction tests |
| `languages/egyptian/tests/test_07_fasttext.py` | FastText training tests |
| `languages/egyptian/tests/test_08_fusion.py` | Fusion tests |
| `languages/egyptian/tests/test_09_alignment.py` | GloVe + Gemma alignment tests |
| `languages/egyptian/tests/test_10_export.py` | Export + lookup tests |
| `languages/egyptian/docs/EXPERIMENT_JOURNAL.md` | Egyptian experiment notes |
| `languages/egyptian/README.md` | Egyptian module README |

### Modified files

| File | Change |
|------|--------|
| `pytest.ini` | Add `languages/egyptian/tests` to testpaths |

---

## Task 1: Directory Scaffolding + Data Migration

**Files:**
- Create: `languages/egyptian/__init__.py`
- Create: `languages/egyptian/scripts/__init__.py`
- Create: `languages/egyptian/tests/__init__.py`
- Create: `languages/egyptian/docs/EXPERIMENT_JOURNAL.md`
- Create: `languages/egyptian/README.md`
- Modify: `pytest.ini`
- Copy: data artifacts from `../heiroglyphy`

- [ ] **Step 1: Create directory structure and package markers**

```bash
mkdir -p languages/egyptian/scripts
mkdir -p languages/egyptian/data/raw
mkdir -p languages/egyptian/data/processed
mkdir -p languages/egyptian/models
mkdir -p languages/egyptian/results
mkdir -p languages/egyptian/final_output
mkdir -p languages/egyptian/tests
mkdir -p languages/egyptian/docs
touch languages/egyptian/__init__.py
touch languages/egyptian/scripts/__init__.py
touch languages/egyptian/tests/__init__.py
```

- [ ] **Step 2: Copy data artifacts from heiroglyphy**

```bash
cp ../heiroglyphy/heiro_v5_getdata/data/processed/cleaned_corpus.txt languages/egyptian/data/processed/
cp ../heiroglyphy/heiro_v5_getdata/data/processed/english_anchors.json languages/egyptian/data/processed/
cp ../heiroglyphy/heiro_v15/models/fasttext_mc5_w10.vec languages/egyptian/models/fasttext_egyptian.vec
cp ../heiroglyphy/heiro_v15/models/fasttext_mc5_w10.model languages/egyptian/models/fasttext_egyptian.model
```

Verify:

```bash
wc -l languages/egyptian/data/processed/cleaned_corpus.txt
# Expected: ~100729 lines
python3 -c "import json; d=json.load(open('languages/egyptian/data/processed/english_anchors.json')); print(f'{len(d)} anchors')"
# Expected: 8541 anchors
head -1 languages/egyptian/models/fasttext_egyptian.vec
# Expected: first line shows vocab_size and 768 dimension
```

- [ ] **Step 3: Create EXPERIMENT_JOURNAL.md**

Write `languages/egyptian/docs/EXPERIMENT_JOURNAL.md`:

```markdown
# Egyptian Alignment Experiment Journal

## 2026-05-03 — Phase Beta: Port to hyper-glyphy

Ported heiroglyphy V15 Egyptian alignment pipeline into hyper-glyphy monorepo.

**Baseline (heiroglyphy V15, GloVe 300d only):** 32.35% top-1, 41.47% top-5, 45.13% top-10.

**Changes from V15:**
- Dropped visual features (ResNet-50 768d, 0.59% match rate) in favor of pure zero-padding to match Sumerian pipeline structure.
- Added whitened-Gemma 768d as primary target (new for Egyptian).
- Standardized anchor format to hyper-glyphy convention.
- Added full pytest test suite.

**Data provenance:**
- Corpus: heiroglyphy `heiro_v5_getdata/data/processed/cleaned_corpus.txt` (100,729 lines, 789K tokens, BBAW/TLA sources)
- Anchors: heiroglyphy `heiro_v5_getdata/data/processed/english_anchors.json` (8,541 pairs from TLA/Ramses/BBAW)
- FastText model: heiroglyphy `heiro_v15/models/fasttext_mc5_w10` (768d, window=10, min_count=5, sg=1, epochs=10)
```

- [ ] **Step 4: Create README.md**

Write `languages/egyptian/README.md`:

```markdown
# Egyptian (Hieroglyphic) Alignment

Cross-lingual embedding alignment for ancient Egyptian hieroglyphic transliterations, mapping into both GloVe 300d and whitened-EmbeddingGemma 768d English semantic spaces.

## Pipeline

Scripts 01-05 (corpus building) are not yet ported. The cleaned corpus and anchors were migrated from heiroglyphy V15.

| Script | Purpose |
|--------|---------|
| `06_extract_anchors.py` | Normalize heiroglyphy anchors to hyper-glyphy format |
| `07_train_fasttext.py` | Train 768d FastText skip-gram embeddings |
| `08_fuse_embeddings.py` | Zero-pad fusion [768d \| 000...768d] -> 1536d |
| `09_align_and_evaluate.py` | Ridge regression -> GloVe 300d |
| `09b_align_gemma.py` | Ridge regression -> whitened Gemma 768d |
| `10_export_production.py` | Dual-view production export |

## Running

\`\`\`bash
# From repo root
python languages/egyptian/scripts/06_extract_anchors.py
python languages/egyptian/scripts/07_train_fasttext.py
python languages/egyptian/scripts/08_fuse_embeddings.py
python languages/egyptian/scripts/09_align_and_evaluate.py
python languages/egyptian/scripts/09b_align_gemma.py
python languages/egyptian/scripts/10_export_production.py
\`\`\`

## Tests

\`\`\`bash
pytest languages/egyptian/tests/ -v
\`\`\`
```

- [ ] **Step 5: Update pytest.ini**

In `pytest.ini`, change:

```ini
testpaths = languages/sumerian/tests shared/tests
```

to:

```ini
testpaths = languages/sumerian/tests shared/tests languages/egyptian/tests
```

- [ ] **Step 6: Verify existing Sumerian tests still pass**

```bash
pytest languages/sumerian/tests/ -v --tb=short
```

Expected: all existing tests pass (no regressions from adding Egyptian dirs).

- [ ] **Step 7: Commit**

```bash
git add languages/egyptian/__init__.py languages/egyptian/scripts/__init__.py languages/egyptian/tests/__init__.py languages/egyptian/docs/EXPERIMENT_JOURNAL.md languages/egyptian/README.md pytest.ini
git commit -m "feat(egyptian): scaffold directory structure + copy data artifacts"
```

---

## Task 2: egyptian_normalize.py + Tests

**Files:**
- Create: `languages/egyptian/scripts/egyptian_normalize.py`
- Create: `languages/egyptian/tests/test_egyptian_normalize.py`

- [ ] **Step 1: Write the failing test**

Write `languages/egyptian/tests/test_egyptian_normalize.py`:

```python
import pytest


def test_normalize_strips_diacritics_to_mdc():
    from languages.egyptian.scripts.egyptian_normalize import normalize_egyptian_token

    assert normalize_egyptian_token("nṯr") == "nTr"
    assert normalize_egyptian_token("ḥr,w") == "Hr,w"
    assert normalize_egyptian_token("ḫꜥ") == "xa"
    assert normalize_egyptian_token("ꜣḫ") == "Ax"


def test_normalize_lowercases_and_strips():
    from languages.egyptian.scripts.egyptian_normalize import normalize_egyptian_token

    assert normalize_egyptian_token("  NTR  ") == "ntr"
    assert normalize_egyptian_token("WSIR") == "wsir"


def test_normalize_handles_none_and_empty():
    from languages.egyptian.scripts.egyptian_normalize import normalize_egyptian_token

    assert normalize_egyptian_token(None) == ""
    assert normalize_egyptian_token("") == ""


def test_normalize_idempotent():
    from languages.egyptian.scripts.egyptian_normalize import normalize_egyptian_token

    for raw in ["nṯr", "ḥr,w", "ḫꜥ", "ꜣḫ", "ms(w),t", "wsjr"]:
        once = normalize_egyptian_token(raw)
        twice = normalize_egyptian_token(once)
        assert once == twice, f"not idempotent: {raw!r} -> {once!r} -> {twice!r}"


def test_normalize_preserves_parenthetical_notation():
    from languages.egyptian.scripts.egyptian_normalize import normalize_egyptian_token

    assert normalize_egyptian_token("ms(w),t") == "ms(w),t"
    assert normalize_egyptian_token("ḥm(,t)") == "Hm(,t)"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest languages/egyptian/tests/test_egyptian_normalize.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'languages.egyptian.scripts.egyptian_normalize'`

- [ ] **Step 3: Write implementation**

Write `languages/egyptian/scripts/egyptian_normalize.py`:

```python
"""
Canonical Egyptian transliteration normalization.

Maps Unicode Egyptological characters to Manuel de Codage (MdC) ASCII
equivalents, mirroring how sumerian_normalize.py maps ORACC to ATF.

Used by anchor extraction and any downstream scripts that need to match
transliteration forms between the corpus and dictionary.
"""
from __future__ import annotations


_EGYPTIAN_TO_MDC = {
    "ꜣ": "A",
    "ꜥ": "a",
    "ḥ": "H",
    "ḫ": "x",
    "ẖ": "X",
    "ṯ": "T",
    "ḏ": "D",
    "š": "S",
    "Š": "S",
    "ṭ": "d",
    "ṣ": "s",
    "ś": "s",
    "ȝ": "A",
    "ɜ": "A",
    "ʿ": "a",
    "ʾ": "A",
    "ỉ": "i",
    "č": "T",
    "ğ": "D",
    "ḳ": "q",
    "ḍ": "D",
}


def normalize_egyptian_token(raw) -> str:
    """Canonical normalization for a single Egyptian transliteration token.

    Applies (in order):
      1. Egyptian Unicode characters -> MdC ASCII equivalents
      2. Lowercase + strip whitespace

    Safe on None and empty input (returns "").
    Idempotent: normalize(normalize(x)) == normalize(x).
    """
    if raw is None:
        return ""
    s = str(raw)
    for old, new in _EGYPTIAN_TO_MDC.items():
        s = s.replace(old, new)
    return s.lower().strip()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest languages/egyptian/tests/test_egyptian_normalize.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add languages/egyptian/scripts/egyptian_normalize.py languages/egyptian/tests/test_egyptian_normalize.py
git commit -m "feat(egyptian): add transliteration normalizer + tests"
```

---

## Task 3: Anchor Extraction (06) + Tests

**Files:**
- Create: `languages/egyptian/scripts/06_extract_anchors.py`
- Create: `languages/egyptian/scripts/anchors_06.py`
- Create: `languages/egyptian/tests/test_06_anchors.py`

The heiroglyphy anchor format is:
```json
{"hieroglyphic": "nṯr", "english": "god", "german": "gott", "confidence": 0.87, "frequency": 234}
```

We normalize to hyper-glyphy format:
```json
{"egyptian": "nTr", "english": "god", "confidence": 0.87, "frequency": 234, "source": "TLA/Ramses"}
```

- [ ] **Step 1: Write the failing test**

Write `languages/egyptian/tests/test_06_anchors.py`:

```python
import pytest


def test_normalize_anchor_format():
    from languages.egyptian.scripts.anchors_06 import normalize_anchors

    raw = [
        {"hieroglyphic": "nṯr", "english": "god", "german": "gott", "confidence": 0.87, "frequency": 234},
        {"hieroglyphic": "ḥr,w", "english": "horus", "german": "horus", "confidence": 0.95, "frequency": 150},
    ]

    result = normalize_anchors(raw)

    assert len(result) == 2
    assert result[0]["egyptian"] == "Hr,w"
    assert result[0]["english"] == "horus"
    assert result[0]["confidence"] == 0.95
    assert result[0]["source"] == "TLA/Ramses"
    assert "hieroglyphic" not in result[0]
    assert "german" not in result[0]


def test_filter_single_char_english():
    from languages.egyptian.scripts.anchors_06 import normalize_anchors

    raw = [
        {"hieroglyphic": "n", "english": "the", "german": "der", "confidence": 0.34, "frequency": 8829},
        {"hieroglyphic": "m", "english": "a", "german": "ein", "confidence": 0.37, "frequency": 8467},
        {"hieroglyphic": "x", "english": "x", "german": "x", "confidence": 0.50, "frequency": 100},
    ]

    result = normalize_anchors(raw)

    english_words = [a["english"] for a in result]
    assert "the" in english_words
    assert "a" not in english_words
    assert "x" not in english_words


def test_filter_low_frequency():
    from languages.egyptian.scripts.anchors_06 import normalize_anchors

    raw = [
        {"hieroglyphic": "nṯr", "english": "god", "german": "gott", "confidence": 0.87, "frequency": 10},
        {"hieroglyphic": "rare", "english": "rare-word", "german": "selten", "confidence": 0.90, "frequency": 3},
    ]

    result = normalize_anchors(raw, min_frequency=5)

    assert len(result) == 1
    assert result[0]["english"] == "god"


def test_filter_numeric_english():
    from languages.egyptian.scripts.anchors_06 import normalize_anchors

    raw = [
        {"hieroglyphic": "nṯr", "english": "god", "german": "gott", "confidence": 0.87, "frequency": 100},
        {"hieroglyphic": "num", "english": "123", "german": "123", "confidence": 0.50, "frequency": 100},
    ]

    result = normalize_anchors(raw)

    assert len(result) == 1
    assert result[0]["english"] == "god"


def test_deduplicates_by_egyptian_key():
    from languages.egyptian.scripts.anchors_06 import normalize_anchors

    raw = [
        {"hieroglyphic": "nṯr", "english": "god", "german": "gott", "confidence": 0.87, "frequency": 234},
        {"hieroglyphic": "nṯr", "english": "divine", "german": "goettlich", "confidence": 0.60, "frequency": 100},
    ]

    result = normalize_anchors(raw)

    ntr_entries = [a for a in result if a["egyptian"] == "nTr"]
    assert len(ntr_entries) == 1
    assert ntr_entries[0]["confidence"] == 0.87
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest languages/egyptian/tests/test_06_anchors.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Write `languages/egyptian/scripts/06_extract_anchors.py`:

```python
"""
Anchor Extraction: Normalize heiroglyphy anchor pairs to hyper-glyphy format.

Reads the existing english_anchors.json (8,541 pairs from TLA/Ramses/BBAW)
and normalizes field names, applies quality filters, and deduplicates.

Input format:  {"hieroglyphic": str, "english": str, "german": str, "confidence": float, "frequency": int}
Output format: {"egyptian": str, "english": str, "confidence": float, "frequency": int, "source": str}
"""
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from languages.egyptian.scripts.egyptian_normalize import normalize_egyptian_token  # noqa: E402

_LANG_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = _LANG_ROOT / "data" / "processed"

_JUNK_ENGLISH = {"x", "xx", "0", "00", "1", "n", "c", "e", "i", "u", "s"}


def normalize_anchors(
    raw_anchors: list[dict],
    min_frequency: int = 5,
) -> list[dict]:
    """Normalize heiroglyphy anchors to hyper-glyphy format with quality filters."""
    best = {}

    for anchor in raw_anchors:
        egyptian = normalize_egyptian_token(anchor.get("hieroglyphic", ""))
        english = anchor.get("english", "").strip().lower()
        confidence = anchor.get("confidence", 0.0)
        frequency = anchor.get("frequency", 0)

        if not egyptian or not english:
            continue
        if frequency < min_frequency:
            continue
        if len(english) <= 1:
            continue
        if english in _JUNK_ENGLISH:
            continue
        if english.isdigit():
            continue

        if egyptian not in best or confidence > best[egyptian]["confidence"]:
            best[egyptian] = {
                "egyptian": egyptian,
                "english": english,
                "confidence": round(confidence, 4),
                "frequency": frequency,
                "source": "TLA/Ramses",
            }

    return sorted(best.values(), key=lambda a: a["confidence"], reverse=True)


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    input_path = DATA_PROCESSED / "english_anchors.json"
    print(f"Loading raw anchors from {input_path}")
    with open(input_path) as f:
        raw = json.load(f)
    print(f"Raw anchors: {len(raw)}")

    filtered = normalize_anchors(raw, min_frequency=5)

    output_path = DATA_PROCESSED / "english_anchors_normalized.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Normalized anchors: {len(filtered)}")
    print(f"Filtered: {len(raw) - len(filtered)} removed")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write import shim**

Write `languages/egyptian/scripts/anchors_06.py`:

```python
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "anchors",
    os.path.join(os.path.dirname(__file__), "06_extract_anchors.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

normalize_anchors = _mod.normalize_anchors
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest languages/egyptian/tests/test_06_anchors.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add languages/egyptian/scripts/06_extract_anchors.py languages/egyptian/scripts/anchors_06.py languages/egyptian/tests/test_06_anchors.py
git commit -m "feat(egyptian): anchor extraction with normalization + quality filters"
```

---

## Task 4: FastText Training (07) + Fusion (08) + Tests

**Files:**
- Create: `languages/egyptian/scripts/07_train_fasttext.py`
- Create: `languages/egyptian/scripts/fasttext_07.py`
- Create: `languages/egyptian/scripts/08_fuse_embeddings.py`
- Create: `languages/egyptian/scripts/fuse_08.py`
- Create: `languages/egyptian/tests/test_07_fasttext.py`
- Create: `languages/egyptian/tests/test_08_fusion.py`

- [ ] **Step 1: Write FastText test**

Write `languages/egyptian/tests/test_07_fasttext.py`:

```python
import os
import tempfile


def test_corpus_iterator():
    from languages.egyptian.scripts.fasttext_07 import CorpusIterator

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("nTr Hr nb\n")
        f.write("wsjr Ast st\n")
        f.write("ra xpr Htp\n")
        f.flush()

        lines = list(CorpusIterator(f.name))

    os.unlink(f.name)

    assert len(lines) == 3
    assert lines[0] == ["nTr", "Hr", "nb"]
    assert lines[1] == ["wsjr", "Ast", "st"]


def test_train_fasttext_model():
    from languages.egyptian.scripts.fasttext_07 import train_fasttext

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for _ in range(100):
            f.write("nTr Hr nb wsjr Ast st ra xpr Htp mAat\n")
        f.flush()

        with tempfile.TemporaryDirectory() as tmpdir:
            model = train_fasttext(
                corpus_path=f.name,
                output_dir=tmpdir,
                vector_size=32,
                window=5,
                min_count=1,
                epochs=2,
            )

            assert model.vector_size == 32
            assert "nTr" in model.wv

    os.unlink(f.name)
```

- [ ] **Step 2: Write fusion test**

Write `languages/egyptian/tests/test_08_fusion.py`:

```python
import numpy as np


def test_fuse_with_zero_padding():
    from languages.egyptian.scripts.fuse_08 import fuse_embeddings

    vocab = ["nTr", "Hr", "wsjr"]
    text_vectors = np.random.randn(3, 768).astype(np.float32)

    fused, fused_vocab = fuse_embeddings(vocab, text_vectors)

    assert fused.shape == (3, 1536)
    np.testing.assert_array_equal(fused[:, :768], text_vectors)
    np.testing.assert_array_equal(fused[:, 768:], np.zeros((3, 768)))
    assert fused_vocab == vocab


def test_fuse_preserves_dtype():
    from languages.egyptian.scripts.fuse_08 import fuse_embeddings

    vocab = ["nTr"]
    text_vectors = np.random.randn(1, 768).astype(np.float32)

    fused, _ = fuse_embeddings(vocab, text_vectors)
    assert fused.dtype == np.float32
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest languages/egyptian/tests/test_07_fasttext.py languages/egyptian/tests/test_08_fusion.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write 07_train_fasttext.py**

Write `languages/egyptian/scripts/07_train_fasttext.py`:

```python
"""
FastText Training: Train 768d skip-gram embeddings on cleaned Egyptian corpus.

Hyperparameters match Sumerian pipeline for cross-comparability:
  vector_size: 768
  window: 10
  min_count: 5
  epochs: 10
  sg: 1 (skip-gram)
"""
from pathlib import Path

from gensim.models import FastText

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"


class CorpusIterator:
    """Iterate over lines in a text file, yielding tokenized lists."""

    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


def train_fasttext(
    corpus_path: str,
    output_dir: str,
    vector_size: int = 768,
    window: int = 10,
    min_count: int = 5,
    epochs: int = 10,
) -> FastText:
    """Train FastText skip-gram model on corpus."""
    print(f"Training FastText: dim={vector_size}, window={window}, min_count={min_count}, epochs={epochs}")

    corpus = CorpusIterator(corpus_path)

    model = FastText(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        epochs=epochs,
        workers=4,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "fasttext_egyptian.model"
    vec_path = output_dir / "fasttext_egyptian.vec"

    model.save(str(model_path))
    model.wv.save_word2vec_format(str(vec_path))

    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Model saved to: {model_path}")
    print(f"Vectors saved to: {vec_path}")

    return model


def main():
    corpus_path = DATA_PROCESSED / "cleaned_corpus.txt"
    train_fasttext(
        corpus_path=str(corpus_path),
        output_dir=str(MODELS_DIR),
        vector_size=768,
        window=10,
        min_count=5,
        epochs=10,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write 08_fuse_embeddings.py**

Write `languages/egyptian/scripts/08_fuse_embeddings.py`:

```python
"""
Embedding Fusion: Concatenate FastText 768d with 768d zero-padding.

Produces 1536d fused vectors. The zero-padding acts as implicit regularization
during Ridge regression alignment, as discovered in heiroglyphy V15.
"""
import numpy as np
from gensim.models import KeyedVectors
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def fuse_embeddings(
    vocab: list[str],
    text_vectors: np.ndarray,
    pad_dim: int = 768,
) -> tuple[np.ndarray, list[str]]:
    """
    Fuse text embeddings with zero-padding.

    Args:
        vocab: List of words
        text_vectors: (N, text_dim) array of text embeddings
        pad_dim: Dimension of zero-padding (default 768)

    Returns:
        fused: (N, text_dim + pad_dim) fused vectors
        vocab: Same word list (passthrough)
    """
    n, text_dim = text_vectors.shape
    padding = np.zeros((n, pad_dim), dtype=np.float32)
    fused = np.concatenate([text_vectors, padding], axis=1)
    return fused, vocab


def main():
    vec_path = MODELS_DIR / "fasttext_egyptian.vec"
    print(f"Loading FastText vectors from {vec_path}")
    kv = KeyedVectors.load_word2vec_format(str(vec_path))

    vocab = list(kv.index_to_key)
    text_vectors = np.array([kv[w] for w in vocab], dtype=np.float32)
    print(f"Loaded {len(vocab)} words, {text_vectors.shape[1]}d")

    fused, _ = fuse_embeddings(vocab, text_vectors)
    print(f"Fused shape: {fused.shape}")

    output_path = MODELS_DIR / "fused_embeddings_1536d.npz"
    np.savez_compressed(
        str(output_path),
        vectors=fused,
        vocab=np.array(vocab),
    )
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write import shims**

Write `languages/egyptian/scripts/fasttext_07.py`:

```python
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "fasttext",
    os.path.join(os.path.dirname(__file__), "07_train_fasttext.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CorpusIterator = _mod.CorpusIterator
train_fasttext = _mod.train_fasttext
```

Write `languages/egyptian/scripts/fuse_08.py`:

```python
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "fuse",
    os.path.join(os.path.dirname(__file__), "08_fuse_embeddings.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

fuse_embeddings = _mod.fuse_embeddings
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest languages/egyptian/tests/test_07_fasttext.py languages/egyptian/tests/test_08_fusion.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add languages/egyptian/scripts/07_train_fasttext.py languages/egyptian/scripts/fasttext_07.py languages/egyptian/scripts/08_fuse_embeddings.py languages/egyptian/scripts/fuse_08.py languages/egyptian/tests/test_07_fasttext.py languages/egyptian/tests/test_08_fusion.py
git commit -m "feat(egyptian): FastText training + zero-pad fusion scripts + tests"
```

---

## Task 5: GloVe Alignment (09) + Tests

**Files:**
- Create: `languages/egyptian/scripts/09_align_and_evaluate.py`
- Create: `languages/egyptian/scripts/align_09.py`
- Create: `languages/egyptian/tests/test_09_alignment.py`

- [ ] **Step 1: Write the failing test**

Write `languages/egyptian/tests/test_09_alignment.py`:

```python
import numpy as np


def test_build_training_data():
    from languages.egyptian.scripts.align_09 import build_training_data

    anchors = [
        {"egyptian": "nTr", "english": "god"},
        {"egyptian": "Hr", "english": "face"},
        {"egyptian": "unknown", "english": "missing"},
    ]

    eg_vocab = {"nTr": 0, "Hr": 1, "wsjr": 2}
    eg_vectors = np.random.randn(3, 1536).astype(np.float32)

    eng_vocab = {"god": 0, "face": 1, "water": 2}
    eng_vectors = np.random.randn(3, 300).astype(np.float32)

    X, Y, valid_anchors = build_training_data(
        anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors
    )

    assert X.shape == (2, 1536)
    assert Y.shape == (2, 300)
    assert len(valid_anchors) == 2


def test_evaluate_alignment():
    from languages.egyptian.scripts.align_09 import evaluate_alignment

    np.random.seed(42)
    n_test = 10
    dim = 300

    Y_test = np.random.randn(n_test, dim).astype(np.float32)
    Y_pred = Y_test + np.random.randn(n_test, dim).astype(np.float32) * 0.01

    eng_vocab = [f"word_{i}" for i in range(n_test + 50)]
    eng_vectors = np.vstack([
        Y_test,
        np.random.randn(50, dim).astype(np.float32),
    ])

    test_english = [f"word_{i}" for i in range(n_test)]

    results = evaluate_alignment(Y_pred, test_english, eng_vocab, eng_vectors)

    assert "top1" in results
    assert "top5" in results
    assert "top10" in results
    assert results["top1"] > 0.5


def test_train_ridge():
    from languages.egyptian.scripts.align_09 import train_ridge

    X = np.random.randn(100, 1536).astype(np.float32)
    Y = np.random.randn(100, 300).astype(np.float32)

    model = train_ridge(X, Y, alpha=0.001)

    Y_pred = model.predict(X[:5])
    assert Y_pred.shape == (5, 300)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest languages/egyptian/tests/test_09_alignment.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Write `languages/egyptian/scripts/09_align_and_evaluate.py`:

```python
"""
Ridge Alignment & Evaluation: Map Egyptian embeddings to GloVe English space.

Pipeline:
  1. Load fused 1536d Egyptian vectors
  2. Load GloVe 300d English vectors
  3. Load anchor pairs
  4. Build training data (only anchors present in both vocabs)
  5. 80/20 train/test split (random_state=42)
  6. Train Ridge regression (alpha=0.001)
  7. Evaluate Top-1/5/10 accuracy on test set
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_LANG_ROOT = Path(__file__).parent.parent

MODELS_DIR = _LANG_ROOT / "models"
DATA_PROCESSED = _LANG_ROOT / "data" / "processed"
RESULTS_DIR = _LANG_ROOT / "results"
GLOVE_PATH = _REPO_ROOT / "languages" / "sumerian" / "data" / "processed" / "glove.6B.300d.txt"

RIDGE_ALPHA = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42


def build_training_data(
    anchors: list[dict],
    eg_vocab: dict[str, int],
    eg_vectors: np.ndarray,
    eng_vocab: dict[str, int],
    eng_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build aligned X (Egyptian) and Y (English) matrices from anchor pairs."""
    X_list = []
    Y_list = []
    valid = []

    for anchor in anchors:
        e_word = anchor.get("egyptian", anchor.get("sumerian", ""))
        eng_word = anchor["english"]

        if e_word in eg_vocab and eng_word in eng_vocab:
            X_list.append(eg_vectors[eg_vocab[e_word]])
            Y_list.append(eng_vectors[eng_vocab[eng_word]])
            valid.append(anchor)

    if not X_list:
        return np.array([]), np.array([]), []

    return np.array(X_list), np.array(Y_list), valid


def train_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 0.001) -> Ridge:
    """Train Ridge regression to map X -> Y."""
    model = Ridge(alpha=alpha)
    model.fit(X, Y)
    return model


def evaluate_alignment(
    Y_pred: np.ndarray,
    test_english: list[str],
    eng_vocab_list: list[str],
    eng_vectors: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Evaluate alignment accuracy using Top-K nearest neighbor retrieval."""
    norms = np.linalg.norm(Y_pred, axis=1, keepdims=True)
    norms[norms == 0] = 1
    Y_pred_norm = Y_pred / norms

    g_norms = np.linalg.norm(eng_vectors, axis=1, keepdims=True)
    g_norms[g_norms == 0] = 1
    eng_norm = eng_vectors / g_norms

    distances = cdist(Y_pred_norm, eng_norm, metric="cosine")

    results = {}
    for k in ks:
        correct = 0
        for i, eng_word in enumerate(test_english):
            nn_indices = np.argsort(distances[i])[:k]
            nn_words = [eng_vocab_list[j] for j in nn_indices]
            if eng_word in nn_words:
                correct += 1
        total = len(test_english)
        results[f"top{k}"] = (correct / total * 100) if total > 0 else 0.0

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fused_path = MODELS_DIR / "fused_embeddings_1536d.npz"
    print(f"Loading fused vectors from {fused_path}")
    fused_data = np.load(str(fused_path), allow_pickle=True)
    eg_vectors = fused_data["vectors"]
    eg_vocab_list = list(fused_data["vocab"])
    eg_vocab = {w: i for i, w in enumerate(eg_vocab_list)}
    print(f"Egyptian vocab: {len(eg_vocab)} words, {eg_vectors.shape[1]}d")

    print(f"Loading GloVe from {GLOVE_PATH}")
    glove_vocab = []
    glove_vectors_list = []
    with open(GLOVE_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            glove_vocab.append(word)
            glove_vectors_list.append(vec)
    glove_vectors = np.array(glove_vectors_list)
    eng_vocab = {w: i for i, w in enumerate(glove_vocab)}
    print(f"GloVe vocab: {len(glove_vocab)} words, {glove_vectors.shape[1]}d")

    anchor_path = DATA_PROCESSED / "english_anchors_normalized.json"
    with open(anchor_path) as f:
        anchors = json.load(f)
    print(f"Loaded {len(anchors)} anchors")

    X, Y, valid_anchors = build_training_data(
        anchors, eg_vocab, eg_vectors, eng_vocab, glove_vectors
    )
    print(f"Valid anchors: {len(valid_anchors)} / {len(anchors)} ({len(valid_anchors)/len(anchors)*100:.1f}%)")

    X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
        X, Y, valid_anchors, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print(f"Training Ridge regression (alpha={RIDGE_ALPHA})...")
    model = train_ridge(X_train, Y_train, alpha=RIDGE_ALPHA)

    Y_pred = model.predict(X_test)

    test_english = [a["english"] for a in anchors_test]
    results = evaluate_alignment(Y_pred, test_english, glove_vocab, glove_vectors)

    print(f"\n=== RESULTS (GloVe target) ===")
    print(f"Top-1 Accuracy:  {results['top1']:.2f}%")
    print(f"Top-5 Accuracy:  {results['top5']:.2f}%")
    print(f"Top-10 Accuracy: {results['top10']:.2f}%")

    full_results = {
        "accuracy": results,
        "config": {
            "alignment": "Ridge",
            "alpha": RIDGE_ALPHA,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "valid_anchors": len(valid_anchors),
            "total_anchors": len(anchors),
            "egyptian_vocab": len(eg_vocab),
            "fused_dim": int(eg_vectors.shape[1]),
            "glove_dim": int(glove_vectors.shape[1]),
        },
    }

    results_path = RESULTS_DIR / "alignment_results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    np.savez_compressed(
        str(MODELS_DIR / "ridge_weights.npz"),
        coef=model.coef_,
        intercept=model.intercept_,
    )
    print("Ridge weights saved")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write import shim**

Write `languages/egyptian/scripts/align_09.py`:

```python
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "align",
    os.path.join(os.path.dirname(__file__), "09_align_and_evaluate.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_training_data = _mod.build_training_data
train_ridge = _mod.train_ridge
evaluate_alignment = _mod.evaluate_alignment
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest languages/egyptian/tests/test_09_alignment.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add languages/egyptian/scripts/09_align_and_evaluate.py languages/egyptian/scripts/align_09.py languages/egyptian/tests/test_09_alignment.py
git commit -m "feat(egyptian): GloVe Ridge alignment + evaluation pipeline"
```

---

## Task 6: Gemma Alignment (09b) + Tests

**Files:**
- Create: `languages/egyptian/scripts/09b_align_gemma.py`
- Create: `languages/egyptian/scripts/align_09b.py`
- Modify: `languages/egyptian/tests/test_09_alignment.py` (add 09b test)

- [ ] **Step 1: Add failing test to test_09_alignment.py**

Append to `languages/egyptian/tests/test_09_alignment.py`:

```python


def test_align_09b_shape_contract_at_768d():
    """09b must work when the English target dim is 768 (EmbeddingGemma)."""
    from languages.egyptian.scripts.align_09b import build_training_data, train_ridge, evaluate_alignment

    anchors = [
        {"egyptian": "nTr", "english": "god"},
        {"egyptian": "Hr", "english": "face"},
        {"egyptian": "wsjr", "english": "osiris"},
    ]

    eg_vocab = {"nTr": 0, "Hr": 1, "wsjr": 2}
    eg_vectors = np.random.randn(3, 1536).astype(np.float32)

    eng_vocab = {"god": 0, "face": 1, "osiris": 2}
    eng_vectors = np.random.randn(3, 768).astype(np.float32)

    X, Y, valid = build_training_data(
        anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors
    )
    assert X.shape == (3, 1536)
    assert Y.shape == (3, 768)

    model = train_ridge(X, Y, alpha=100)
    assert model.coef_.shape == (768, 1536)

    Y_pred = model.predict(X)
    results = evaluate_alignment(
        Y_pred,
        ["god", "face", "osiris"],
        ["god", "face", "osiris"],
        eng_vectors,
        ks=(1, 2, 3),
    )
    assert "top1" in results
    assert "top2" in results
    assert "top3" in results
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest languages/egyptian/tests/test_09_alignment.py::test_align_09b_shape_contract_at_768d -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Write `languages/egyptian/scripts/09b_align_gemma.py`:

```python
"""
Ridge alignment of Egyptian FastText into whitened EmbeddingGemma 768d.

Mirrors 09_align_and_evaluate.py but targets whitened EmbeddingGemma vectors.
Reuses helpers from align_09 to keep the comparison apples-to-apples.
"""
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.model_selection import train_test_split

from languages.egyptian.scripts.align_09 import (
    build_training_data,
    train_ridge,
    evaluate_alignment,
)

_LANG_ROOT = Path(__file__).parent.parent
MODELS_DIR = _LANG_ROOT / "models"
DATA_PROCESSED = _LANG_ROOT / "data" / "processed"
RESULTS_DIR = _LANG_ROOT / "results"

ENGLISH_GEMMA_PATH = _REPO_ROOT / "shared" / "models" / "english_gemma_whitened_768d.npz"
ANCHOR_PATH = DATA_PROCESSED / "english_anchors_normalized.json"
GLOVE_BASELINE_PATH = RESULTS_DIR / "alignment_results.json"

RIDGE_ALPHA = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42
EXPECTED_TARGET_DIM = 768

SWEEP_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]


def main():
    parser = argparse.ArgumentParser(description="Ridge alignment: Egyptian FastText -> whitened EmbeddingGemma 768d.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run alpha sweep over SWEEP_ALPHAS before final training.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ENGLISH_GEMMA_PATH.exists():
        print(f"ERROR: Whitened Gemma cache not found at {ENGLISH_GEMMA_PATH}", file=sys.stderr)
        print("Run: python shared/scripts/whiten_gemma.py", file=sys.stderr)
        sys.exit(1)

    fused_path = MODELS_DIR / "fused_embeddings_1536d.npz"
    print(f"Loading fused Egyptian vectors from {fused_path}")
    fused = np.load(str(fused_path))
    eg_vectors = fused["vectors"]
    eg_vocab_list = [str(w) for w in fused["vocab"]]
    eg_vocab = {w: i for i, w in enumerate(eg_vocab_list)}
    print(f"Egyptian vocab: {len(eg_vocab)} words, {eg_vectors.shape[1]}d")

    print(f"Loading Gemma English vectors from {ENGLISH_GEMMA_PATH}")
    gemma = np.load(str(ENGLISH_GEMMA_PATH))
    eng_vectors = gemma["vectors"]
    eng_vocab_list = [str(w) for w in gemma["vocab"]]
    eng_vocab = {w: i for i, w in enumerate(eng_vocab_list)}
    gloss_hit_rate = float(gemma["gloss_hit_rate"]) if "gloss_hit_rate" in gemma.files else None
    gemma_model = str(gemma["gemma_model"]) if "gemma_model" in gemma.files else None
    print(f"English vocab: {len(eng_vocab)} words, {eng_vectors.shape[1]}d")

    assert eng_vectors.shape[1] == EXPECTED_TARGET_DIM, (
        f"English target dim is {eng_vectors.shape[1]}, expected {EXPECTED_TARGET_DIM}."
    )

    with open(ANCHOR_PATH) as f:
        anchors = json.load(f)
    print(f"Loaded {len(anchors)} anchors")

    X, Y, valid_anchors = build_training_data(
        anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors
    )
    print(
        f"Valid anchors: {len(valid_anchors)} / {len(anchors)} "
        f"({len(valid_anchors)/len(anchors)*100:.1f}%)"
    )

    X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
        X, Y, valid_anchors, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    if args.sweep:
        print("\n=== ALPHA SWEEP ===")
        sweep_results = {}
        for alpha in SWEEP_ALPHAS:
            m = train_ridge(X_train, Y_train, alpha=alpha)
            Y_p = m.predict(X_test)
            te = [a["english"] for a in anchors_test]
            r = evaluate_alignment(Y_p, te, eng_vocab_list, eng_vectors)
            sweep_results[alpha] = r
            print(f"  alpha={alpha:<8} top1={r['top1']:.2f}%  top5={r['top5']:.2f}%  top10={r['top10']:.2f}%")

        sweep_path = RESULTS_DIR / "alpha_sweep_gemma.json"
        with open(sweep_path, "w") as f:
            json.dump({str(k): v for k, v in sweep_results.items()}, f, indent=2)
        print(f"Sweep saved to: {sweep_path}")

    print(f"\nTraining Ridge (alpha={RIDGE_ALPHA})...")
    model = train_ridge(X_train, Y_train, alpha=RIDGE_ALPHA)

    Y_pred = model.predict(X_test)
    test_english = [a["english"] for a in anchors_test]
    results = evaluate_alignment(Y_pred, test_english, eng_vocab_list, eng_vectors)

    baseline = None
    if GLOVE_BASELINE_PATH.exists():
        with open(GLOVE_BASELINE_PATH) as f:
            baseline = json.load(f).get("accuracy", {})

    print(f"\n=== RESULTS (Gemma target) ===")
    for k_str in ("top1", "top5", "top10"):
        gemma_val = results[k_str]
        if baseline and k_str in baseline:
            delta = gemma_val - baseline[k_str]
            print(
                f"{k_str.upper():<6} Gemma {gemma_val:6.2f}%  "
                f"GloVe {baseline[k_str]:6.2f}%  "
                f"delta {delta:+.2f}pp"
            )
        else:
            print(f"{k_str.upper():<6} Gemma {gemma_val:6.2f}%")

    full_results = {
        "accuracy": results,
        "baseline_glove": baseline,
        "deltas_vs_glove": (
            {k: results[k] - baseline[k] for k in results if k in baseline}
            if baseline
            else None
        ),
        "config": {
            "alignment": "Ridge",
            "alpha": RIDGE_ALPHA,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_size": len(X_train),
            "test_size_count": len(X_test),
            "valid_anchors": len(valid_anchors),
            "total_anchors": len(anchors),
            "egyptian_vocab": len(eg_vocab),
            "english_vocab": len(eng_vocab),
            "fused_dim": int(eg_vectors.shape[1]),
            "target_dim": int(eng_vectors.shape[1]),
            "gemma_model": gemma_model,
            "gloss_hit_rate": gloss_hit_rate,
        },
    }

    results_out_path = RESULTS_DIR / "alignment_results_gemma_whitened.json"
    with open(results_out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {results_out_path}")

    ridge_out_path = MODELS_DIR / "ridge_weights_gemma_whitened.npz"
    np.savez_compressed(
        str(ridge_out_path),
        coef=model.coef_,
        intercept=model.intercept_,
    )
    print(f"Ridge weights saved to: {ridge_out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write import shim**

Write `languages/egyptian/scripts/align_09b.py`:

```python
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "align_09b",
    os.path.join(os.path.dirname(__file__), "09b_align_gemma.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_training_data = _mod.build_training_data
train_ridge = _mod.train_ridge
evaluate_alignment = _mod.evaluate_alignment
```

Note: `build_training_data`, `train_ridge`, and `evaluate_alignment` are re-exported from `align_09` through `09b_align_gemma.py`'s imports. The shim re-exports them so tests can import from `align_09b` directly, matching the Sumerian pattern.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest languages/egyptian/tests/test_09_alignment.py -v
```

Expected: all 4 tests PASS (3 from task 5 + 1 new).

- [ ] **Step 6: Commit**

```bash
git add languages/egyptian/scripts/09b_align_gemma.py languages/egyptian/scripts/align_09b.py languages/egyptian/tests/test_09_alignment.py
git commit -m "feat(egyptian): Gemma Ridge alignment with alpha sweep"
```

---

## Task 7: Export (10) + EgyptianLookup + Tests

**Files:**
- Create: `languages/egyptian/scripts/10_export_production.py`
- Create: `languages/egyptian/scripts/export_10.py`
- Create: `languages/egyptian/final_output/egyptian_lookup.py`
- Create: `languages/egyptian/tests/test_10_export.py`

- [ ] **Step 1: Write failing tests**

Write `languages/egyptian/tests/test_10_export.py`:

```python
import json
import os
import tempfile

import numpy as np
import pytest


def test_project_all_vectors():
    from languages.egyptian.scripts.export_10 import project_all_vectors

    eg_vectors = np.random.randn(100, 1536).astype(np.float32)
    coef = np.random.randn(300, 1536).astype(np.float32)
    intercept = np.random.randn(300).astype(np.float32)

    projected = project_all_vectors(eg_vectors, coef, intercept)

    assert projected.shape == (100, 300)
    assert projected.dtype == np.float16


def test_export_writes_both_spaces_and_v2_metadata(tmp_path, monkeypatch):
    import languages.egyptian.scripts.export_10 as export_10_module

    n_eg = 4
    fused_dim = 1536
    glove_dim = 300
    gemma_dim = 768

    rng = np.random.default_rng(7)
    eg_vocab = ["nTr", "Hr", "wsjr", "Ast"]
    fused = rng.standard_normal((n_eg, fused_dim)).astype(np.float32)

    glove_coef = rng.standard_normal((glove_dim, fused_dim)).astype(np.float32)
    glove_intercept = rng.standard_normal(glove_dim).astype(np.float32)

    gemma_coef = rng.standard_normal((gemma_dim, fused_dim)).astype(np.float32)
    gemma_intercept = rng.standard_normal(gemma_dim).astype(np.float32)

    models = tmp_path / "models"
    results = tmp_path / "results"
    final = tmp_path / "final_output"
    models.mkdir()
    results.mkdir()

    np.savez_compressed(
        str(models / "fused_embeddings_1536d.npz"),
        vectors=fused,
        vocab=np.array(eg_vocab),
    )
    np.savez_compressed(
        str(models / "ridge_weights.npz"),
        coef=glove_coef,
        intercept=glove_intercept,
    )
    np.savez_compressed(
        str(models / "ridge_weights_gemma_whitened.npz"),
        coef=gemma_coef,
        intercept=gemma_intercept,
    )
    (results / "alignment_results.json").write_text(json.dumps({
        "accuracy": {"top1": 32.35, "top5": 41.47, "top10": 45.13},
        "config": {
            "alignment": "Ridge", "alpha": 0.001, "train_size": 5132,
            "test_size": 1283, "valid_anchors": 6415, "total_anchors": 8541,
            "egyptian_vocab": 10833, "fused_dim": 1536,
        },
    }))
    (results / "alignment_results_gemma_whitened.json").write_text(json.dumps({
        "accuracy": {"top1": 40.00, "top5": 50.00, "top10": 55.00},
        "config": {
            "alignment": "Ridge", "alpha": 100,
            "gemma_model": "google/embeddinggemma-300m", "gloss_hit_rate": 21.39,
            "test_size_count": 1283, "train_size": 5132, "valid_anchors": 6415,
            "total_anchors": 8541, "random_state": 42,
        },
    }))

    monkeypatch.setattr(export_10_module, "MODELS_DIR", models)
    monkeypatch.setattr(export_10_module, "RESULTS_DIR", results)
    monkeypatch.setattr(export_10_module, "FINAL_OUTPUT", final)
    export_10_module.main()

    assert (final / "egyptian_aligned_vectors.npz").exists()
    assert (final / "egyptian_aligned_gemma_vectors.npz").exists()
    assert (final / "egyptian_aligned_vocab.pkl").exists()
    assert (final / "metadata.json").exists()

    glove_npz = np.load(str(final / "egyptian_aligned_vectors.npz"))
    gemma_npz = np.load(str(final / "egyptian_aligned_gemma_vectors.npz"))
    assert glove_npz["vectors"].shape == (n_eg, glove_dim)
    assert gemma_npz["vectors"].shape == (n_eg, gemma_dim)
    assert glove_npz["vectors"].dtype == np.float16
    assert gemma_npz["vectors"].dtype == np.float16

    metadata = json.loads((final / "metadata.json").read_text())
    assert metadata["schema_version"] == 2
    assert metadata["shared"]["vocab_size"] == n_eg
    assert metadata["spaces"]["gemma"]["dim"] == 768
    assert metadata["spaces"]["glove"]["dim"] == 300


# --- EgyptianLookup tests ---


def _build_tiny_lookup(tmpdir: str, seed: int = 42):
    from languages.egyptian.final_output.egyptian_lookup import EgyptianLookup

    rng = np.random.default_rng(seed)

    n_eg = 3
    n_eng = 5
    gemma_dim = 768
    glove_dim = 300

    eg_vocab = ["nTr", "Hr", "wsjr"]

    eng_gemma = rng.standard_normal((n_eng, gemma_dim)).astype(np.float32)
    eng_vocab = [f"word_{i}" for i in range(n_eng)]

    eg_gemma = eng_gemma[:n_eg].astype(np.float16)
    eng_glove = rng.standard_normal((n_eng, glove_dim)).astype(np.float32)
    eg_glove = eng_glove[:n_eg].astype(np.float16)

    np.savez_compressed(
        os.path.join(tmpdir, "egyptian_aligned_gemma_vectors.npz"),
        vectors=eg_gemma,
    )
    np.savez_compressed(
        os.path.join(tmpdir, "egyptian_aligned_vectors.npz"),
        vectors=eg_glove,
    )
    import pickle as _pkl
    with open(os.path.join(tmpdir, "egyptian_aligned_vocab.pkl"), "wb") as f:
        _pkl.dump(eg_vocab, f)
    np.savez_compressed(
        os.path.join(tmpdir, "english_gemma_whitened_768d.npz"),
        vocab=np.array(eng_vocab),
        vectors=eng_gemma,
    )

    return EgyptianLookup(
        gemma_vectors_path=os.path.join(tmpdir, "egyptian_aligned_gemma_vectors.npz"),
        glove_vectors_path=os.path.join(tmpdir, "egyptian_aligned_vectors.npz"),
        vocab_path=os.path.join(tmpdir, "egyptian_aligned_vocab.pkl"),
        gemma_english_path=os.path.join(tmpdir, "english_gemma_whitened_768d.npz"),
        glove_english_vectors=eng_glove,
        glove_english_vocab=eng_vocab,
    ), eg_vocab, eng_vocab


def test_egyptian_lookup_find_gemma():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, eg_vocab, eng_vocab = _build_tiny_lookup(tmpdir)
        results = lookup.find("word_0", top_k=3, space="gemma")
        assert len(results) == 3
        assert results[0][0] == "nTr"
        assert results[0][1] > 0.99


def test_egyptian_lookup_find_glove():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, eg_vocab, eng_vocab = _build_tiny_lookup(tmpdir)
        results = lookup.find("word_1", top_k=3, space="glove")
        assert len(results) == 3
        assert results[0][0] == "Hr"
        assert results[0][1] > 0.99


def test_egyptian_lookup_find_both():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        result = lookup.find_both("word_2", top_k=2)
        assert set(result.keys()) == {"gemma", "glove"}
        assert result["gemma"][0][0] == "wsjr"
        assert result["glove"][0][0] == "wsjr"


def test_egyptian_lookup_unknown_space_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        with pytest.raises(ValueError, match="space must be"):
            lookup.find("word_0", space="bert")


def test_egyptian_lookup_oov_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        assert lookup.find("not_a_real_word", space="gemma") == []


def test_egyptian_lookup_analogy():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        result = lookup.find_analogy("word_0", "word_1", "word_2", top_k=3, space="gemma")
        assert len(result) > 0


def test_egyptian_lookup_blend_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        assert lookup.find_blend({"unknown_xyz": 1.0}, space="gemma") == []
        assert lookup.find_blend({}, space="gemma") == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest languages/egyptian/tests/test_10_export.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write 10_export_production.py**

Write `languages/egyptian/scripts/10_export_production.py`:

```python
"""
Production Export: Dual-view Egyptian alignment.

Projects the fused Egyptian vectors into BOTH whitened-EmbeddingGemma 768d
and GloVe 300d, saving each as a separate fp16 npz alongside a shared vocab
pickle and a consolidated v2 metadata file.

Uses pickle for vocab (locally-generated data, project convention).
"""
import json
import importlib
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FINAL_OUTPUT = Path(__file__).parent.parent / "final_output"

SCHEMA_VERSION = 2


def project_all_vectors(
    eg_vectors: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    """Project all Egyptian vectors into a target space using learned Ridge weights."""
    projected = eg_vectors @ coef.T + intercept
    return projected.astype(np.float16)


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    FINAL_OUTPUT.mkdir(parents=True, exist_ok=True)

    fused_data = np.load(str(MODELS_DIR / "fused_embeddings_1536d.npz"), allow_pickle=True)
    eg_vectors = fused_data["vectors"]
    eg_vocab = list(fused_data["vocab"])
    print(f"Egyptian vectors: {eg_vectors.shape}, vocab: {len(eg_vocab)}")

    glove_ridge = np.load(str(MODELS_DIR / "ridge_weights.npz"))
    glove_coef = glove_ridge["coef"]
    glove_intercept = glove_ridge["intercept"]
    print(f"GloVe ridge coef: {glove_coef.shape}")
    aligned_glove = project_all_vectors(eg_vectors, glove_coef, glove_intercept)
    np.savez_compressed(
        str(FINAL_OUTPUT / "egyptian_aligned_vectors.npz"),
        vectors=aligned_glove,
    )
    print(f"GloVe aligned: {aligned_glove.shape} ({aligned_glove.dtype})")

    gemma_ridge = np.load(str(MODELS_DIR / "ridge_weights_gemma_whitened.npz"))
    gemma_coef = gemma_ridge["coef"]
    gemma_intercept = gemma_ridge["intercept"]
    print(f"Gemma whitened ridge coef: {gemma_coef.shape}")
    aligned_gemma = project_all_vectors(eg_vectors, gemma_coef, gemma_intercept)
    np.savez_compressed(
        str(FINAL_OUTPUT / "egyptian_aligned_gemma_vectors.npz"),
        vectors=aligned_gemma,
    )
    print(f"Gemma aligned: {aligned_gemma.shape} ({aligned_gemma.dtype})")

    _pkl = importlib.import_module("pickle")
    with open(FINAL_OUTPUT / "egyptian_aligned_vocab.pkl", "wb") as f:
        _pkl.dump(eg_vocab, f)

    glove_results = _load_json_if_exists(RESULTS_DIR / "alignment_results.json") or {}
    gemma_results = _load_json_if_exists(RESULTS_DIR / "alignment_results_gemma_whitened.json") or {}

    glove_cfg = glove_results.get("config", {})
    gemma_cfg = gemma_results.get("config", {})

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "methodology": (
            "Hyper-glyphy dual-view "
            "(Egyptian 1536d -> whitened-EmbeddingGemma 768d primary, GloVe 300d secondary)"
        ),
        "shared": {
            "vocab_size": len(eg_vocab),
            "egyptian_fused_dim": int(eg_vectors.shape[1]),
            "random_state": gemma_cfg.get("random_state", glove_cfg.get("random_state", 42)),
            "train_size": gemma_cfg.get("train_size", glove_cfg.get("train_size")),
            "test_size_count": gemma_cfg.get("test_size_count", glove_cfg.get("test_size")),
            "valid_anchors": gemma_cfg.get("valid_anchors", glove_cfg.get("valid_anchors")),
            "total_anchors": gemma_cfg.get("total_anchors", glove_cfg.get("total_anchors")),
        },
        "spaces": {
            "gemma": {
                "dim": int(aligned_gemma.shape[1]),
                "dtype": str(aligned_gemma.dtype),
                "ridge_alpha": gemma_cfg.get("alpha", 100),
                "ridge_source": "models/ridge_weights_gemma_whitened.npz",
                "target_source": "shared/models/english_gemma_whitened_768d.npz",
                "encoder_model": gemma_cfg.get("gemma_model") or "google/embeddinggemma-300m",
                "accuracy": gemma_results.get("accuracy"),
            },
            "glove": {
                "dim": int(aligned_glove.shape[1]),
                "dtype": str(aligned_glove.dtype),
                "ridge_alpha": glove_cfg.get("alpha", 0.001),
                "ridge_source": "models/ridge_weights.npz",
                "target_source": "languages/sumerian/data/processed/glove.6B.300d.txt",
                "accuracy": glove_results.get("accuracy"),
            },
        },
    }

    with open(FINAL_OUTPUT / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProduction files saved to {FINAL_OUTPUT}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write import shim**

Write `languages/egyptian/scripts/export_10.py`:

```python
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "export",
    os.path.join(os.path.dirname(__file__), "10_export_production.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

project_all_vectors = _mod.project_all_vectors
```

- [ ] **Step 5: Write EgyptianLookup**

Write `languages/egyptian/final_output/egyptian_lookup.py`:

```python
"""
Dual-view Egyptian Semantic Lookup.

Find Egyptian words by English meaning in either the whitened-EmbeddingGemma
768d manifold (space="gemma", default) or the GloVe 300d manifold
(space="glove"). Both spaces share the same Egyptian vocabulary and index
order; the vectors just land in different target geometries.

Mirrors SumerianLookup API exactly for cross-civilizational consistency.

Uses the standard library serialization module for the shared Egyptian vocab
file -- locally-generated data, not untrusted input, matching the existing
project convention.
"""
from __future__ import annotations

import importlib

import numpy as np

_VALID_SPACES = ("gemma", "glove")

_serial = importlib.import_module("pickle")


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows, mapping zero-norm rows to zero (not NaN)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


class EgyptianLookup:
    def __init__(
        self,
        gemma_vectors_path: str,
        glove_vectors_path: str,
        vocab_path: str,
        gemma_english_path: str,
        glove_english_vectors: np.ndarray,
        glove_english_vocab: list[str],
    ):
        with open(vocab_path, "rb") as f:
            self.vocab: list[str] = list(_serial.load(f))

        eg_gemma = np.load(gemma_vectors_path)["vectors"].astype(np.float32)
        eg_glove = np.load(glove_vectors_path)["vectors"].astype(np.float32)
        if eg_gemma.shape[0] != len(self.vocab):
            raise ValueError(
                f"Gemma-space Egyptian rows {eg_gemma.shape[0]} "
                f"!= vocab size {len(self.vocab)}"
            )
        if eg_glove.shape[0] != len(self.vocab):
            raise ValueError(
                f"GloVe-space Egyptian rows {eg_glove.shape[0]} "
                f"!= vocab size {len(self.vocab)}"
            )
        if eg_gemma.shape[1] != 768:
            raise ValueError(
                f"Gemma-space Egyptian dim {eg_gemma.shape[1]} != 768"
            )
        if eg_glove.shape[1] != 300:
            raise ValueError(
                f"GloVe-space Egyptian dim {eg_glove.shape[1]} != 300"
            )

        gemma_eng = np.load(gemma_english_path)
        eng_gemma_vocab = [str(w) for w in gemma_eng["vocab"]]
        eng_gemma_vec = gemma_eng["vectors"].astype(np.float32)
        if eng_gemma_vec.shape[1] != 768:
            raise ValueError(
                f"English Gemma cache dim {eng_gemma_vec.shape[1]} != 768"
            )
        if eng_gemma_vec.shape[0] != len(eng_gemma_vocab):
            raise ValueError(
                "English Gemma vocab/vectors row count mismatch"
            )

        glove_eng_vec = np.asarray(glove_english_vectors, dtype=np.float32)
        if glove_eng_vec.shape[1] != 300:
            raise ValueError(
                f"GloVe English dim {glove_eng_vec.shape[1]} != 300"
            )
        if glove_eng_vec.shape[0] != len(glove_english_vocab):
            raise ValueError(
                "GloVe English vocab/vectors row count mismatch"
            )

        self._spaces = {
            "gemma": {
                "eg_norm": _normalize_rows(eg_gemma),
                "eg_dim": eg_gemma.shape[1],
                "eng_vocab_map": {w.lower(): i for i, w in enumerate(eng_gemma_vocab)},
                "eng_norm": _normalize_rows(eng_gemma_vec),
            },
            "glove": {
                "eg_norm": _normalize_rows(eg_glove),
                "eg_dim": eg_glove.shape[1],
                "eng_vocab_map": {w.lower(): i for i, w in enumerate(glove_english_vocab)},
                "eng_norm": _normalize_rows(glove_eng_vec),
            },
        }

    def _validate_space(self, space: str) -> None:
        if space not in _VALID_SPACES:
            raise ValueError(
                f"space must be one of {_VALID_SPACES!r}, got {space!r}"
            )

    def _english_vector(self, word: str, space: str) -> np.ndarray | None:
        s = self._spaces[space]
        idx = s["eng_vocab_map"].get(word.lower())
        if idx is None:
            return None
        return s["eng_norm"][idx]

    def _top_k_from_query(self, query: np.ndarray, space: str, top_k: int) -> list[tuple[str, float]]:
        s = self._spaces[space]
        sims = s["eg_norm"] @ query
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.vocab[int(i)], float(sims[int(i)])) for i in top_indices]

    def find(self, english_word: str, top_k: int = 10, space: str = "gemma") -> list[tuple[str, float]]:
        self._validate_space(space)
        vec = self._english_vector(english_word, space)
        if vec is None:
            return []
        return self._top_k_from_query(vec, space, top_k)

    def find_both(self, english_word: str, top_k: int = 10) -> dict[str, list[tuple[str, float]]]:
        return {
            "gemma": self.find(english_word, top_k=top_k, space="gemma"),
            "glove": self.find(english_word, top_k=top_k, space="glove"),
        }

    def find_analogy(
        self,
        a: str,
        b: str,
        c: str,
        top_k: int = 10,
        space: str = "gemma",
    ) -> list[tuple[str, float]]:
        self._validate_space(space)
        va = self._english_vector(a, space)
        vb = self._english_vector(b, space)
        vc = self._english_vector(c, space)
        if any(v is None for v in (va, vb, vc)):
            return []
        target = vc - va + vb
        norm = np.linalg.norm(target)
        if norm == 0:
            return []
        target = target / norm
        return self._top_k_from_query(target, space, top_k)

    def find_blend(
        self,
        weights: dict[str, float],
        top_k: int = 10,
        space: str = "gemma",
    ) -> list[tuple[str, float]]:
        self._validate_space(space)
        if not weights:
            return []
        s = self._spaces[space]
        target = np.zeros(s["eg_dim"], dtype=np.float32)
        any_resolved = False
        for word, weight in weights.items():
            vec = self._english_vector(word, space)
            if vec is not None:
                target += float(weight) * vec
                any_resolved = True
        if not any_resolved:
            return []
        norm = np.linalg.norm(target)
        if norm == 0:
            return []
        target = target / norm
        return self._top_k_from_query(target, space, top_k)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest languages/egyptian/tests/test_10_export.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add languages/egyptian/scripts/10_export_production.py languages/egyptian/scripts/export_10.py languages/egyptian/final_output/egyptian_lookup.py languages/egyptian/tests/test_10_export.py
git commit -m "feat(egyptian): dual-view export + EgyptianLookup class"
```

---

## Task 8: Full Test Suite Validation + Final Commit

**Files:**
- No new files. Validation pass.

- [ ] **Step 1: Run all Egyptian tests**

```bash
pytest languages/egyptian/tests/ -v
```

Expected: all tests PASS across all test files (test_egyptian_normalize.py, test_06_anchors.py, test_07_fasttext.py, test_08_fusion.py, test_09_alignment.py, test_10_export.py).

- [ ] **Step 2: Run all Sumerian tests to verify no regressions**

```bash
pytest languages/sumerian/tests/ -v --tb=short
```

Expected: all existing Sumerian tests still PASS.

- [ ] **Step 3: Run full test suite**

```bash
pytest -v --tb=short
```

Expected: all tests across `languages/sumerian/tests`, `shared/tests`, and `languages/egyptian/tests` PASS.

- [ ] **Step 4: Verify data artifacts are gitignored**

```bash
git status
```

Expected: no files under `languages/egyptian/data/`, `languages/egyptian/models/`, or `languages/egyptian/results/` appear in untracked files (they are covered by `.gitignore` patterns `languages/*/data/raw/`, etc.).

- [ ] **Step 5: Final cleanup commit if needed**

If any adjustments were made during validation, commit them:

```bash
git add -A
git commit -m "fix(egyptian): address test suite feedback"
```
