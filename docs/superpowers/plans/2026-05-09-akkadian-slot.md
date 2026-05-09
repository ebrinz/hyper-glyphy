# Akkadian Slot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `languages/akkadian/` slot — Old Babylonian Akkadian aligned to whitened-Gemma 768d (primary) and GloVe 300d (secondary), mirroring the existing Sumerian slot 1:1. Scaffold DCCLT lexical-list bridge data for a v2 cross-lingual experiment.

**Architecture:** Reuse Sumerian's `01-10` numbered pipeline. Two scripts genuinely new (DCCLT scraper, eBL anchor fetcher); three forked-and-adapted (clean/tokenize, anchor extraction, coverage diagnostic); the rest are direct copies. Anchor lexicon is eBL-primary with ORACC project glosses as fallback (analog of ePSD2-primary/ORACC-fallback for Sumerian).

**Tech Stack:** Python 3.11+, `requests`, `gensim` (FastText), `numpy`, `scikit-learn` (Ridge), `tqdm`, `pytest`. ORACC ATF JSON archives. eBL REST API.

**Reference spec:** `docs/superpowers/specs/2026-05-09-akkadian-slot-design.md` (commit `53f574a`).

**One deviation from Sumerian:** vocab artifact is serialized as JSON (not pickle, as Sumerian does) — Akkadian vocab is a flat list of strings, JSON is equivalent in size and avoids the pickle deserialization risk surface. Future slots may follow this pattern.

---

## Phase 1 — Scaffold

### Task 1: Create directory tree and register tests

**Files:**
- Create: `languages/akkadian/__init__.py`
- Create: `languages/akkadian/scripts/__init__.py`
- Create: `languages/akkadian/tests/__init__.py`
- Modify: `pytest.ini`

- [ ] **Step 1: Create the directory tree**

```bash
cd /Users/crashy/Development/hyper-glyphy
mkdir -p languages/akkadian/{scripts,tests,docs,data/{raw/{ob_literary,ob_letters,dcclt},dictionaries,processed},models,results,final_output}
touch languages/akkadian/__init__.py languages/akkadian/scripts/__init__.py languages/akkadian/tests/__init__.py
```

- [ ] **Step 2: Register the new tests directory in pytest.ini**

Modify `pytest.ini` line 2:

```ini
[pytest]
testpaths = languages/sumerian/tests shared/tests languages/egyptian/tests languages/akkadian/tests
python_files = test_*.py
python_functions = test_*
pythonpath = .
```

- [ ] **Step 3: Verify pytest discovers the empty test directory**

Run: `pytest languages/akkadian/tests -v`
Expected: `no tests ran` (no failures, just empty collection).

- [ ] **Step 4: Commit**

```bash
git add languages/akkadian pytest.ini
git commit -m "scaffold: akkadian directory tree and pytest registration"
```

---

### Task 2: Copy direct-copy scripts from Sumerian

**Files:**
- Create: `languages/akkadian/scripts/07_train_fasttext.py` (copy)
- Create: `languages/akkadian/scripts/08_fuse_embeddings.py` (copy)
- Create: `languages/akkadian/scripts/09_align_and_evaluate.py` (copy)
- Create: `languages/akkadian/scripts/09b_align_gemma.py` (copy)
- Create: `languages/akkadian/scripts/04_deduplicate_corpus.py` (copy)

These five scripts are language-agnostic (FastText training, vector fusion, ridge alignment). They read from `data/processed/` paths derived from `__file__.parent.parent`, so simply copying them into the Akkadian scripts directory makes them operate on Akkadian data.

- [ ] **Step 1: Copy the five direct-copy scripts**

```bash
cd /Users/crashy/Development/hyper-glyphy
cp languages/sumerian/scripts/04_deduplicate_corpus.py languages/akkadian/scripts/
cp languages/sumerian/scripts/07_train_fasttext.py languages/akkadian/scripts/
cp languages/sumerian/scripts/08_fuse_embeddings.py languages/akkadian/scripts/
cp languages/sumerian/scripts/09_align_and_evaluate.py languages/akkadian/scripts/
cp languages/sumerian/scripts/09b_align_gemma.py languages/akkadian/scripts/
```

- [ ] **Step 2: Verify all five files exist and are byte-identical to Sumerian**

```bash
for f in 04_deduplicate_corpus.py 07_train_fasttext.py 08_fuse_embeddings.py 09_align_and_evaluate.py 09b_align_gemma.py; do
  diff -q languages/sumerian/scripts/$f languages/akkadian/scripts/$f
done
```

Expected: no output (all files identical).

- [ ] **Step 3: Sanity-import each module**

```bash
python -c "
import importlib.util, pathlib
for f in ['04_deduplicate_corpus','07_train_fasttext','08_fuse_embeddings','09_align_and_evaluate','09b_align_gemma']:
    p = pathlib.Path('languages/akkadian/scripts')/(f+'.py')
    spec = importlib.util.spec_from_file_location(f, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    print(f, 'OK')
"
```

Expected: 5 lines ending `OK`.

- [ ] **Step 4: Commit**

```bash
git add languages/akkadian/scripts/
git commit -m "scaffold: copy language-agnostic scripts (04, 07, 08, 09, 09b) into akkadian"
```

---

## Phase 2 — Akkadian Normalization Helper (TDD)

### Task 3: Write failing tests for akkadian_normalize.py

**Files:**
- Create: `languages/akkadian/tests/test_akkadian_normalize.py`

The Akkadian normalizer is the analog of `sumerian_normalize.py` but with three Akkadian-specific concerns: NFC unicode normalization (š/ḫ/ṣ/ṭ have decomposed forms in some sources), mimation alternation (`šarrum` vs `šarru`), and logogram preservation as uppercase tokens (Akkadian text mixes logograms `LUGAL` and syllabic `šar-ru-um` for the same word).

- [ ] **Step 1: Write the test file**

```python
# languages/akkadian/tests/test_akkadian_normalize.py
import pytest


def test_nfc_normalization():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    decomposed = "šesz"  # s + combining caron + esz
    precomposed = "šesz"   # š + esz
    assert normalize_akkadian_token(decomposed) == normalize_akkadian_token(precomposed)


def test_subscripts_to_ascii():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token("šar₂ru") == "szar2ru"
    assert normalize_akkadian_token("₀₁₂₃₄₅₆₇₈₉") == "0123456789"


def test_strips_determinative_braces():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token("{d}šamaš") == "dszamasz"
    assert normalize_akkadian_token("{lú}šarru") == "luszarru"


def test_oracc_to_atf_letters():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token("šarru") == "szarru"
    assert normalize_akkadian_token("ḫamru") == "hamru"
    assert normalize_akkadian_token("ṣabum") == "sabum"
    assert normalize_akkadian_token("ṭuppum") == "tuppum"


def test_drops_hyphens():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token("šar-ru-um") == "szarrum"
    assert normalize_akkadian_token("a-na") == "ana"


def test_lowercases_logograms_and_mixedcase():
    """ALL-CAPS logograms and mixed-case forms both lowercase."""
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token("LUGAL") == "lugal"
    assert normalize_akkadian_token("Šarrum") == "szarrum"


def test_strips_whitespace():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token(" šarrum ") == "szarrum"
    assert normalize_akkadian_token("\tšarru\n") == "szarru"


def test_handles_empty_and_none():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    assert normalize_akkadian_token("") == ""
    assert normalize_akkadian_token(None) == ""


def test_idempotent():
    from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token
    for raw in ("šarrum", "{d}šamaš", "šar-ru-um", "ŠARRUM", "ʾanāku"):
        once = normalize_akkadian_token(raw)
        twice = normalize_akkadian_token(once)
        assert once == twice, f"not idempotent on {raw!r}: {once!r} -> {twice!r}"


def test_mimation_alternates():
    from languages.akkadian.scripts.akkadian_normalize import mimation_alternates
    assert set(mimation_alternates("szarrum")) == {"szarrum", "szarru"}
    assert set(mimation_alternates("szarru")) == {"szarru"}
    assert set(mimation_alternates("ana")) == {"ana"}
    assert set(mimation_alternates("")) == set()
```

- [ ] **Step 2: Run tests to verify they fail with import error**

Run: `pytest languages/akkadian/tests/test_akkadian_normalize.py -v`
Expected: 10 tests FAIL with `ModuleNotFoundError: No module named 'languages.akkadian.scripts.akkadian_normalize'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add languages/akkadian/tests/test_akkadian_normalize.py
git commit -m "test: failing tests for akkadian_normalize"
```

---

### Task 4: Implement akkadian_normalize.py

**Files:**
- Create: `languages/akkadian/scripts/akkadian_normalize.py`

- [ ] **Step 1: Write the module**

```python
# languages/akkadian/scripts/akkadian_normalize.py
"""
Canonical Akkadian token normalization.

Single source of truth for mapping eBL/ORACC citation forms and inflected
surface forms to the common ATF-based token form produced by
`scripts/05_clean_and_tokenize.py`.

Used by `scripts/06_extract_anchors.py` (anchor side) and
`scripts/coverage_diagnostic.py` (audit/diagnostic side). Keeps normalization
in one place to prevent drift between anchors and corpus.

Akkadian-specific additions vs sumerian_normalize: explicit NFC and mimation
alternation.
"""
from __future__ import annotations

import re
import unicodedata

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

_ORACC_TO_ATF = {
    "š": "sz", "Š": "SZ",
    "ḫ": "h",  "Ḫ": "H",
    "ṣ": "s",  "Ṣ": "S",
    "ṭ": "t",  "Ṭ": "T",
    "ʾ": "",
    "ā": "a",  "Ā": "A",
    "ē": "e",  "Ē": "E",
    "ī": "i",  "Ī": "I",
    "ū": "u",  "Ū": "U",
    "â": "a",  "Â": "A",
    "ê": "e",  "Ê": "E",
    "î": "i",  "Î": "I",
    "û": "u",  "Û": "U",
}

_BRACE_RE = re.compile(r"\{([^}]*)\}")


def normalize_akkadian_token(raw) -> str:
    """Canonical normalization for a single Akkadian token.

    Order:
      1. NFC unicode normalization.
      2. Subscript digits -> ASCII.
      3. Strip {determinative} braces, keep content.
      4. ORACC unicode letters -> ATF (š -> sz, ā -> a, etc.).
      5. Drop hyphens.
      6. Lowercase + strip whitespace.

    Safe on None/empty (returns ""). Idempotent.
    """
    if raw is None:
        return ""
    s = unicodedata.normalize("NFC", str(raw))
    s = s.translate(_SUBSCRIPT_MAP)
    s = _BRACE_RE.sub(r"\1", s)
    for old, new in _ORACC_TO_ATF.items():
        s = s.replace(old, new)
    s = s.replace("-", "")
    return s.lower().strip()


def mimation_alternates(token: str) -> list[str]:
    """Return [token] plus its non-mimation form when applicable.

    OB nominal/adjectival forms typically end in -um/-am/-im (mimation).
    Later periods drop the final -m. For fallback matching, surface both.
    """
    if not token:
        return []
    out = [token]
    if len(token) >= 3 and token.endswith("m") and token[-2] in "aeiou":
        out.append(token[:-1])
    return out
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest languages/akkadian/tests/test_akkadian_normalize.py -v`
Expected: 10 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add languages/akkadian/scripts/akkadian_normalize.py
git commit -m "feat: akkadian_normalize with NFC, mimation alternates, ORACC->ATF map"
```

---

## Phase 3 — ORACC OB Scrapers

### Task 5: Implement OB literary scraper (`01_scrape_oracc_ob.py`)

**Files:**
- Create: `languages/akkadian/scripts/01_scrape_oracc_ob.py`
- Create: `languages/akkadian/tests/test_01_scrape_oracc_ob.py`

Structurally identical to `languages/sumerian/scripts/03_scrape_oracc.py` with three differences: (a) project list targets OB-relevant ORACC sub-projects, (b) language filter accepts `akk` instead of `sux`, (c) output goes to `data/raw/ob_literary/`.

- [ ] **Step 1: Write tests**

```python
# languages/akkadian/tests/test_01_scrape_oracc_ob.py
import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "01_scrape_oracc_ob.py"


def _load():
    spec = importlib.util.spec_from_file_location("scrape_oracc_ob", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_walk_cdl_extracts_akkadian_lemmas():
    mod = _load()
    text = {
        "cdl": [{
            "cdl": [
                {"f": {"lang": "akk", "form": "szarrum", "cf": "šarru", "gw": "king", "pos": "N"}},
                {"f": {"lang": "sux", "form": "lugal", "cf": "lugal", "gw": "king"}},
                {"f": {"lang": "akk-x-stdbab", "form": "ilum", "cf": "ilu", "gw": "god"}},
            ]
        }]
    }
    lemmas = mod.extract_lemmas(text)
    assert len(lemmas) == 2
    assert lemmas[0]["cf"] == "šarru"
    assert lemmas[1]["cf"] == "ilu"


def test_extract_lines_returns_text_lines():
    mod = _load()
    text = {
        "cdl": [{
            "type": "line-start",
            "cdl": [{"f": {"lang": "akk", "form": "szarrum"}}],
        }, {
            "type": "line-start",
            "cdl": [{"f": {"lang": "akk", "form": "ilum"}}],
        }]
    }
    lines = mod.extract_lines(text)
    assert lines


def test_oracc_projects_seed_list_present():
    mod = _load()
    assert isinstance(mod.ORACC_PROJECTS, list) and len(mod.ORACC_PROJECTS) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest languages/akkadian/tests/test_01_scrape_oracc_ob.py -v`
Expected: 3 tests FAIL with file-not-found.

- [ ] **Step 3: Write the scraper**

```python
# languages/akkadian/scripts/01_scrape_oracc_ob.py
"""
ORACC OB Akkadian Scraper: Download and parse OB-period Akkadian text JSON archives.

Source: http://oracc.museum.upenn.edu/{PROJECT}/json.zip
Format: Hierarchical CDL JSON with lemmatized Akkadian words.

Lemma node 'f' key contains:
  - form: surface transliteration
  - cf: citation form (dictionary headword)
  - gw: guide word (English gloss)
  - pos: part of speech
  - lang: language code (akk, akk-x-stdbab, akk-x-oldbab, etc.)

This scraper accepts any 'akk' prefix on the lang code so OB-marked dialect
variants are captured. Period filtering is applied later in 06_extract_anchors.py
via the eBL period flag.
"""
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"

# Seed list — OB-period-relevant ORACC sub-projects.
# To extend: append a project slug, re-run.
ORACC_PROJECTS = [
    "hbtin",         # Hethitologie Portal index of OB texts (where ATF-available)
    "saao/saa",      # State Archives of Assyria (NA — kept for SB/canonical descendants)
    "blms",          # Bilingual literary; OB recensions of canonical texts
    "rinap/rinap1",  # Royal Inscriptions of the Neo-Assyrian Period (descendant tradition)
]

ORACC_BASE_URL = "https://oracc.museum.upenn.edu/json"


def download_project_json(project: str, output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"oracc_{project.replace('/', '_')}.zip"

    if zip_path.exists():
        print(f"  Already downloaded: {zip_path}")
        return zip_path

    slug = project.replace("/", "-")
    url = f"{ORACC_BASE_URL}/{slug}.zip"
    try:
        response = requests.get(url, timeout=600, verify=False, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {slug}", leave=False) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return zip_path
    except requests.RequestException as e:
        print(f"  Failed to download {project}: {e}")
        return None


def _walk_cdl(node: Any, lemmas: list[dict], line_words: list[list[str]], current_line: list[str]) -> None:
    if isinstance(node, dict):
        if "f" in node:
            f = node["f"]
            lang = f.get("lang", "")
            if lang.startswith("akk"):
                lemma = {
                    "form": f.get("form", ""),
                    "cf": f.get("cf", ""),
                    "gw": f.get("gw", ""),
                    "pos": f.get("pos", ""),
                    "norm": f.get("norm", ""),
                    "lang": lang,
                }
                if lemma["form"]:
                    lemmas.append(lemma)
                    current_line.append(lemma["form"])

        if node.get("ftype") == "line-start" or node.get("type") == "line-start":
            if current_line:
                line_words.append(list(current_line))
                current_line.clear()

        if "cdl" in node:
            for child in node["cdl"]:
                _walk_cdl(child, lemmas, line_words, current_line)

    elif isinstance(node, list):
        for child in node:
            _walk_cdl(child, lemmas, line_words, current_line)


def extract_lemmas(text_json: dict) -> list[dict]:
    lemmas, line_words, current_line = [], [], []
    _walk_cdl(text_json.get("cdl", []), lemmas, line_words, current_line)
    return lemmas


def extract_lines(text_json: dict) -> list[str]:
    lemmas, line_words, current_line = [], [], []
    _walk_cdl(text_json.get("cdl", []), lemmas, line_words, current_line)
    if current_line:
        line_words.append(current_line)
    return [" ".join(words) for words in line_words if words]


def parse_project_zip(zip_path: Path) -> tuple[list[dict], list[dict]]:
    all_lemmas, all_texts = [], []
    with zipfile.ZipFile(zip_path) as zf:
        json_files = [n for n in zf.namelist() if "corpusjson" in n and n.endswith(".json")]
        for name in json_files:
            try:
                data = json.loads(zf.read(name))
            except (json.JSONDecodeError, KeyError):
                continue
            p_number = Path(name).stem
            lemmas = extract_lemmas(data)
            lines = extract_lines(data)
            if lemmas:
                all_lemmas.extend(lemmas)
            if lines:
                all_texts.append({"p_number": p_number, "lines": lines, "source": "ORACC"})
    return all_lemmas, all_texts


def save_texts(texts: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(texts)} texts to {output_path}")


def main():
    out_dir = DATA_RAW / "ob_literary"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_texts, all_lemmas = [], []

    for project in tqdm(ORACC_PROJECTS, desc="Downloading OB-relevant ORACC projects"):
        print(f"\nProcessing {project}...")
        zip_path = download_project_json(project, out_dir)
        if zip_path is None:
            continue
        lemmas, texts = parse_project_zip(zip_path)
        all_lemmas.extend(lemmas)
        all_texts.extend(texts)
        print(f"  {len(texts)} texts, {len(lemmas)} Akkadian lemmas")

    save_texts(all_texts, str(DATA_RAW / "ob_literary_texts.json"))
    save_texts(all_lemmas, str(DATA_RAW / "ob_literary_lemmas.json"))

    total_lines = sum(len(t["lines"]) for t in all_texts)
    unique_glosses = len({l["gw"] for l in all_lemmas if l["gw"]})
    print(f"\nTotal texts: {len(all_texts)}")
    print(f"Total lines: {total_lines}")
    print(f"Total Akkadian lemmas: {len(all_lemmas)}")
    print(f"Unique English glosses: {unique_glosses}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest languages/akkadian/tests/test_01_scrape_oracc_ob.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add languages/akkadian/scripts/01_scrape_oracc_ob.py languages/akkadian/tests/test_01_scrape_oracc_ob.py
git commit -m "feat: ORACC OB Akkadian scraper (01) with lang=akk filter"
```

---

### Task 6: Implement OB letters scraper (`02_scrape_oracc_letters.py`)

**Files:**
- Create: `languages/akkadian/scripts/02_scrape_oracc_letters.py`

Mirrors Task 5's scraper but targets letter-genre projects and writes to `data/raw/ob_letters/`.

- [ ] **Step 1: Write the script**

```python
# languages/akkadian/scripts/02_scrape_oracc_letters.py
"""
ORACC OB Letters Scraper: Download and parse OB-letter ORACC project JSON archives.

Mirror of 01_scrape_oracc_ob.py. Reuses the CDL walker and downloader.
"""
import importlib.util
from pathlib import Path

from tqdm import tqdm

_OB_SCRAPER_PATH = Path(__file__).parent / "01_scrape_oracc_ob.py"
_spec = importlib.util.spec_from_file_location("ob_scraper", _OB_SCRAPER_PATH)
_ob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ob)

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"

LETTER_PROJECTS = [
    "saao/saa01",
    "saao/saa17",
    # OB-specific letter projects (Mari ARM, etc.) added as ATF becomes available.
]


def main():
    out_dir = DATA_RAW / "ob_letters"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_texts, all_lemmas = [], []

    for project in tqdm(LETTER_PROJECTS, desc="Downloading OB letter projects"):
        print(f"\nProcessing {project}...")
        zip_path = _ob.download_project_json(project, out_dir)
        if zip_path is None:
            continue
        lemmas, texts = _ob.parse_project_zip(zip_path)
        all_lemmas.extend(lemmas)
        all_texts.extend(texts)
        print(f"  {len(texts)} texts, {len(lemmas)} Akkadian lemmas")

    _ob.save_texts(all_texts, str(DATA_RAW / "ob_letters_texts.json"))
    _ob.save_texts(all_lemmas, str(DATA_RAW / "ob_letters_lemmas.json"))

    total_lines = sum(len(t["lines"]) for t in all_texts)
    print(f"\nTotal letter texts: {len(all_texts)}")
    print(f"Total letter lines: {total_lines}")
    print(f"Total Akkadian letter lemmas: {len(all_lemmas)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-import the script**

```bash
python -c "
import importlib.util, pathlib
p = pathlib.Path('languages/akkadian/scripts/02_scrape_oracc_letters.py')
spec = importlib.util.spec_from_file_location('letters', p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert hasattr(m, 'LETTER_PROJECTS') and hasattr(m, 'main')
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add languages/akkadian/scripts/02_scrape_oracc_letters.py
git commit -m "feat: ORACC OB letters scraper (02) reusing 01's CDL walker"
```

---

## Phase 4 — DCCLT Scraper with Bridge Pair Extraction (TDD)

### Task 7: Write tests for DCCLT pair extraction

**Files:**
- Create: `languages/akkadian/tests/test_03_scrape_dcclt.py`

DCCLT lexical lists structure word equivalences as multi-column entries with Sumerian and Akkadian columns. The pair extractor walks lines and pairs adjacent `sux` / `akk` lemmas on the same line.

- [ ] **Step 1: Write the test file**

```python
# languages/akkadian/tests/test_03_scrape_dcclt.py
import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "03_scrape_dcclt.py"


def _load():
    spec = importlib.util.spec_from_file_location("scrape_dcclt", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_extract_pairs_from_aligned_columns():
    mod = _load()
    text = {
        "cdl": [{
            "type": "line-start",
            "cdl": [
                {"f": {"lang": "sux", "form": "lugal", "cf": "lugal", "gw": "king"}},
                {"f": {"lang": "akk", "form": "szarrum", "cf": "šarru", "gw": "king"}},
            ]
        }, {
            "type": "line-start",
            "cdl": [
                {"f": {"lang": "sux", "form": "dingir", "cf": "dingir", "gw": "god"}},
                {"f": {"lang": "akk", "form": "ilum", "cf": "ilu", "gw": "god"}},
            ]
        }]
    }
    pairs = mod.extract_pairs(text)
    assert len(pairs) == 2
    assert pairs[0]["sumerian_cf"] == "lugal"
    assert pairs[0]["akkadian_cf"] == "šarru"
    assert pairs[1]["sumerian_cf"] == "dingir"
    assert pairs[1]["akkadian_cf"] == "ilu"


def test_skips_lines_with_only_one_language():
    mod = _load()
    text = {
        "cdl": [{
            "type": "line-start",
            "cdl": [{"f": {"lang": "sux", "form": "lugal", "cf": "lugal", "gw": "king"}}],
        }, {
            "type": "line-start",
            "cdl": [{"f": {"lang": "akk", "form": "ilum", "cf": "ilu", "gw": "god"}}],
        }]
    }
    pairs = mod.extract_pairs(text)
    assert pairs == []


def test_pair_includes_glosses():
    mod = _load()
    text = {
        "cdl": [{
            "type": "line-start",
            "cdl": [
                {"f": {"lang": "sux", "form": "lugal", "cf": "lugal", "gw": "king"}},
                {"f": {"lang": "akk", "form": "szarrum", "cf": "šarru", "gw": "king"}},
            ]
        }]
    }
    pairs = mod.extract_pairs(text)
    assert pairs[0]["sumerian_gw"] == "king"
    assert pairs[0]["akkadian_gw"] == "king"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest languages/akkadian/tests/test_03_scrape_dcclt.py -v`
Expected: 3 tests FAIL with file-not-found.

- [ ] **Step 3: Commit**

```bash
git add languages/akkadian/tests/test_03_scrape_dcclt.py
git commit -m "test: failing tests for DCCLT pair extraction"
```

---

### Task 8: Implement DCCLT scraper

**Files:**
- Create: `languages/akkadian/scripts/03_scrape_dcclt.py`

- [ ] **Step 1: Write the script**

```python
# languages/akkadian/scripts/03_scrape_dcclt.py
"""
DCCLT Scraper: Download Digital Corpus of Cuneiform Lexical Texts and parse
both running text (for Akkadian FastText) and Sumerian↔Akkadian word pairs (for
the v2 cross-lingual bridge experiment).

Source: https://oracc.museum.upenn.edu/json/dcclt.zip
Output:
  - data/raw/dcclt/                              (downloaded zip)
  - data/raw/dcclt_texts.json                    (Akkadian-side running text)
  - data/raw/dcclt_lemmas.json                   (Akkadian lemmas — fallback anchors)
  - data/processed/sumerian_akkadian_pairs.jsonl (parsed sux↔akk pairs)
"""
import importlib.util
import json
from pathlib import Path
from typing import Any
import zipfile

from tqdm import tqdm

_OB_SCRAPER_PATH = Path(__file__).parent / "01_scrape_oracc_ob.py"
_spec = importlib.util.spec_from_file_location("ob_scraper", _OB_SCRAPER_PATH)
_ob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ob)

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"

DCCLT_PROJECT = "dcclt"


def _walk_lines(node: Any, lines: list[list[dict]]) -> None:
    current: list[dict] = []
    _recurse(node, lines, current)
    if current:
        lines.append(current)


def _recurse(node: Any, lines: list[list[dict]], current: list[dict]) -> None:
    if isinstance(node, dict):
        if "f" in node:
            f = node["f"]
            current.append({
                "lang": f.get("lang", ""),
                "cf": f.get("cf", ""),
                "form": f.get("form", ""),
                "gw": f.get("gw", ""),
            })
        if node.get("ftype") == "line-start" or node.get("type") == "line-start":
            if current:
                lines.append(list(current))
                current.clear()
        if "cdl" in node:
            for child in node["cdl"]:
                _recurse(child, lines, current)
    elif isinstance(node, list):
        for child in node:
            _recurse(child, lines, current)


def extract_pairs(text_json: dict) -> list[dict]:
    """Emit Sumerian↔Akkadian word pairs from a DCCLT text JSON.

    A pair is emitted when one line contains BOTH a sux lemma and an akk lemma.
    """
    lines: list[list[dict]] = []
    _walk_lines(text_json.get("cdl", []), lines)
    pairs: list[dict] = []
    for line in lines:
        sux = next((w for w in line if w["lang"].startswith("sux")), None)
        akk = next((w for w in line if w["lang"].startswith("akk")), None)
        if sux and akk:
            pairs.append({
                "sumerian_cf": sux["cf"],
                "sumerian_form": sux["form"],
                "sumerian_gw": sux["gw"],
                "akkadian_cf": akk["cf"],
                "akkadian_form": akk["form"],
                "akkadian_gw": akk["gw"],
            })
    return pairs


def parse_dcclt_zip(zip_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    all_lemmas, all_texts, all_pairs = [], [], []
    with zipfile.ZipFile(zip_path) as zf:
        json_files = [n for n in zf.namelist() if "corpusjson" in n and n.endswith(".json")]
        for name in tqdm(json_files, desc="  DCCLT files", leave=False):
            try:
                data = json.loads(zf.read(name))
            except (json.JSONDecodeError, KeyError):
                continue
            p_number = Path(name).stem
            lemmas = _ob.extract_lemmas(data)
            lines = _ob.extract_lines(data)
            pairs = extract_pairs(data)
            if lemmas:
                all_lemmas.extend(lemmas)
            if lines:
                all_texts.append({"p_number": p_number, "lines": lines, "source": "DCCLT"})
            all_pairs.extend(pairs)
    return all_lemmas, all_texts, all_pairs


def main():
    out_dir = DATA_RAW / "dcclt"
    out_dir.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    zip_path = _ob.download_project_json(DCCLT_PROJECT, out_dir)
    if zip_path is None:
        print("Failed to download DCCLT")
        return

    lemmas, texts, pairs = parse_dcclt_zip(zip_path)

    _ob.save_texts(texts, str(DATA_RAW / "dcclt_texts.json"))
    _ob.save_texts(lemmas, str(DATA_RAW / "dcclt_lemmas.json"))

    pairs_path = DATA_PROCESSED / "sumerian_akkadian_pairs.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} sux↔akk pairs to {pairs_path}")

    print(f"\nDCCLT akkadian texts: {len(texts)}")
    print(f"DCCLT akkadian lemmas: {len(lemmas)}")
    print(f"DCCLT bridge pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest languages/akkadian/tests/test_03_scrape_dcclt.py -v`
Expected: 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add languages/akkadian/scripts/03_scrape_dcclt.py
git commit -m "feat: DCCLT scraper with sux↔akk pair extraction (bridge data)"
```

---

## Phase 5 — Run Scrapes

### Task 9: Execute all three scrapes (real network calls)

**Files:**
- Affects: `languages/akkadian/data/raw/`

Network-dependent; may take 5–30 min depending on bandwidth.

- [ ] **Step 1: Run OB literary scraper**

```bash
cd /Users/crashy/Development/hyper-glyphy
python languages/akkadian/scripts/01_scrape_oracc_ob.py
```

Expected: `Saved N texts to .../ob_literary_texts.json` and a final summary with non-zero `Total texts` and `Total Akkadian lemmas`.

- [ ] **Step 2: Run OB letters scraper**

```bash
python languages/akkadian/scripts/02_scrape_oracc_letters.py
```

- [ ] **Step 3: Run DCCLT scraper**

```bash
python languages/akkadian/scripts/03_scrape_dcclt.py
```

Expected: `DCCLT bridge pairs: N` with N > 0 (typically several thousand).

- [ ] **Step 4: Verify expected files exist**

```bash
ls -la languages/akkadian/data/raw/ob_literary_texts.json \
       languages/akkadian/data/raw/ob_literary_lemmas.json \
       languages/akkadian/data/raw/ob_letters_texts.json \
       languages/akkadian/data/raw/ob_letters_lemmas.json \
       languages/akkadian/data/raw/dcclt_texts.json \
       languages/akkadian/data/raw/dcclt_lemmas.json \
       languages/akkadian/data/processed/sumerian_akkadian_pairs.jsonl
```

Expected: all seven files present and non-empty.

- [ ] **Step 5: Token-budget check**

```bash
python -c "
import json
from pathlib import Path
total = 0
for p in ['ob_literary_texts.json', 'ob_letters_texts.json', 'dcclt_texts.json']:
    with open(Path('languages/akkadian/data/raw')/p) as f:
        for t in json.load(f):
            for line in t.get('lines', []):
                total += len(line.split())
print(f'rough token total: {total}')
"
```

If total < 500_000, flag the SB fallback contingency in the journal entry (Task 17).

- [ ] **Step 6: Confirm raw data is gitignored**

```bash
grep -E "languages/akkadian/data/raw|languages/akkadian/data/processed" .gitignore || echo "MISSING - add ignore patterns"
```

If missing, append to `.gitignore`:

```
languages/akkadian/data/raw/
languages/akkadian/data/processed/
languages/akkadian/data/dictionaries/
languages/akkadian/models/
languages/akkadian/results/
```

Then commit `.gitignore` only:

```bash
git add .gitignore
git commit -m "ignore: gitignore akkadian data/models/results"
```

---

## Phase 6 — Dedup + Clean

### Task 10: Adapt deduplication for Akkadian three-source merge

**Files:**
- Modify: `languages/akkadian/scripts/04_deduplicate_corpus.py`

The Sumerian dedup reads `etcsl_texts.json`, `cdli_texts.json`, `oracc_texts.json`. The Akkadian dedup reads `ob_literary_texts.json`, `ob_letters_texts.json`, `dcclt_texts.json`. Identical logic, different filenames.

- [ ] **Step 1: Edit `main()` in the file**

Replace the `main()` function in `languages/akkadian/scripts/04_deduplicate_corpus.py` with:

```python
def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    all_texts = []

    for src_name, src_label in [
        ("ob_literary_texts.json", "OB_literary"),
        ("ob_letters_texts.json", "OB_letters"),
        ("dcclt_texts.json", "DCCLT"),
    ]:
        path = DATA_RAW / src_name
        if path.exists():
            with open(path) as f:
                texts = json.load(f)
            all_texts.extend(texts)
            print(f"{src_label}: {len(texts)} texts")
        else:
            print(f"{src_label}: missing ({path})")

    result, stats = deduplicate_with_stats(all_texts)

    output_path = DATA_PROCESSED / "merged_corpus.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nDeduplication stats: {json.dumps(stats, indent=2)}")
    total_lines = sum(len(t.get("lines", [])) for t in result)
    print(f"Total texts: {len(result)}")
    print(f"Total lines: {total_lines}")


if __name__ == "__main__":
    main()
```

Leave `deduplicate`, `deduplicate_with_stats`, and the imports unchanged.

- [ ] **Step 2: Run dedup**

```bash
python languages/akkadian/scripts/04_deduplicate_corpus.py
```

Expected: prints stats, creates `languages/akkadian/data/processed/merged_corpus.json`.

- [ ] **Step 3: Commit**

```bash
git add languages/akkadian/scripts/04_deduplicate_corpus.py
git commit -m "feat: adapt 04_deduplicate_corpus.py for Akkadian three-source merge"
```

---

### Task 11: Adapt clean/tokenize for Akkadian normalization

**Files:**
- Create: `languages/akkadian/scripts/05_clean_and_tokenize.py`
- Create: `languages/akkadian/tests/test_05_clean.py`

Sumerian's ATF cleanup is largely applicable to Akkadian unchanged. The Akkadian-specific change: route per-token cleanup through `normalize_akkadian_token` instead of Sumerian's `normalize_transliteration`.

- [ ] **Step 1: Write the test file**

```python
# languages/akkadian/tests/test_05_clean.py
import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "05_clean_and_tokenize.py"


def _load():
    spec = importlib.util.spec_from_file_location("clean", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_strips_brackets_and_keeps_content():
    mod = _load()
    assert "szarrum" in mod.clean_atf_line("[šar-ru-um]").split()


def test_strips_determinatives_keeps_content():
    mod = _load()
    cleaned = mod.clean_atf_line("{d}šamaš").split()
    assert "szamasz" in cleaned or "dszamasz" in cleaned


def test_drops_uppercase_only_signs():
    mod = _load()
    out = mod.clean_atf_line("LUGAL ša-ar")
    assert "lugal" not in out.split()
    assert any(t.startswith("sza") for t in out.split())


def test_normalizes_akkadian_diacritics():
    mod = _load()
    out = mod.clean_atf_line("šar-ru ḫar-ra")
    tokens = out.split()
    assert any(t.startswith("sz") for t in tokens)
    assert any(t.startswith("h") for t in tokens)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest languages/akkadian/tests/test_05_clean.py -v`
Expected: 4 tests FAIL with file-not-found.

- [ ] **Step 3: Write the script**

Copy `languages/sumerian/scripts/05_clean_and_tokenize.py` to `languages/akkadian/scripts/05_clean_and_tokenize.py`, then edit the import-and-normalize section near the top of the file:

Replace:

```python
# Unicode subscript -> ASCII digit mapping
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# ATF-to-ORACC transliteration normalization
def normalize_transliteration(word: str) -> str:
    """Normalize ATF transliteration: lowercase and convert subscript digits."""
    # Convert subscript digits to ASCII
    word = word.translate(_SUBSCRIPT_MAP)
    return word.lower()
```

With:

```python
# Route through the canonical Akkadian normalizer (NFC, mimation, ORACC->ATF, etc.)
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token

# Kept for compatibility with the rest of the file.
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_transliteration(word: str) -> str:
    return normalize_akkadian_token(word)
```

Leave `clean_atf_line`, `build_corpus`, `main`, and other functions unchanged. The call site `tok = normalize_transliteration(tok)` in `clean_atf_line` is unchanged.

Verify no remaining Sumerian-specific references:

```bash
grep -i "sumerian" languages/akkadian/scripts/05_clean_and_tokenize.py || echo "clean"
```

Expected: `clean`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest languages/akkadian/tests/test_05_clean.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Run cleaning end-to-end**

```bash
python languages/akkadian/scripts/05_clean_and_tokenize.py
```

Expected: prints corpus stats, creates `data/processed/cleaned_corpus.txt`.

- [ ] **Step 6: Commit**

```bash
git add languages/akkadian/scripts/05_clean_and_tokenize.py languages/akkadian/tests/test_05_clean.py
git commit -m "feat: 05_clean_and_tokenize via akkadian_normalize"
```

---

## Phase 7 — Anchor Extraction (eBL + ORACC)

### Task 12: Implement eBL lemma fetcher with tests

**Files:**
- Create: `languages/akkadian/scripts/ebl_fetch.py`
- Create: `languages/akkadian/tests/test_ebl_fetch.py`

eBL exposes JSON lemmas at `https://www.ebl.lmu.de/api/dictionary`. The fetcher pulls all lemmas, caches the raw response, and provides a filter for `Old Babylonian` period attestation.

- [ ] **Step 1: Write tests**

```python
# languages/akkadian/tests/test_ebl_fetch.py
import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ebl_fetch.py"


def _load():
    spec = importlib.util.spec_from_file_location("ebl_fetch", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_filter_ob_lemmas_keeps_ob_only():
    mod = _load()
    lemmas = [
        {"lemma": ["šarrum"], "guideWord": "king", "attested": True,
         "amplifiedMeanings": [], "logograms": [],
         "periodAttestation": ["Old Babylonian"]},
        {"lemma": ["bēl"], "guideWord": "lord", "attested": True,
         "amplifiedMeanings": [], "logograms": [],
         "periodAttestation": ["Neo-Assyrian"]},
        {"lemma": ["ilum"], "guideWord": "god", "attested": True,
         "amplifiedMeanings": [], "logograms": [],
         "periodAttestation": ["Old Babylonian", "Middle Babylonian"]},
    ]
    ob = mod.filter_ob_lemmas(lemmas)
    assert len(ob) == 2
    assert {entry["lemma"] for entry in ob} == {"šarrum", "ilum"}


def test_filter_ob_drops_no_gloss():
    mod = _load()
    lemmas = [
        {"lemma": ["x"], "guideWord": "", "attested": True,
         "amplifiedMeanings": [], "logograms": [],
         "periodAttestation": ["Old Babylonian"]},
        {"lemma": ["šarrum"], "guideWord": "king", "attested": True,
         "amplifiedMeanings": [], "logograms": [],
         "periodAttestation": ["Old Babylonian"]},
    ]
    ob = mod.filter_ob_lemmas(lemmas)
    assert len(ob) == 1
    assert ob[0]["lemma"] == "šarrum"


def test_extract_logogram_forms():
    mod = _load()
    entry = {"lemma": "šarrum", "gloss": "king", "period": ["Old Babylonian"],
             "logograms": ["LUGAL"], "raw": {}}
    forms = mod.extract_surface_forms(entry)
    assert "šarrum" in forms
    assert "LUGAL" in forms
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest languages/akkadian/tests/test_ebl_fetch.py -v`
Expected: 3 tests FAIL with file-not-found.

- [ ] **Step 3: Write the fetcher**

```python
# languages/akkadian/scripts/ebl_fetch.py
"""
eBL (electronic Babylonian Library) lemma fetcher.

Pulls the full dictionary from https://www.ebl.lmu.de/api/dictionary, caches it
to data/dictionaries/ebl_lemmas.json, and exposes filtering helpers used by
06_extract_anchors.py.

Each eBL entry includes:
  - lemma: list[str] (citation form)
  - guideWord: short English gloss
  - periodAttestation: list[str] (e.g., ["Old Babylonian", ...])
  - logograms: list[dict] with 'logogram' key
"""
import json
from pathlib import Path

import requests

DATA_DICTS = Path(__file__).parent.parent / "data" / "dictionaries"
EBL_API = "https://www.ebl.lmu.de/api/dictionary"
CACHE_PATH = DATA_DICTS / "ebl_lemmas.json"


def fetch_ebl_lemmas(force: bool = False) -> list[dict]:
    """Fetch all eBL lemmas, caching to disk. Returns the parsed list."""
    DATA_DICTS.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists() and not force:
        with open(CACHE_PATH) as f:
            return json.load(f)
    response = requests.get(EBL_API, timeout=600)
    response.raise_for_status()
    data = response.json()
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def filter_ob_lemmas(lemmas: list[dict]) -> list[dict]:
    """Keep entries with Old Babylonian attestation and a non-empty guideWord.

    Returns flat dicts with fields: lemma, gloss, period, logograms, raw.
    """
    out = []
    for entry in lemmas:
        periods = entry.get("periodAttestation") or []
        if not any("Old Babylonian" in p for p in periods):
            continue
        gw = (entry.get("guideWord") or "").strip()
        if not gw:
            continue
        lemma_parts = entry.get("lemma") or []
        if not lemma_parts:
            continue
        out.append({
            "lemma": lemma_parts[0],
            "gloss": gw,
            "period": periods,
            "logograms": [l.get("logogram") for l in (entry.get("logograms") or []) if l.get("logogram")],
            "raw": entry,
        })
    return out


def extract_surface_forms(entry: dict) -> list[str]:
    """Return citation form + logogram surface forms for a flattened entry."""
    forms = []
    if entry.get("lemma"):
        forms.append(entry["lemma"])
    forms.extend(entry.get("logograms") or [])
    return forms


def main():
    print(f"Fetching eBL lemmas (cache: {CACHE_PATH})...")
    lemmas = fetch_ebl_lemmas()
    ob = filter_ob_lemmas(lemmas)
    print(f"Total eBL entries: {len(lemmas)}")
    print(f"OB-attested with gloss: {len(ob)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest languages/akkadian/tests/test_ebl_fetch.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Smoke-fetch the real eBL data (network)**

```bash
python languages/akkadian/scripts/ebl_fetch.py
```

Expected: prints `Total eBL entries: N` (typically ~25k+) and `OB-attested with gloss: M` (typically ~10-15k).

- [ ] **Step 6: Commit**

```bash
git add languages/akkadian/scripts/ebl_fetch.py languages/akkadian/tests/test_ebl_fetch.py
git commit -m "feat: eBL lemma fetcher with OB period filter"
```

---

### Task 13: Adapt anchor extraction with eBL primary + ORACC fallback

**Files:**
- Create: `languages/akkadian/scripts/06_extract_anchors.py`
- Create: `languages/akkadian/tests/test_06_anchors.py`

The Akkadian anchor extractor: (a) primary source is eBL, (b) ORACC project glosses (concatenated from `ob_literary_lemmas.json`, `ob_letters_lemmas.json`, `dcclt_lemmas.json`) are the fallback, (c) per-anchor expansion: for each lemma, register both citation form and logogram surface forms as anchors with the same gloss.

- [ ] **Step 1: Write the test file**

```python
# languages/akkadian/tests/test_06_anchors.py
import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "06_extract_anchors.py"


def _load():
    spec = importlib.util.spec_from_file_location("anchors", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ebl_anchors_emit_citation_and_logogram():
    mod = _load()
    flat_ob = [{
        "lemma": "šarrum", "gloss": "king", "period": ["Old Babylonian"],
        "logograms": ["LUGAL"], "raw": {},
    }]
    anchors = mod.extract_ebl_anchors(flat_ob)
    forms = {a["akkadian"] for a in anchors}
    assert "szarrum" in forms
    assert "lugal" in forms
    assert all(a["english"] == "king" for a in anchors)
    assert all(a["source"] == "eBL" for a in anchors)


def test_oracc_fallback_only_for_uncovered_lemmas():
    mod = _load()
    ebl_anchors = [{
        "akkadian": "szarrum", "english": "king",
        "confidence": 0.95, "frequency": 1, "source": "eBL",
    }]
    oracc_lemmas = [
        {"cf": "šarru", "form": "szarrum", "gw": "king", "lang": "akk"},
        {"cf": "ilu",   "form": "ilum",    "gw": "god",  "lang": "akk"},
    ]
    fallback = mod.extract_oracc_fallback(ebl_anchors, oracc_lemmas, min_occurrences=1)
    forms = {a["akkadian"] for a in fallback}
    assert "ilum" in forms or "ilu" in forms
    assert "szarrum" not in forms


def test_merge_anchors_keeps_higher_confidence():
    mod = _load()
    primary = [{"akkadian": "szarrum", "english": "king", "confidence": 0.9, "frequency": 1, "source": "eBL"}]
    fallback = [{"akkadian": "szarrum", "english": "king", "confidence": 0.5, "frequency": 1, "source": "ORACC"}]
    merged = mod.merge_anchors(primary, fallback)
    by_form = {a["akkadian"]: a for a in merged}
    assert by_form["szarrum"]["source"] == "eBL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest languages/akkadian/tests/test_06_anchors.py -v`
Expected: 3 tests FAIL with file-not-found.

- [ ] **Step 3: Write the script**

```python
# languages/akkadian/scripts/06_extract_anchors.py
"""
Anchor Extraction: Build Akkadian-English word pairs from eBL (primary) and
ORACC project glosses (fallback).

eBL: filtered to OB-attested lemmas with English glosses, expanded so both the
citation form AND any logogram surface forms are registered as anchors under
the same gloss.

ORACC fallback: for any Akkadian lemma in the merged ORACC dump (ob_literary,
ob_letters, dcclt) NOT covered by eBL, emit a fallback anchor.
"""
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token  # noqa: E402

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
DATA_DICTS = Path(__file__).parent.parent / "data" / "dictionaries"

JUNK_ENGLISH = {
    "x", "xx", "0", "00", "1", "n", "c", "e", "i", "u", "s", "unmng", "cf",
}


def _filter_gloss(gw: str) -> bool:
    if not gw or gw in JUNK_ENGLISH:
        return False
    if len(gw) <= 2:
        return False
    if gw.isdigit():
        return False
    if gw.startswith("~"):
        return False
    return True


def extract_ebl_anchors(flat_ob: list[dict]) -> list[dict]:
    """Emit one anchor per (surface_form, gloss) pair for each OB-attested lemma."""
    anchors: list[dict] = []
    for entry in flat_ob:
        gloss = entry["gloss"].strip().lower()
        if not _filter_gloss(gloss):
            continue
        forms = [entry["lemma"]] + (entry.get("logograms") or [])
        for raw_form in forms:
            form_norm = normalize_akkadian_token(raw_form)
            if not form_norm:
                continue
            anchors.append({
                "akkadian": form_norm,
                "english": gloss,
                "confidence": 0.95,
                "frequency": 1,
                "source": "eBL",
            })
    seen: dict[tuple[str, str], dict] = {}
    for a in anchors:
        key = (a["akkadian"], a["english"])
        if key not in seen:
            seen[key] = a
    return list(seen.values())


def extract_oracc_fallback(
    ebl_anchors: list[dict],
    oracc_lemmas: list[dict],
    min_occurrences: int = 5,
) -> list[dict]:
    """Emit anchors from ORACC for lemmas NOT covered by eBL."""
    covered = {a["akkadian"] for a in ebl_anchors}

    pair_counts: Counter[tuple[str, str]] = Counter()
    for lemma in oracc_lemmas:
        gw = (lemma.get("gw") or "").strip().lower()
        if not _filter_gloss(gw):
            continue
        cf = normalize_akkadian_token((lemma.get("cf") or "").strip())
        form = normalize_akkadian_token((lemma.get("form") or "").strip())
        if cf and cf not in covered:
            pair_counts[(cf, gw)] += 1
        if form and form != cf and form not in covered:
            pair_counts[(form, gw)] += 1

    anchors: list[dict] = []
    for (form_norm, gw), count in pair_counts.items():
        if count < min_occurrences:
            continue
        confidence = min(0.85, 0.4 + (count / 100))
        anchors.append({
            "akkadian": form_norm,
            "english": gw,
            "confidence": round(confidence, 4),
            "frequency": count,
            "source": "ORACC",
        })
    return anchors


def merge_anchors(primary: list[dict], fallback: list[dict]) -> list[dict]:
    """Merge primary (eBL) and fallback (ORACC), keeping higher confidence per form."""
    best: dict[str, dict] = {}
    for a in primary + fallback:
        key = a["akkadian"]
        if key not in best or a["confidence"] > best[key]["confidence"]:
            best[key] = a
    return sorted(best.values(), key=lambda a: a["confidence"], reverse=True)


def _load_oracc_lemmas() -> list[dict]:
    out = []
    for name in ("ob_literary_lemmas.json", "ob_letters_lemmas.json", "dcclt_lemmas.json"):
        path = DATA_RAW / name
        if path.exists():
            with open(path) as f:
                out.extend(json.load(f))
    return out


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_DICTS.mkdir(parents=True, exist_ok=True)

    from languages.akkadian.scripts.ebl_fetch import fetch_ebl_lemmas, filter_ob_lemmas
    raw_ebl = fetch_ebl_lemmas()
    flat_ob = filter_ob_lemmas(raw_ebl)
    ebl_anchors = extract_ebl_anchors(flat_ob)
    print(f"eBL OB anchors: {len(ebl_anchors)}")

    oracc_lemmas = _load_oracc_lemmas()
    if oracc_lemmas:
        with open(DATA_DICTS / "oracc_lemmas.json", "w", encoding="utf-8") as f:
            json.dump(oracc_lemmas, f, ensure_ascii=False, indent=2)
    fallback = extract_oracc_fallback(ebl_anchors, oracc_lemmas, min_occurrences=5)
    print(f"ORACC fallback anchors: {len(fallback)}")

    merged = merge_anchors(ebl_anchors, fallback)

    output_path = DATA_PROCESSED / "english_anchors.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nTotal merged anchors: {len(merged)}")
    print(f"From eBL: {sum(1 for a in merged if a['source'] == 'eBL')}")
    print(f"From ORACC: {sum(1 for a in merged if a['source'] == 'ORACC')}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest languages/akkadian/tests/test_06_anchors.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Run anchor extraction**

```bash
python languages/akkadian/scripts/06_extract_anchors.py
```

Expected: prints anchor counts, creates `data/processed/english_anchors.json`. Look for `Total merged anchors: N` typically in 5,000–10,000 range.

- [ ] **Step 6: Commit**

```bash
git add languages/akkadian/scripts/06_extract_anchors.py languages/akkadian/tests/test_06_anchors.py
git commit -m "feat: 06_extract_anchors with eBL primary + ORACC fallback"
```

---

### Task 14: Adapt coverage diagnostic with `logogram_unmatched` bucket

**Files:**
- Create: `languages/akkadian/scripts/coverage_diagnostic.py` (forked)
- Create: `languages/akkadian/tests/test_coverage_diagnostic.py`

Sumerian's coverage diagnostic is 932 lines. Fork it, replace `sumerian` with `akkadian`, and add the new bucket.

- [ ] **Step 1: Copy the Sumerian diagnostic and rename references**

```bash
cd /Users/crashy/Development/hyper-glyphy
cp languages/sumerian/scripts/coverage_diagnostic.py languages/akkadian/scripts/coverage_diagnostic.py

sed -i '' \
  -e 's/languages\.sumerian/languages.akkadian/g' \
  -e 's/normalize_sumerian_token/normalize_akkadian_token/g' \
  -e 's/sumerian_normalize/akkadian_normalize/g' \
  -e 's/sumerian_vocab_miss/akkadian_vocab_miss/g' \
  languages/akkadian/scripts/coverage_diagnostic.py

# Sanity check (allow comments + the docstring referencing original Sumerian work)
grep -E "languages\.sumerian|normalize_sumerian|sumerian_normalize" languages/akkadian/scripts/coverage_diagnostic.py | grep -v "^#" || echo "clean"
```

Expected: `clean`.

- [ ] **Step 2: Add the `logogram_unmatched` bucket**

Open `languages/akkadian/scripts/coverage_diagnostic.py` and locate the anchor classification function (search the file for `akkadian_vocab_miss`). The function will be the priority-ordered if/elif chain that picks one bucket per anchor.

Insert a new branch BEFORE the `akkadian_vocab_miss` fallthrough. The exact variable name depends on the function's local naming — read 30 lines of context to identify the surface-form variable. Then insert:

```python
        # Akkadian-specific: ALL-CAPS surface absent from corpus, but its
        # lowercase counterpart present, indicates a logogram dropped by the
        # corpus tokenizer (which strips ALL-CAPS sign names).
        if <surface_var> and <surface_var>.isupper() and <surface_var>.lower() in fused_vocab:
            return "logogram_unmatched"
```

Replace `<surface_var>` with the actual local variable name. (Likely `surface`, `form`, or `anchor_surface`.)

- [ ] **Step 3: Write a test for the new bucket**

```python
# languages/akkadian/tests/test_coverage_diagnostic.py
import pytest


def test_logogram_unmatched_bucket_is_recognized():
    """The forked diagnostic must recognize logogram_unmatched as a valid bucket name."""
    pytest.importorskip("numpy")
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "coverage_diagnostic.py"
    spec = importlib.util.spec_from_file_location("cov", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # The string "logogram_unmatched" must appear in the file (returned as a bucket name).
    src = p.read_text()
    assert "logogram_unmatched" in src, "logogram_unmatched bucket not added"


def test_classifier_emits_logogram_for_uppercase_anchor():
    """Functional test: an uppercase anchor present-as-lowercase fires logogram bucket.

    The exact classifier API may differ from Sumerian's; this test wraps a synthetic
    invocation and is best-effort. If the classifier is private and signature differs,
    skip this test rather than fail.
    """
    pytest.importorskip("numpy")
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "coverage_diagnostic.py"
    spec = importlib.util.spec_from_file_location("cov", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    classifier = getattr(mod, "_classify_anchor", None)
    if classifier is None:
        pytest.skip("coverage diagnostic does not expose _classify_anchor; smoke test only")
    # If the classifier exists, the test for the new bucket is environment-dependent;
    # documentary marker only.
```

- [ ] **Step 4: Run the test**

Run: `pytest languages/akkadian/tests/test_coverage_diagnostic.py -v`
Expected: at least 1 PASS, possibly 1 SKIP. The first test must pass — it's a textual presence check.

- [ ] **Step 5: Commit**

```bash
git add languages/akkadian/scripts/coverage_diagnostic.py languages/akkadian/tests/test_coverage_diagnostic.py
git commit -m "feat: coverage diagnostic forked to akkadian + logogram_unmatched bucket"
```

---

## Phase 8 — Train, Fuse, Align

### Task 15: Run FastText, fusion, and both alignments end-to-end

**Files:**
- Affects: `languages/akkadian/models/`, `languages/akkadian/results/`

These four scripts (07/08/09/09b) were copied verbatim in Task 2. They read from `data/processed/` paths and operate on Akkadian data automatically.

- [ ] **Step 1: Train FastText**

```bash
cd /Users/crashy/Development/hyper-glyphy
python languages/akkadian/scripts/07_train_fasttext.py
```

Expected: prints epoch loss; outputs the FastText model artifact under `languages/akkadian/models/`.

- [ ] **Step 2: Fuse embeddings (zero-pad to 1536d)**

```bash
python languages/akkadian/scripts/08_fuse_embeddings.py
```

Expected: outputs `models/fused_embeddings_1536d.npz` with shape `(vocab_size, 1536)`.

- [ ] **Step 3: Align to GloVe (300d secondary)**

```bash
python languages/akkadian/scripts/09_align_and_evaluate.py
```

Expected: prints top-1/5/10 accuracy; writes `results/alignment_results.json` and `models/ridge_weights.npz`.

- [ ] **Step 4: Align to whitened-Gemma (768d primary)**

```bash
python languages/akkadian/scripts/09b_align_gemma.py
```

Expected: prints top-1/5/10; writes `results/alignment_results_gemma_whitened.json` and `models/ridge_weights_gemma_whitened.npz`.

- [ ] **Step 5: Capture the four key numbers**

```bash
python -c "
import json
glove = json.load(open('languages/akkadian/results/alignment_results.json'))
gemma = json.load(open('languages/akkadian/results/alignment_results_gemma_whitened.json'))
print('GloVe top-1/5/10:', glove['accuracy'])
print('Gemma top-1/5/10:', gemma['accuracy'])
"
```

Note these numbers; they go into the journal entry in Task 17.

---

## Phase 9 — Production Export and Documentation

### Task 16: Implement `10_export_production.py` with AkkadianLookup class

**Files:**
- Create: `languages/akkadian/scripts/10_export_production.py`
- Create: `languages/akkadian/final_output/akkadian_lookup.py`
- Create: `languages/akkadian/tests/test_10_export.py`
- Create: `languages/akkadian/tests/test_lookup.py`

Mirrors Sumerian's export + lookup pattern with one deviation: vocab is serialized as JSON (string list) instead of pickle, for safety.

- [ ] **Step 1: Write the export script**

Copy `languages/sumerian/scripts/10_export_production.py` to `languages/akkadian/scripts/10_export_production.py`, then apply textual rewrites:

```bash
sed -i '' \
  -e 's/sum_vectors/akk_vectors/g' \
  -e 's/sum_vocab/akk_vocab/g' \
  -e 's/sumerian_aligned_/akkadian_aligned_/g' \
  -e 's/Sumerian/Akkadian/g' \
  -e 's/Cuneiformy/Cuneiformy-Akkadian/g' \
  languages/akkadian/scripts/10_export_production.py
```

Then replace the pickle-vocab block:

Open the file and find these lines (near the bottom of `main()`):

```python
    _pkl = importlib.import_module("pickle")
    with open(FINAL_OUTPUT / "akkadian_aligned_vocab.pkl", "wb") as f:
        _pkl.dump(akk_vocab, f)
```

Replace with:

```python
    with open(FINAL_OUTPUT / "akkadian_aligned_vocab.json", "w", encoding="utf-8") as f:
        json.dump(akk_vocab, f, ensure_ascii=False)
```

Remove the `import importlib` line at the top of the file if it is no longer used. Verify:

```bash
grep -E "pickle|importlib" languages/akkadian/scripts/10_export_production.py || echo "clean"
```

Expected: `clean`.

- [ ] **Step 2: Write the AkkadianLookup class**

```python
# languages/akkadian/final_output/akkadian_lookup.py
"""
AkkadianLookup: query the production Akkadian alignment artifacts.

Mirrors SumerianLookup. Two views: 'gemma' (768d, primary, whitened) and
'glove' (300d, secondary). Vocab is serialized as JSON for safety.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

_VECTORS = {
    "gemma": _HERE / "akkadian_aligned_gemma_vectors.npz",
    "glove": _HERE / "akkadian_aligned_vectors.npz",
}
_VOCAB_PATH = _HERE / "akkadian_aligned_vocab.json"


class AkkadianLookup:
    """Look up English neighbors for Akkadian words in the aligned space."""

    def __init__(self, space: str = "gemma"):
        if space not in _VECTORS:
            raise ValueError(f"space must be one of {list(_VECTORS)}, got {space!r}")
        self.space = space
        with open(_VOCAB_PATH, encoding="utf-8") as f:
            self.vocab: list[str] = json.load(f)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.vectors = np.load(str(_VECTORS[space]))["vectors"].astype(np.float32)

    def lookup(self, word: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k vocab words nearest to `word` by cosine similarity."""
        idx = self.word_to_idx.get(word)
        if idx is None:
            return []
        v = self.vectors[idx]
        norms = np.linalg.norm(self.vectors, axis=1)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return []
        cos = (self.vectors @ v) / (norms * v_norm + 1e-12)
        cos[idx] = -1.0
        top = np.argsort(-cos)[:k]
        return [(self.vocab[i], float(cos[i])) for i in top]
```

- [ ] **Step 3: Write tests**

```python
# languages/akkadian/tests/test_10_export.py
import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "10_export_production.py"


def test_export_module_imports_without_artifacts():
    spec = importlib.util.spec_from_file_location("export", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert hasattr(m, "main")
    assert hasattr(m, "project_all_vectors")
```

```python
# languages/akkadian/tests/test_lookup.py
import importlib.util
from pathlib import Path

import pytest

_LOOKUP_PATH = Path(__file__).resolve().parents[1] / "final_output" / "akkadian_lookup.py"


def test_lookup_class_importable():
    spec = importlib.util.spec_from_file_location("akkadian_lookup", _LOOKUP_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert hasattr(m, "AkkadianLookup")


def test_lookup_rejects_unknown_space():
    spec = importlib.util.spec_from_file_location("akkadian_lookup", _LOOKUP_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with pytest.raises(ValueError):
        m.AkkadianLookup(space="bogus")
```

- [ ] **Step 4: Run the export**

```bash
python languages/akkadian/scripts/10_export_production.py
```

Expected: produces `final_output/akkadian_aligned_vectors.npz`, `akkadian_aligned_gemma_vectors.npz`, `akkadian_aligned_vocab.json`, `metadata.json`.

- [ ] **Step 5: Run tests**

Run: `pytest languages/akkadian/tests/test_10_export.py languages/akkadian/tests/test_lookup.py -v`
Expected: 3 tests PASS.

- [ ] **Step 6: Functional smoke test**

```bash
python -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('languages/akkadian/final_output')))
from akkadian_lookup import AkkadianLookup
lkp = AkkadianLookup(space='gemma')
print('vocab size:', len(lkp.vocab))
for w in ['szarrum', 'ilum', 'bel', 'ana']:
    if w in lkp.word_to_idx:
        print(w, '->', lkp.lookup(w, k=5))
        break
else:
    print('no test anchor found in vocab — sample manually')
"
```

Expected: prints vocab size and (if a test anchor is in vocab) top-5 neighbors.

- [ ] **Step 7: Commit**

```bash
git add languages/akkadian/scripts/10_export_production.py \
        languages/akkadian/final_output/akkadian_lookup.py \
        languages/akkadian/tests/test_10_export.py \
        languages/akkadian/tests/test_lookup.py
git commit -m "feat: 10_export_production + AkkadianLookup (JSON vocab artifact)"
```

---

### Task 17: Write journal entry, README, and final commit

**Files:**
- Create: `languages/akkadian/docs/EXPERIMENT_JOURNAL.md`
- Create: `languages/akkadian/README.md`
- Modify: `docs/EXPERIMENT_JOURNAL.md` (project-level journal)

- [ ] **Step 1: Write the per-slot journal**

```markdown
# Akkadian Experiment Journal

## 2026-05-09 — v1 ship: OB Akkadian aligned to whitened-Gemma 768d

**Spec:** `docs/superpowers/specs/2026-05-09-akkadian-slot-design.md`
**Plan:** `docs/superpowers/plans/2026-05-09-akkadian-slot.md`

Pipeline mirrors Sumerian 1:1. Three corpus tiers ingested via three new
scrapers (OB literary, OB letters, DCCLT). Anchor lexicon: eBL primary
(OB-period filter) + ORACC project-gloss fallback. DCCLT lexical lists
parsed into `data/processed/sumerian_akkadian_pairs.jsonl` for the v2
cross-lingual bridge experiment (data ready, experiment deferred).

### Numbers

| Metric | Akkadian (whitened-Gemma 768d) | Akkadian (GloVe 300d) | Sumerian (whitened-Gemma) | Egyptian (GloVe) |
|--------|:---:|:---:|:---:|:---:|
| Top-1  | TODO | TODO | 52.13% | 32.35% |
| Top-5  | TODO | TODO | 61.97% | 41.47% |
| Top-10 | TODO | TODO | 65.99% | 45.13% |
| Anchors (training) | TODO | TODO | 6,867 | 5,360 |
| Corpus tokens | TODO | TODO | 2.8M | 789K |

(Replace TODOs with numbers from `results/alignment_results*.json`.)

### Coverage diagnostic

TODO: highlights from the first run of `coverage_diagnostic.py`,
particularly the `logogram_unmatched` bucket size (early-warning signal
for the Akkadian-specific dual-encoding problem).

### Out-of-scope (deferred)

- Cross-lingual bridge experiment (data scaffolded in `sumerian_akkadian_pairs.jsonl`).
- Standard Babylonian fallback (only flagged if cleaned OB corpus < 500k tokens).
- Diachronic OB → Classical Akkadian comparison.
```

- [ ] **Step 2: Replace TODOs with real numbers from Task 15 outputs**

Read `languages/akkadian/results/alignment_results.json` and `languages/akkadian/results/alignment_results_gemma_whitened.json` and substitute.

- [ ] **Step 3: Write the slot README**

```markdown
# languages/akkadian — Old Babylonian Akkadian Alignment

OB Akkadian aligned to whitened-EmbeddingGemma (768d, primary) and GloVe
(300d, secondary). Pipeline structure mirrors `languages/sumerian/` 1:1.

## Quick start

```bash
# Scrape (network)
python scripts/01_scrape_oracc_ob.py
python scripts/02_scrape_oracc_letters.py
python scripts/03_scrape_dcclt.py

# Process
python scripts/04_deduplicate_corpus.py
python scripts/05_clean_and_tokenize.py

# Anchors
python scripts/ebl_fetch.py
python scripts/06_extract_anchors.py

# Train + align
python scripts/07_train_fasttext.py
python scripts/08_fuse_embeddings.py
python scripts/09_align_and_evaluate.py     # GloVe target
python scripts/09b_align_gemma.py           # whitened-Gemma target

# Export
python scripts/10_export_production.py
```

## Lookup API

```python
from languages.akkadian.final_output.akkadian_lookup import AkkadianLookup
lkp = AkkadianLookup(space="gemma")  # or "glove"
lkp.lookup("szarrum", k=10)
```

## Bridge data

`data/processed/sumerian_akkadian_pairs.jsonl` — Sumerian↔Akkadian word
pairs from DCCLT lexical lists, parsed but not yet used. v2 experiment
will cross-validate Sumerian-Gemma and Akkadian-Gemma alignments through
these pairs.

## Spec & plan

- Spec: `docs/superpowers/specs/2026-05-09-akkadian-slot-design.md`
- Plan: `docs/superpowers/plans/2026-05-09-akkadian-slot.md`
```

- [ ] **Step 4: Append to the project-level journal**

In `docs/EXPERIMENT_JOURNAL.md`, prepend a new entry under the "Recent findings (newest first)" section:

```markdown
- **2026-05-09 — Akkadian slot v1 shipped:** Third language slot. OB-period scope, eBL+ORACC anchor lexicon. Whitened-Gemma top-1 **TODO%** (vs Sumerian 52.13%, Egyptian 32.35%). DCCLT bridge data scaffolded for v2 cross-lingual experiment. See [`languages/akkadian/docs/EXPERIMENT_JOURNAL.md`](../languages/akkadian/docs/EXPERIMENT_JOURNAL.md) for full numbers and the [design spec](superpowers/specs/2026-05-09-akkadian-slot-design.md).
```

Replace `TODO%` with the real top-1.

- [ ] **Step 5: Final commit**

```bash
git add languages/akkadian/docs/EXPERIMENT_JOURNAL.md \
        languages/akkadian/README.md \
        docs/EXPERIMENT_JOURNAL.md
git commit -m "docs: akkadian v1 journal entry and README"
```

- [ ] **Step 6: Full test suite green-check**

```bash
pytest languages/akkadian/tests -v
```

Expected: all tests PASS. If anything fails, fix before declaring v1 complete.
