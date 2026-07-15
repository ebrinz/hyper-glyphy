# Sanskrit Sixth Slot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Sanskrit as the sixth language slot (DCS corpus + Monier-Williams anchors) on the proven pipeline, and run the pre-registered Procrustes anchor-quality read-out.

**Architecture:** Three new parse/normalize files (`01_parse_dcs.py`, `02_parse_mw.py`, `sanskrit_normalize.py`) plus a thin new `05_clean_and_tokenize.py`; `04`, `07`–`10`, `align_09.py` are deterministic sed-clones of the Greek canonicals verified by diff-gates; `06_extract_anchors.py` is a sed-clone plus three enumerated edits (negation-gloss rule, hit-rate gate, stats persistence). FastText trains on the sandhi-resolved FORM stream; anchors are keyed on gold DCS lemmas joined to MW headwords.

**Tech Stack:** Python 3.12, xml.etree, gensim FastText, scikit-learn Ridge, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-07-13-sanskrit-slot-design.md`

## Global Constraints

- Branch `sanskrit-slot`. All commands run from the repo root. Full `pytest` green before every commit.
- Everything under `languages/sanskrit/data/`, `models/`, `results/`, and `final_output/*.npz` is gitignored (existing repo-root patterns) — never commit data or models.
- Canonical token form: **lowercase IAST, Unicode NFC** (`normalize_sanskrit_token`). Diacritics are PRESERVED (unlike `greek_normalize`, which strips combining marks — do not reuse it).
- FastText trains on the **inflected FORM stream** (conllu column 2, sandhi-resolved), NOT the lemma stream and NOT `Unsandhied`. Anchors keyed on lemma (column 3).
- `MIN_OCCURRENCES = 5` (anchor frequency floor, Greek value). FastText: 768d, window 10, min_count 2, epochs 10, sg 1 (unchanged in the clone).
- **Gloss hit-rate gate:** `MIN_HIT_RATE = 0.40` on the token-level DCS-lemma→MW join. Below gate = `SystemExit` after persisting `anchor_stats.json` — stop-and-surface, never a silent continue.
- **Approved deviation from the Greek 06 recipe (survey finding A1, user-approved 2026-07-14):** a gloss containing a negator (`not`, `no`, `without`, `never`) before its first in-vocab content word is skipped entirely (fall through to the entry's next gloss) instead of harvesting the negated word ("not injuring anything" must NOT become anchor "injuring"). Must be disclosed in the journal entry.
- Cloned scripts must be byte-equal to their sed derivation (diff-gate) and contain no `greek`/`sumerian` identifiers (grep-gate). No other edits to clones.
- **Pre-registered read-out bands (interpretation only — the slot ships on the word-level suite regardless):** Procrustes val cosine ≥ 0.20 ⇒ anchors were binding, lever stays live; ≤ 0.12 ⇒ retire the stronger-anchors lever and Plane A; between ⇒ inconclusive, stated as such. No threshold adjustment after measurement.
- Alignment runs take ~1–4 h per target on this host: run 07/09/09b detached (`nohup ... &`), never foreground.
- Data sources (pinned 2026-07-14): DCS = `https://github.com/OliverHellwig/sanskrit.git` (CC BY 4.0), conllu under `dcs/data/conllu/files/` (one folder per text, one file per chapter). MW = `https://www.sanskrit-lexicon.uni-koeln.de/scans/MWScan/2020/downloads/mwxml.zip` (11.7 MB, verified live; contains `xml/mw.xml`, 67 MB, SLP1 `key1` headwords).

---

### Task 1: Slot scaffold + `sanskrit_normalize.py`

**Files:**
- Create: `languages/sanskrit/__init__.py`, `languages/sanskrit/scripts/__init__.py`, `languages/sanskrit/tests/__init__.py` (all empty)
- Create: `languages/sanskrit/scripts/sanskrit_normalize.py`
- Test: `languages/sanskrit/tests/test_normalize.py`

**Interfaces:**
- Produces: `normalize_sanskrit_token(raw) -> str` — imported by 02, 05, 06 as `from languages.sanskrit.scripts.sanskrit_normalize import normalize_sanskrit_token`.

- [ ] **Step 1: Create the package dirs and empty `__init__.py` files**

```bash
mkdir -p languages/sanskrit/scripts languages/sanskrit/tests/fixtures
touch languages/sanskrit/__init__.py languages/sanskrit/scripts/__init__.py languages/sanskrit/tests/__init__.py
```

- [ ] **Step 2: Write the failing tests**

`languages/sanskrit/tests/test_normalize.py`:

```python
import unicodedata

from languages.sanskrit.scripts.sanskrit_normalize import normalize_sanskrit_token


def test_lowercase_and_strip():
    assert normalize_sanskrit_token("  Agni ") == "agni"


def test_nfc_composition():
    # "a" + combining macron (U+0304) must compose to precomposed ā (U+0101)
    decomposed = "ātman"
    out = normalize_sanskrit_token(decomposed)
    assert out == "ātman"
    assert unicodedata.is_normalized("NFC", out)


def test_diacritics_preserved():
    # Unlike greek_normalize, macrons and dots must survive
    assert normalize_sanskrit_token("ṛṣi") == "ṛṣi"
    assert normalize_sanskrit_token("kṛṣṇa") == "kṛṣṇa"
    assert normalize_sanskrit_token("ahiṃsā") == "ahiṃsā"


def test_idempotent():
    once = normalize_sanskrit_token("Āsīt")
    assert normalize_sanskrit_token(once) == once


def test_none_and_empty():
    assert normalize_sanskrit_token(None) == ""
    assert normalize_sanskrit_token("") == ""
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest languages/sanskrit/tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (sanskrit_normalize does not exist).

- [ ] **Step 4: Write the implementation**

`languages/sanskrit/scripts/sanskrit_normalize.py`:

```python
"""
Canonical Sanskrit (IAST) token normalization.

Single source of truth for mapping DCS conllu forms/lemmas and MW headwords
(after SLP1->IAST conversion in 02) to a common token form.

Normalization choices:
  - Unicode NFC (compose base + combining marks into precomposed chars where
    they exist; candrabindu m̐ has no precomposed form and stays combining).
  - Lowercase.
  - Strip surrounding whitespace.
  - Diacritics are PRESERVED: ā/ī/ū/ṛ/ṝ/ḷ/ḹ/ṃ/ḥ/ṅ/ñ/ṭ/ḍ/ṇ/ś/ṣ are
    phonemic in IAST. Do NOT reuse greek_normalize here — it drops macron
    (U+0304) and breve (U+0306), which would collapse ā->a, ī->i, ū->u.

Used by scripts/02_parse_mw.py, scripts/05_clean_and_tokenize.py, and
scripts/06_extract_anchors.py.
"""
from __future__ import annotations

import unicodedata


def normalize_sanskrit_token(raw) -> str:
    """Canonical normalization for a single Sanskrit (IAST) token.

    Order: NFC -> lowercase -> strip. Safe on None/empty (returns "").
    Idempotent.
    """
    if raw is None:
        return ""
    s = unicodedata.normalize("NFC", str(raw))
    return s.lower().strip()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest languages/sanskrit/tests/test_normalize.py -v`
Expected: 5 passed.

- [ ] **Step 6: Full pytest + commit**

```bash
pytest -q
git add languages/sanskrit/
git commit -m "feat(sanskrit): slot scaffold + IAST normalizer"
```

---

### Task 2: `01_parse_dcs.py` — DCS conllu → texts + lemmas JSON

**Files:**
- Create: `languages/sanskrit/scripts/01_parse_dcs.py`
- Create: `languages/sanskrit/tests/fixtures/sample.conllu`
- Test: `languages/sanskrit/tests/test_01_dcs.py`

**Interfaces:**
- Produces: `data/raw/sanskrit_texts.json` = `[{p_number: "dcs-<text_id>-<chapter_id>", text_name, chapter, lines: [str], source: "DCS"}]`; `data/raw/sanskrit_lemmas.json` = `[{form, cf, gw: "", pos, lang: "san"}]`. `parse_file(path: Path) -> dict` with keys `text, text_id, chapter, chapter_id, lines, lemmas, token_lines, bad_lines`.
- Consumed by: 04 (texts), 06 (lemmas).

- [ ] **Step 1: Write the fixture**

`languages/sanskrit/tests/fixtures/sample.conllu` (tab-separated token lines — ensure real tabs):

```
## text: Aitareyopaniṣad
## text_id: 421
## chapter: AU, 1, 1
## chapter_id: 8816
# text = ātmā vai idam ekaḥ eva agre āsīt
# sent_id = 556276_1
1	ātmā	ātman	NOUN	_	Case=Nom|Gender=Masc|Number=Sing	0	root	_	LemmaId=57749|OccId=3542461|Unsandhied=ātmā
2	vai	vai	PART	_	_	1	discourse	_	LemmaId=118154|OccId=5298231|Unsandhied=vai
3	idam	idam	PRON	_	Case=Nom|Gender=Neut|Number=Sing	1	nsubj	_	LemmaId=37876|OccId=3542463|Unsandhied=idam
4	ekaḥ	eka	NUM	_	Case=Nom|Gender=Masc|Number=Sing	1	acl	_	LemmaId=39102|OccId=3542464|Unsandhied=ekaḥ
5	eva	eva	PART	_	_	4	discourse	_	LemmaId=39754
6	agre	agre	ADV	_	_	1	advmod	_	LemmaId=207355
7	āsīt	as	VERB	_	Mood=Ind|Number=Sing|Person=3|Tense=Past	1	cop	_	LemmaId=156122|Punctuation=fullStop

# text = saḥ īkṣata iti
# sent_id = 556278_1
1-2	tacchrutvā	_	_	_	_	_	_	_	_
1	saḥ	tad	PRON	_	Case=Nom|Gender=Masc|Number=Sing	2	nsubj	_	LemmaId=37875
2	īkṣata	īkṣ	VERB	_	Mood=Ind|Number=Sing|Person=3	0	root	_	LemmaId=156133
2.1	_	_	_	_	_	_	_	_	_
3	iti	iti	PART
4	,	,	PUNCT	_	_	2	punct	_	_
```

(The `1-2` line is a UD multiword range, `2.1` an empty node — both skipped, word lines carry the data. Line `3` has only 4 columns — a malformed line for parse-loss accounting. Line `4` is PUNCT — skipped.)

- [ ] **Step 2: Write the failing tests**

`languages/sanskrit/tests/test_01_dcs.py`:

```python
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parent.parent / "scripts" / "01_parse_dcs.py"
_spec = spec_from_file_location("san_parse_01", str(_path))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.conllu"


def test_parse_file_meta_and_lines():
    parsed = _mod.parse_file(FIXTURE)
    assert parsed["text"] == "Aitareyopaniṣad"
    assert parsed["text_id"] == "421"
    assert parsed["chapter_id"] == "8816"
    # sentence 1: 7 forms; sentence 2: saḥ īkṣata (malformed "iti" line and
    # PUNCT excluded, range/empty-node lines skipped)
    assert parsed["lines"][0] == "ātmā vai idam ekaḥ eva agre āsīt"
    assert parsed["lines"][1] == "saḥ īkṣata"


def test_lemma_records():
    parsed = _mod.parse_file(FIXTURE)
    recs = parsed["lemmas"]
    first = recs[0]
    assert first == {"form": "ātmā", "cf": "ātman", "gw": "", "pos": "NOUN", "lang": "san"}
    # gold lemma for saḥ is tad (suppletive) — the join key later
    by_form = {r["form"]: r for r in recs}
    assert by_form["saḥ"]["cf"] == "tad"
    # multiword surface string and empty node never become records
    assert "tacchrutvā" not in by_form
    assert "," not in by_form


def test_parse_loss_accounting():
    parsed = _mod.parse_file(FIXTURE)
    assert parsed["bad_lines"] == 1          # the 4-column "iti" line
    assert parsed["token_lines"] >= 10       # counted before validity checks
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest languages/sanskrit/tests/test_01_dcs.py -v`
Expected: FAIL (file `01_parse_dcs.py` not found).

- [ ] **Step 4: Write the implementation**

`languages/sanskrit/scripts/01_parse_dcs.py`:

```python
"""
DCS CoNLL-U Parser: convert Digital Corpus of Sanskrit conllu dumps to
hyper-glyphy lemma + text JSON.

Source: github.com/OliverHellwig/sanskrit (CC BY 4.0), dcs/data/conllu/files/
— one folder per text, one .conllu file per chapter. ~745k lines, ~5.46M words.

Chapter file headers:
  ## text: Aitareyopaniṣad
  ## text_id: 421
  ## chapter: AU, 1, 1
  ## chapter_id: 8816

Token lines are UD CoNLL-U: FORM (col 2) is the sandhi-resolved word form in
IAST; LEMMA (col 3) is the gold lemma. Multiword range lines (ID "1-2") and
empty nodes (ID "2.1") are skipped — their word lines carry the data. PUNCT
rows are skipped. Malformed token lines (fewer than 10 columns) are counted
and reported as parse loss, never silently dropped.

Output:
  - data/raw/sanskrit_texts.json:
      [{p_number: "dcs-<text_id>-<chapter_id>", text_name, chapter,
        lines: [str], source: "DCS"}]
  - data/raw/sanskrit_lemmas.json:
      [{form, cf, gw (EMPTY — glosses joined in 06 from MW), pos, lang: "san"}]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tqdm import tqdm

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DCS_FILES_DEFAULT = DATA_RAW / "dcs" / "dcs" / "data" / "conllu" / "files"

_HEADER_RE = re.compile(r"##\s+(text|text_id|chapter|chapter_id):\s*(.*)")


def parse_file(path: Path) -> dict:
    """Parse one DCS chapter .conllu file."""
    meta = {"text": "", "text_id": "", "chapter": "", "chapter_id": ""}
    lines: list[str] = []
    lemmas: list[dict] = []
    token_lines = 0
    bad_lines = 0
    sent_tokens: list[str] = []

    def flush():
        nonlocal sent_tokens
        if sent_tokens:
            lines.append(" ".join(sent_tokens))
            sent_tokens = []

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("##"):
                m = _HEADER_RE.match(line)
                if m:
                    meta[m.group(1)] = m.group(2).strip()
                continue
            if not line.strip():
                flush()
                continue
            if line.startswith("#"):
                continue
            if not line[0].isdigit():
                continue
            token_lines += 1
            cols = line.split("\t")
            if len(cols) < 10:
                bad_lines += 1
                continue
            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:
                continue  # multiword range / empty node
            form, lemma, upos = cols[1], cols[2], cols[3]
            if upos == "PUNCT" or form in ("", "_"):
                continue
            sent_tokens.append(form)
            lemmas.append({
                "form": form,
                "cf": lemma if lemma != "_" else "",
                "gw": "",  # DCS has no glosses; injected by 06 from MW
                "pos": upos,
                "lang": "san",
            })
    flush()
    return {**meta, "lines": lines, "lemmas": lemmas,
            "token_lines": token_lines, "bad_lines": bad_lines}


def main(dcs_files: Path = DCS_FILES_DEFAULT):
    if not dcs_files.exists():
        raise SystemExit(
            f"DCS conllu files not found at {dcs_files}. Run the documented "
            "fetch step (languages/sanskrit/README.md): sparse clone of "
            "https://github.com/OliverHellwig/sanskrit.git into data/raw/dcs."
        )
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    all_texts: list[dict] = []
    all_lemmas: list[dict] = []
    token_lines = 0
    bad_lines = 0

    conllu_files = sorted(dcs_files.rglob("*.conllu"))
    print(f"Parsing {len(conllu_files)} DCS conllu files from {dcs_files}")

    for path in tqdm(conllu_files, desc="DCS"):
        parsed = parse_file(path)
        token_lines += parsed["token_lines"]
        bad_lines += parsed["bad_lines"]
        if parsed["lines"]:
            all_texts.append({
                "p_number": f"dcs-{parsed['text_id']}-{parsed['chapter_id']}",
                "text_name": parsed["text"],
                "chapter": parsed["chapter"],
                "lines": parsed["lines"],
                "source": "DCS",
            })
        all_lemmas.extend(parsed["lemmas"])

    with open(DATA_RAW / "sanskrit_texts.json", "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    with open(DATA_RAW / "sanskrit_lemmas.json", "w", encoding="utf-8") as f:
        json.dump(all_lemmas, f, ensure_ascii=False, indent=2)

    total_lines = sum(len(t["lines"]) for t in all_texts)
    unique_cf = len({l["cf"] for l in all_lemmas if l["cf"]})
    loss_pct = 100.0 * bad_lines / max(1, token_lines)
    print(f"\nTotal chapter files: {len(all_texts)}")
    print(f"Total lines: {total_lines}")
    print(f"Total token-lemma records: {len(all_lemmas)}")
    print(f"Unique lemmas (cf): {unique_cf}")
    print(f"Parse loss: {bad_lines}/{token_lines} token lines ({loss_pct:.3f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest languages/sanskrit/tests/test_01_dcs.py -v`
Expected: 3 passed.

- [ ] **Step 6: Full pytest + commit**

```bash
pytest -q
git add languages/sanskrit/
git commit -m "feat(sanskrit): DCS conllu parser (FORM stream + gold lemmas, parse-loss accounting)"
```

---

### Task 3: `02_parse_mw.py` — MW XML → gloss JSON (SLP1→IAST)

**Files:**
- Create: `languages/sanskrit/scripts/02_parse_mw.py`
- Create: `languages/sanskrit/tests/fixtures/sample_mw.xml`
- Test: `languages/sanskrit/tests/test_02_mw.py`

**Interfaces:**
- Produces: `data/dictionaries/mw_glosses.json` = `[{key1, lemma_iast, lemma_norm, glosses: [str], gloss_first: str}]` — same shape as Greek's `lsj_glosses.json` so 06's index-by-`lemma_norm` works after sed. Exports `slp1_to_iast(slp: str) -> str` and `extract_glosses(body_el) -> list[str]`.
- Consumed by: 06.

Note: the spec sketch said `data/raw/mw_glosses.json` keyed `{headword: [glosses]}`; we follow the Greek convention instead (`data/dictionaries/`, list of entry dicts) so the 06 clone needs no structural changes. Recorded here as the deliberate resolution.

- [ ] **Step 1: Write the fixture**

`languages/sanskrit/tests/fixtures/sample_mw.xml` (real MW entries, verbatim structure):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<mw>
<H1><h><key1>deva</key1><key2>deva/</key2></h><body><s>deva/</s>   <lex>mf(<s>I</s>)n.</lex> (<ab>fr.</ab> <hom>3.</hom> <s>div</s>) heavenly, divine (also said of terrestrial things of high excellence), <ls>RV.</ls>; <ls>AV.</ls><info lex="m:f#I:n"/></body><tail><L>95518</L><pc>492,2</pc></tail></H1>
<H2><h><key1>ahiMsA</key1><key2>a/-hiMsA</key2></h><body><s>a/-hiMsA</s>   <lex>f.</lex> not injuring anything, harmlessness (one of the cardinal virtues of most <ns>Hindū</ns> sects), <ls>ChUp.</ls> &amp;c.<info lex="f"/></body><tail><L>21896</L><pc>125,2</pc></tail></H2>
<H3><h><key1>akAra</key1><key2>a—kAra</key2></h><body><s>a—kAra</s>   <lex>m.</lex> the letter or sound <s>a</s>.<info lex="m"/></body><tail><L>2</L><pc>1,1</pc></tail></H3>
<H1><h><key1>deva</key1><key2>deva</key2><hom>2</hom></h><body>duplicate headword — must be dropped by dedup.</body><tail><L>99999</L><pc>1,1</pc></tail></H1>
</mw>
```

- [ ] **Step 2: Write the failing tests**

`languages/sanskrit/tests/test_02_mw.py`:

```python
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parent.parent / "scripts" / "02_parse_mw.py"
_spec = spec_from_file_location("san_parse_02", str(_path))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_mw.xml"


def test_slp1_to_iast_full_coverage():
    cases = {
        "ahiMsA": "ahiṃsā",
        "kfzRa": "kṛṣṇa",
        "SAstra": "śāstra",
        "buddhO": "buddhau",   # O -> au digraph expansion
        "vEdya": "vaidya",     # E -> ai digraph expansion
        "guRa": "guṇa",
        "duHKa": "duḥkha",
        "aNga": "aṅga",
        "jYAna": "jñāna",
        "wIkA": "ṭīkā",
        "QORqa": "ḍhauṇḍa",
        "pfTvI": "pṛthvī",
        "kAvya": "kāvya",
    }
    for slp, iast in cases.items():
        assert _mod.slp1_to_iast(slp) == iast, slp


def test_parse_entries_and_glosses():
    entries = _mod.parse_mw_file(FIXTURE)
    by_norm = {e["lemma_norm"]: e for e in entries}

    deva = by_norm["deva"]
    assert deva["key1"] == "deva"
    assert deva["glosses"][:2] == ["heavenly", "divine"]
    assert deva["gloss_first"] == "heavenly"

    ahimsa = by_norm["ahiṃsā"]
    # parenthetical stripped; sources (<ls>) and grammar (<lex>) excluded
    assert ahimsa["glosses"] == ["not injuring anything", "harmlessness"]

    akara = by_norm["akāra"]
    assert akara["glosses"] == ["the letter or sound"]


def test_dedup_keeps_first_homonym():
    entries = _mod.parse_mw_file(FIXTURE)
    deva_entries = [e for e in entries if e["lemma_norm"] == "deva"]
    assert len(deva_entries) == 1
    assert deva_entries[0]["gloss_first"] == "heavenly"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest languages/sanskrit/tests/test_02_mw.py -v`
Expected: FAIL (file `02_parse_mw.py` not found).

- [ ] **Step 4: Write the implementation**

`languages/sanskrit/scripts/02_parse_mw.py`:

```python
"""
Monier-Williams Parser: extract Sanskrit-English glosses from the Cologne
CDSL MW XML (1899 edition, 2020 digitization).

Source: https://www.sanskrit-lexicon.uni-koeln.de/scans/MWScan/2020/downloads/mwxml.zip
  -> xml/mw.xml (67 MB). Entry elements are <H1>..<H4> plus suffixed
  variants (<H1A>, <H2B>, <H1E>, ...). Each has <h><key1>SLP1 headword</key1>
  and a <body> mixing English gloss text with tagged non-English material:
  <s> (SLP1 Sanskrit), <ls> (literary sources), <ab> (abbreviations),
  <lex> (grammar), <gk>/<lang>/<etym> (etymology), <ns> (proper names), etc.

Gloss extraction: collect body text OUTSIDE excluded tags, strip
parentheticals, split on [;,], trim noise segments. First gloss segment is
`gloss_first` (06 falls back to later segments — including when a segment is
negation-led, e.g. "not injuring anything" -> falls through to "harmlessness").

SLP1 -> IAST at parse time via a deterministic in-repo table (no dependency).

Output: data/dictionaries/mw_glosses.json
  [{key1, lemma_iast, lemma_norm, glosses: [...], gloss_first}]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from languages.sanskrit.scripts.sanskrit_normalize import normalize_sanskrit_token  # noqa: E402

DATA_DICTS = Path(__file__).parent.parent / "data" / "dictionaries"
MW_XML_DEFAULT = Path(__file__).parent.parent / "data" / "raw" / "mw" / "xml" / "mw.xml"

# Full SLP1 set including digraph expansions (spec 2026-07-13).
SLP1_TO_IAST = {
    "a": "a", "A": "ā", "i": "i", "I": "ī", "u": "u", "U": "ū",
    "f": "ṛ", "F": "ṝ", "x": "ḷ", "X": "ḹ",
    "e": "e", "E": "ai", "o": "o", "O": "au",
    "M": "ṃ", "H": "ḥ", "~": "m̐",
    "k": "k", "K": "kh", "g": "g", "G": "gh", "N": "ṅ",
    "c": "c", "C": "ch", "j": "j", "J": "jh", "Y": "ñ",
    "w": "ṭ", "W": "ṭh", "q": "ḍ", "Q": "ḍh", "R": "ṇ",
    "t": "t", "T": "th", "d": "d", "D": "dh", "n": "n",
    "p": "p", "P": "ph", "b": "b", "B": "bh", "m": "m",
    "y": "y", "r": "r", "l": "l", "v": "v",
    "S": "ś", "z": "ṣ", "s": "s", "h": "h", "L": "ḻ",
}

_ENTRY_TAG_RE = re.compile(r"^H[1-4][A-E]?$")
_PAREN_RE = re.compile(r"\([^)]*\)")
_NOISE_SEGMENTS = {"&c", "etc", "ib", "q.v", "cf"}

# Body child tags whose TEXT is non-English (their tails are kept — English
# gloss text frequently sits in tails, e.g. "(<ab>fr.</ab> <s>div</s>) heavenly").
EXCLUDE_TAGS = {
    "s", "s1", "ls", "ab", "lex", "hom", "info", "lang", "gk", "etym",
    "ns", "bio", "bot", "root", "pb", "srs", "shc", "vlex", "abE", "pcol",
}


def slp1_to_iast(slp: str) -> str:
    """Deterministic char-by-char SLP1 -> IAST; unknown chars pass through."""
    return "".join(SLP1_TO_IAST.get(ch, ch) for ch in slp)


def _localname(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _collect_english_text(el) -> str:
    """Text of `el` excluding the text of EXCLUDE_TAGS subtrees (tails kept)."""
    parts = [el.text or ""]
    for child in el:
        if _localname(child.tag) not in EXCLUDE_TAGS:
            parts.append(_collect_english_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def extract_glosses(body_el) -> list[str]:
    """English gloss segments from a <body>: strip parentheticals, split on
    [;,], drop noise. Downstream 06 handles negation-led segments."""
    text = _collect_english_text(body_el)
    text = _PAREN_RE.sub(" ", text)
    glosses: list[str] = []
    for seg in re.split(r"[;,]", text):
        seg = re.sub(r"\s+", " ", seg).strip(" .:—–-'’\"")
        if not seg or seg.lower() in _NOISE_SEGMENTS:
            continue
        if not re.search(r"[A-Za-z]{2}", seg):
            continue
        glosses.append(seg)
    return glosses


def parse_mw_file(path: Path) -> list[dict]:
    """Parse mw.xml. Returns deduplicated entries (first occurrence per
    lemma_norm wins — usually the main entry, matching the LSJ recipe)."""
    entries: list[dict] = []
    for _, elem in ET.iterparse(str(path), events=("end",)):
        tag = _localname(elem.tag)
        if not _ENTRY_TAG_RE.match(tag):
            continue
        h = elem.find("h")
        body = elem.find("body")
        key1 = h.findtext("key1") if h is not None else None
        if not key1 or body is None:
            elem.clear()
            continue
        glosses = extract_glosses(body)
        if glosses:
            lemma_iast = slp1_to_iast(key1)
            lemma_norm = normalize_sanskrit_token(lemma_iast)
            if lemma_norm:
                entries.append({
                    "key1": key1,
                    "lemma_iast": lemma_iast,
                    "lemma_norm": lemma_norm,
                    "glosses": glosses,
                    "gloss_first": glosses[0],
                })
        elem.clear()

    seen: dict[str, dict] = {}
    for entry in entries:
        seen.setdefault(entry["lemma_norm"], entry)
    return list(seen.values())


def main(mw_xml: Path = MW_XML_DEFAULT):
    if not mw_xml.exists():
        raise SystemExit(
            f"MW XML not found at {mw_xml}. Run the documented fetch step "
            "(languages/sanskrit/README.md): download mwxml.zip from Cologne "
            "CDSL and unzip into data/raw/mw/."
        )
    DATA_DICTS.mkdir(parents=True, exist_ok=True)

    deduped = parse_mw_file(mw_xml)
    output_path = DATA_DICTS / "mw_glosses.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"Entries after dedup by lemma_norm: {len(deduped)}")
    print(f"Saved to: {output_path}")
    print("\nFirst 10 glosses:")
    for e in deduped[:10]:
        print(f"  {e['lemma_iast']:>20s}  ->  {e['gloss_first']!r}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest languages/sanskrit/tests/test_02_mw.py -v`
Expected: 3 passed.

- [ ] **Step 6: Full pytest + commit**

```bash
pytest -q
git add languages/sanskrit/
git commit -m "feat(sanskrit): MW XML parser with SLP1->IAST table and gloss extraction"
```

---

### Task 4: Data acquisition + real 01/02 runs + slot README

**Files:**
- Create: `languages/sanskrit/README.md`
- Produces (gitignored): `data/raw/dcs/`, `data/raw/mw/xml/mw.xml`, `data/raw/sanskrit_texts.json`, `data/raw/sanskrit_lemmas.json`, `data/dictionaries/mw_glosses.json`

- [ ] **Step 1: Fetch DCS (sparse clone — full repo is ~1.8 GB; conllu subtree only)**

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/OliverHellwig/sanskrit.git \
  languages/sanskrit/data/raw/dcs
git -C languages/sanskrit/data/raw/dcs sparse-checkout set dcs/data/conllu
ls languages/sanskrit/data/raw/dcs/dcs/data/conllu/files | head
```

Expected: text-named directories (Aitareyopaniṣad, Atharvavedasaṃhitā, Ṛgveda, …).

- [ ] **Step 2: Fetch MW XML**

```bash
curl -o languages/sanskrit/data/raw/mwxml.zip \
  https://www.sanskrit-lexicon.uni-koeln.de/scans/MWScan/2020/downloads/mwxml.zip
unzip -o languages/sanskrit/data/raw/mwxml.zip -d languages/sanskrit/data/raw/mw
ls -la languages/sanskrit/data/raw/mw/xml/
```

Expected: `mw.xml` (~67 MB), `mw.dtd`, `mwheader.xml`.

- [ ] **Step 3: Run 01 and 02 on the real data**

```bash
python languages/sanskrit/scripts/01_parse_dcs.py
python languages/sanskrit/scripts/02_parse_mw.py
```

Expected: 01 reports on the order of ~5M token-lemma records and parse loss well under 1% (if higher, STOP and inspect the offending files before proceeding). 02 reports on the order of 100k+ deduplicated entries. Record the exact printed numbers — they go into the README and journal.

- [ ] **Step 4: Write `languages/sanskrit/README.md`**

Follow `languages/greek/README.md` structure exactly (title, Status, Corpus, Anchors, Pipeline scripts table, Running, Tests). Document under **Corpus**:
- DCS: `git clone --depth 1 --filter=blob:none --sparse https://github.com/OliverHellwig/sanskrit.git languages/sanskrit/data/raw/dcs` + `git -C languages/sanskrit/data/raw/dcs sparse-checkout set dcs/data/conllu`. License CC BY 4.0. Cite Hellwig, *The Digital Corpus of Sanskrit (DCS)*, 2010–2024. Record the measured text/line/record counts from Step 3.
- MW: `curl -o languages/sanskrit/data/raw/mwxml.zip https://www.sanskrit-lexicon.uni-koeln.de/scans/MWScan/2020/downloads/mwxml.zip` + unzip to `data/raw/mw/`. Cologne CDSL 2020 digitization; licensing details in `mwheader.xml`. Record the measured entry count.
- Note the two deliberate deviations from the Greek recipe: (1) FORM-stream tokenization is the Greek convention (kept, spec-locked); (2) 06's negation-gloss rule (survey A1, user-approved) with a pointer to the journal entry.

Status section says: pipeline scaffolded, runs pending (updated in Task 11).

- [ ] **Step 5: Commit**

```bash
pytest -q
git add languages/sanskrit/README.md
git commit -m "docs(sanskrit): slot README — pinned data sources, fetch steps, measured parse stats"
```

---

### Task 5: `04_deduplicate_corpus.py` (sed-clone) + thin `05_clean_and_tokenize.py`

**Files:**
- Create: `languages/sanskrit/scripts/04_deduplicate_corpus.py` (sed-clone)
- Create: `languages/sanskrit/scripts/05_clean_and_tokenize.py` (new thin file — NOT a clone; Greek's 05 is an ATF cleaner that would drop avagraha tokens and mangle IAST)
- Produces (gitignored): `data/processed/merged_corpus.json`, `data/processed/cleaned_corpus.txt`

**Interfaces:**
- Consumes: `data/raw/sanskrit_texts.json` (Task 2 schema).
- Produces: `cleaned_corpus.txt` — one line per chapter text, space-separated normalized IAST tokens; consumed by 07.

- [ ] **Step 1: Sed-clone 04 with diff-gate**

```bash
sed -e 's/greek/sanskrit/g' -e 's/Greek/Sanskrit/g' -e 's/Diorisis/DCS/g' \
  languages/greek/scripts/04_deduplicate_corpus.py > languages/sanskrit/scripts/04_deduplicate_corpus.py
diff <(sed -e 's/greek/sanskrit/g' -e 's/Greek/Sanskrit/g' -e 's/Diorisis/DCS/g' \
  languages/greek/scripts/04_deduplicate_corpus.py) languages/sanskrit/scripts/04_deduplicate_corpus.py
```

Expected: empty diff. (p_numbers `dcs-<text_id>-<chapter_id>` are unique per chapter, so dedup is a pass-through — kept for pipeline-shape parity.)

- [ ] **Step 2: Write `05_clean_and_tokenize.py`**

```python
"""
Corpus Tokenization: normalize the DCS FORM stream for FastText.

Deliberately NOT a clone of the Greek/Sumerian 05 — that script is an ATF
transliteration cleaner (hyphen morpheme-splitting, all-caps sign-name drops,
leading-apostrophe token drops) which would corrupt IAST text (e.g. delete
avagraha forms like 'bravīt). DCS conllu FORM tokens are already clean,
sandhi-resolved IAST words; the only work is canonical normalization.

Output: cleaned_corpus.txt (one line per chapter text, space-separated tokens)
"""
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from languages.sanskrit.scripts.sanskrit_normalize import normalize_sanskrit_token  # noqa: E402

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def clean_line(line: str) -> str:
    """Normalize each whitespace-separated token; drop empties."""
    tokens = [normalize_sanskrit_token(tok) for tok in line.split()]
    return " ".join(t for t in tokens if t)


def build_corpus(texts: list[dict]) -> list[str]:
    """One corpus line per text (all its lines joined), matching the
    format 07's CorpusIterator expects."""
    corpus_lines = []
    for text in texts:
        cleaned_words = []
        for line in text.get("lines", []):
            cleaned = clean_line(line)
            if cleaned:
                cleaned_words.append(cleaned)
        if cleaned_words:
            corpus_lines.append(" ".join(cleaned_words))
    return corpus_lines


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    with open(DATA_PROCESSED / "merged_corpus.json") as f:
        texts = json.load(f)

    corpus_lines = build_corpus(texts)

    output_path = DATA_PROCESSED / "cleaned_corpus.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(line + "\n")

    total_tokens = sum(len(line.split()) for line in corpus_lines)
    vocab = set()
    for line in corpus_lines:
        vocab.update(line.split())

    print(f"Corpus lines: {len(corpus_lines)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Unique tokens: {len(vocab)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
```

(No test file: `clean_line` is a two-line composition of the already-tested normalizer; correctness is covered by `test_normalize.py` plus the run below.)

- [ ] **Step 3: Run 04 and 05**

```bash
python languages/sanskrit/scripts/04_deduplicate_corpus.py
python languages/sanskrit/scripts/05_clean_and_tokenize.py
```

Expected: 04 stats show `duplicates_removed: 0`; 05 total tokens ≈ 01's token-lemma record count (small delta from empty normalizations only). Spot-check: `head -c 400 languages/sanskrit/data/processed/cleaned_corpus.txt` shows lowercase IAST with diacritics intact (ā, ṛ, ṣ visible).

- [ ] **Step 4: Full pytest + commit**

```bash
pytest -q
git add languages/sanskrit/scripts/04_deduplicate_corpus.py languages/sanskrit/scripts/05_clean_and_tokenize.py
git commit -m "feat(sanskrit): corpus dedup clone + thin IAST tokenizer (no ATF machinery)"
```

---

### Task 6: `06_extract_anchors.py` — clone + negation rule + hit-rate gate

**Files:**
- Create: `languages/sanskrit/scripts/06_extract_anchors.py` (sed-clone of Greek 06 + three enumerated edits)
- Test: `languages/sanskrit/tests/test_06_anchors.py`
- Produces (gitignored): `data/processed/english_anchors.json`, `data/processed/anchor_stats.json`

**Interfaces:**
- Consumes: `data/raw/sanskrit_lemmas.json` (Task 2), `data/dictionaries/mw_glosses.json` (Task 3), `shared/models/english_gemma_768d.npz`.
- Produces: `english_anchors.json` `[{sanskrit, english, confidence, frequency, source: "DCS+MW", lemmas: [cf_norm]}]` — the `lemmas` field drives `group_split`'s union-find in 09/09b. `extract_anchors(lemmas, mw_index, eng_vocab_set, min_occurrences=5) -> tuple[list[dict], dict]` (NOTE: returns `(anchors, stats)` — a deliberate signature change vs Greek, needed by the gate). `MIN_HIT_RATE = 0.40`.

- [ ] **Step 1: Sed-clone the base**

```bash
sed -e 's/greek/sanskrit/g' -e 's/Greek/Sanskrit/g' \
    -e 's/lsj/mw/g' -e 's/LSJ/MW/g' -e 's/Diorisis/DCS/g' \
  languages/greek/scripts/06_extract_anchors.py > languages/sanskrit/scripts/06_extract_anchors.py
```

This lands: `normalize_sanskrit_token` import, `sanskrit_lemmas.json` / `mw_glosses.json` inputs, `build_mw_index`, anchor key `"sanskrit"`, source `"DCS+MW"`. (`"Perseus MW XML"` in the docstring gets rewritten in Step 2.)

- [ ] **Step 2: Apply the three enumerated edits**

**(a) Docstring** — replace the module docstring with:

```python
"""
Sanskrit Anchor Extraction: DCS lemmas + MW glosses.

Pipeline:
1. Load DCS lemmas (token-lemma records from the conllu dump).
2. Load MW glosses (Cologne CDSL Monier-Williams entries, 02_parse_mw.py).
3. Join: for each DCS lemma's `cf`, normalize and look up in MW keyed by
   `lemma_norm`. Use the first usable MW gloss as the anchor's English.
4. From the gloss, extract the first English content word that exists in the
   english_gemma_768d cache vocab — that becomes the anchor's `english`.

Deviation from the Greek/LSJ recipe (user-approved 2026-07-14, survey A1):
a gloss containing a negator ("not", "no", "without", "never") before its
first in-vocab content word is skipped entirely — MW is dense with privative
glosses ("not injuring anything") and harvesting the negated word would
manufacture antonym anchors. The caller falls through to the entry's next
gloss segment.

Gate (PGM lesson): the DCS-lemma -> MW token-level join hit rate is computed
and persisted to anchor_stats.json BEFORE any FastText compute; below
MIN_HIT_RATE (40%) is a hard SystemExit, never a silent continue.

Mirrors Akkadian/Hittite anchor schema: {sanskrit, english, confidence,
frequency, source, lemmas}.
"""
```

**(b) Negation rule** — replace the `STOP_WORDS` constant and `_load_gloss_first_english` (post-sed state) with:

```python
MIN_HIT_RATE = 0.40

# "not"/"no" are deliberately NOT stop-words here (they are in the Greek
# recipe): a negator must invalidate the gloss, not be skipped over.
NEGATORS = {"not", "no", "without", "never"}

STOP_WORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with",
    "be", "is", "are", "was", "were", "as", "or", "and",
    "but", "if", "so", "do", "did", "have", "has", "had", "from", "into",
    "out", "up", "down", "over", "under", "between", "during", "before",
    "after", "above", "below", "any", "some", "all", "each", "every",
    "one", "two", "three", "four", "five",
}

_WORD_RE = re.compile(r"[a-z][a-z'\-]*")


def _load_gloss_first_english(eng_vocab_set: set[str], gloss: str) -> str | None:
    """Return the first English content word in `gloss` that exists in
    eng_vocab — or None if the gloss is negated before any match.

    "not injuring anything" -> None (caller tries the entry's next gloss);
    "harmlessness" -> "harmlessness". Strips hyphens for compound matches.
    """
    if not gloss:
        return None
    lowered = gloss.lower()
    for word in _WORD_RE.findall(lowered):
        if word in NEGATORS:
            return None
        if word in STOP_WORDS:
            continue
        if word in eng_vocab_set:
            return word
        if "-" in word:
            joined = word.replace("-", "")
            if joined in eng_vocab_set:
                return joined
    return None
```

**(c) Stats + gate** — in `extract_anchors` (post-sed the counters are named `mw_hits`/`mw_misses`), change the return to a tuple and build stats. Replace the tail of `extract_anchors` (from the two `print` lines through `return sorted(...)`) with:

```python
    print(f"  MW join: {mw_hits} hits, {mw_misses} misses (cf not in MW)")
    print(f"  Glosses with no in-vocab English: {gloss_no_eng}")

    anchors: list[dict] = []
    for (sanskrit_form, eng), count in pair_counts.items():
        if count < min_occurrences:
            continue
        confidence = min(0.95, 0.5 + (count / 100))
        anchors.append({
            "sanskrit": sanskrit_form,
            "english": eng,
            "confidence": round(confidence, 4),
            "frequency": count,
            "source": "DCS+MW",
            "lemmas": sorted(pair_lemmas[(sanskrit_form, eng)]),
        })
    anchors = sorted(anchors, key=lambda a: a["confidence"], reverse=True)

    stats = {
        "mw_hits": mw_hits,
        "mw_misses": mw_misses,
        "token_hit_rate": mw_hits / max(1, mw_hits + mw_misses),
        "gloss_no_eng": gloss_no_eng,
        "anchors": len(anchors),
    }
    return anchors, stats
```

And in `main()`, replace the extraction/save block with:

```python
    anchors, stats = extract_anchors(lemmas, mw_index, eng_vocab_set)
    output_path = DATA_PROCESSED / "english_anchors.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(anchors, f, ensure_ascii=False, indent=2)
    stats_path = DATA_PROCESSED / "anchor_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTotal anchors: {len(anchors)}")
    print(f"Token-level MW join hit rate: {stats['token_hit_rate']:.1%}")
    print(f"Saved to: {output_path} (+ {stats_path.name})")

    if stats["token_hit_rate"] < MIN_HIT_RATE:
        raise SystemExit(
            f"MW join hit rate {stats['token_hit_rate']:.1%} is below the "
            f"{MIN_HIT_RATE:.0%} gate (PGM lesson). Inspect lemma "
            "normalization / MW parse before any FastText compute."
        )
```

- [ ] **Step 3: Write the failing tests**

`languages/sanskrit/tests/test_06_anchors.py`:

```python
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parent.parent / "scripts" / "06_extract_anchors.py"
_spec = spec_from_file_location("san_anchors_06", str(_path))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _entry(norm, glosses):
    return {"lemma_norm": norm, "gloss_first": glosses[0], "glosses": glosses}


def test_anchors_carry_lemmas_and_surfaces():
    lemmas = [{"cf": "Deva", "form": "devāḥ"}] * 5
    mw_index = {"deva": _entry("deva", ["heavenly", "divine"])}
    anchors, stats = _mod.extract_anchors(lemmas, mw_index, {"heavenly"}, min_occurrences=5)
    by_surface = {a["sanskrit"]: a for a in anchors}
    assert by_surface["deva"]["lemmas"] == ["deva"]      # cf normalized (lowercased)
    assert by_surface["devāḥ"]["lemmas"] == ["deva"]     # surface registered too
    assert by_surface["deva"]["english"] == "heavenly"
    assert by_surface["deva"]["source"] == "DCS+MW"
    assert stats["token_hit_rate"] == 1.0


def test_negated_gloss_skipped_not_harvested():
    # MW ahiṃsā: "not injuring anything, harmlessness". The Greek recipe
    # would anchor to "injuring" (antonym). We must fall through to
    # "harmlessness".
    lemmas = [{"cf": "ahiṃsā", "form": "ahiṃsā"}] * 5
    mw_index = {"ahiṃsā": _entry("ahiṃsā", ["not injuring anything", "harmlessness"])}
    vocab = {"injuring", "harmlessness"}
    anchors, _ = _mod.extract_anchors(lemmas, mw_index, vocab, min_occurrences=5)
    assert anchors
    assert all(a["english"] == "harmlessness" for a in anchors)


def test_all_glosses_negated_drops_anchor():
    lemmas = [{"cf": "abhāva", "form": "abhāva"}] * 5
    mw_index = {"abhāva": _entry("abhāva", ["not existing", "without form"])}
    anchors, stats = _mod.extract_anchors(
        lemmas, mw_index, {"existing", "form"}, min_occurrences=5)
    assert anchors == []
    assert stats["gloss_no_eng"] == 5


def test_hit_rate_in_stats():
    lemmas = [{"cf": "deva", "form": "deva"}] * 3 + [{"cf": "nope", "form": "nope"}] * 7
    mw_index = {"deva": _entry("deva", ["heavenly"])}
    _, stats = _mod.extract_anchors(lemmas, mw_index, {"heavenly"}, min_occurrences=1)
    assert stats["mw_hits"] == 3 and stats["mw_misses"] == 7
    assert abs(stats["token_hit_rate"] - 0.30) < 1e-9
    assert stats["token_hit_rate"] < _mod.MIN_HIT_RATE
```

- [ ] **Step 4: Run tests to verify they fail, then reconcile**

Run: `pytest languages/sanskrit/tests/test_06_anchors.py -v`
Expected before edits are complete: failures (tuple unpacking / negation). After Step 2 edits: 4 passed. Also `python -m py_compile languages/sanskrit/scripts/06_extract_anchors.py`.

- [ ] **Step 5: Run on real data — the gate moment**

```bash
python languages/sanskrit/scripts/06_extract_anchors.py
cat languages/sanskrit/data/processed/anchor_stats.json
```

Expected: token hit rate WELL above 40% (DCS lemmas are gold, MW is comprehensive; Greek's analogous join was 77%). If SystemExit fires: STOP — surface the number and the first 20 missed lemmas to the user; do not proceed to Task 7.

- [ ] **Step 6: Full pytest + commit**

```bash
pytest -q
git add languages/sanskrit/scripts/06_extract_anchors.py languages/sanskrit/tests/test_06_anchors.py
git commit -m "feat(sanskrit): anchor extraction — MW join, negation-gloss rule, 40% hit-rate gate"
```

---

### Task 7: Sed-clone 07/08/09/09b/10 + `align_09.py`

**Files:**
- Create: `languages/sanskrit/scripts/{07_train_fasttext,08_fuse_embeddings,09_align_and_evaluate,09b_align_gemma,10_export_production}.py`, `languages/sanskrit/scripts/align_09.py`

**Interfaces:**
- Consumes: `cleaned_corpus.txt` (05), `english_anchors.json` (06).
- Produces: `models/fasttext_sanskrit.model|.vec` (07), `models/fused_embeddings_1536d.npz` (08), `results/alignment_results*.json` + eval artifacts (09/09b), `final_output/sanskrit_aligned*_vectors.npz` + vocab JSON (10).

The `sumerian→sanskrit` substitution is included deliberately: Greek's clones carry legacy `fasttext_sumerian.model` filenames and "Sumerian" docstrings; a fresh slot has no trained artifact to protect, so Sanskrit gets consistent `fasttext_sanskrit.*` names. (`anchor.get("greek") or anchor.get("sumerian")` in 09 becomes a harmless duplicated `anchor.get("sanskrit")`.)

- [ ] **Step 1: Sed-clone with diff-gate**

```bash
for f in 07_train_fasttext.py 08_fuse_embeddings.py 09_align_and_evaluate.py \
         09b_align_gemma.py 10_export_production.py; do
  sed -e 's/greek/sanskrit/g' -e 's/Greek/Sanskrit/g' \
      -e 's/sumerian/sanskrit/g' -e 's/Sumerian/Sanskrit/g' \
      -e 's/Diorisis/DCS/g' \
    languages/greek/scripts/$f > languages/sanskrit/scripts/$f
  diff <(sed -e 's/greek/sanskrit/g' -e 's/Greek/Sanskrit/g' \
      -e 's/sumerian/sanskrit/g' -e 's/Sumerian/Sanskrit/g' \
      -e 's/Diorisis/DCS/g' \
    languages/greek/scripts/$f) languages/sanskrit/scripts/$f
done
cp languages/greek/scripts/align_09.py languages/sanskrit/scripts/align_09.py
```

Expected: every diff empty. (`align_09.py` is slot-agnostic — it loads `09_align_and_evaluate.py` by relative path.)

- [ ] **Step 2: Grep-gate + compile-gate**

```bash
grep -rn 'greek\|Greek\|sumerian\|Sumerian' languages/sanskrit/scripts/ && echo "GATE FAIL" || echo "GATE PASS"
python -m py_compile languages/sanskrit/scripts/*.py
```

Expected: `GATE PASS`, clean compile. Also verify the load-bearing names landed:

```bash
grep -n 'fasttext_sanskrit' languages/sanskrit/scripts/0{7,8,9}* languages/sanskrit/scripts/09b_align_gemma.py
grep -n 'SURFACE_KEY' languages/sanskrit/scripts/09_align_and_evaluate.py languages/sanskrit/scripts/09b_align_gemma.py
```

Expected: `fasttext_sanskrit.model/.vec` in 07/08/09/09b; `SURFACE_KEY = "sanskrit"` in both.

- [ ] **Step 3: Full pytest + commit**

```bash
pytest -q
git add languages/sanskrit/scripts/
git commit -m "feat(sanskrit): pipeline clones 07-10 + align_09 (sed from greek canonicals, gated)"
```

---

### Task 8: Train FastText + fuse (pipeline runs)

**Files:**
- Produces (gitignored): `models/fasttext_sanskrit.model|.vec`, `models/fused_embeddings_1536d.npz`

- [ ] **Step 1: Train FastText (detached — ~5.4M tokens × 10 epochs)**

```bash
mkdir -p languages/sanskrit/logs
nohup python languages/sanskrit/scripts/07_train_fasttext.py \
  > languages/sanskrit/logs/07_train.log 2>&1 &
```

Poll `languages/sanskrit/logs/07_train.log` until it prints the saved-model lines. Expected: `dim=768, window=10, min_count=2, epochs=10` echoed; both `fasttext_sanskrit.model` and `.vec` written under `languages/sanskrit/models/`.

- [ ] **Step 2: Fuse**

```bash
python languages/sanskrit/scripts/08_fuse_embeddings.py
python - <<'EOF'
import numpy as np
d = np.load("languages/sanskrit/models/fused_embeddings_1536d.npz")
assert d["vectors"].shape[1] == 1536, d["vectors"].shape
print("fused:", d["vectors"].shape, "vocab:", len(d["vocab"]))
EOF
```

Expected: fused dim 1536, vocab = FastText vocab size.

- [ ] **Step 3: Commit (log/journal note only — models are gitignored; nothing to add unless 07/08 needed no changes, in which case skip the commit)**

No commit expected in this task; record vocab size and training wall-time for the journal entry (Task 11).

---

### Task 9: Alignment runs (09 GloVe + 09b whitened-Gemma) + suite record

**Files:**
- Produces (gitignored): `languages/sanskrit/results/alignment_results.json`, `alignment_results_gemma_whitened.json`, eval artifacts

- [ ] **Step 1: Run both alignments detached, sequentially (each ~1–4 h)**

```bash
nohup bash -c '
python languages/sanskrit/scripts/09_align_and_evaluate.py &&
python languages/sanskrit/scripts/09b_align_gemma.py --mode whitened
' > languages/sanskrit/logs/09_alignments.log 2>&1 &
```

Poll the log until both suite blocks print.

- [ ] **Step 2: Verify suite integrity**

From each results JSON record: selected alpha, the stratified CSLS suite (dictionary / interpolation / zero-shot, exact AND syn, top-1/5/10), `cand_size` (must be 50000), and the leak check (expected **0.00%**; anything >1% is a STOP — inspect `group_split` behavior before proceeding). Also record `gold_oov_candidates` — MW's Latinate gloss vocabulary may push this high; it goes in the journal caveats verbatim.

- [ ] **Step 3: Commit**

Results are gitignored; nothing to commit unless a clone needed a fix (it shouldn't — any fix must be applied via the Task 7 sed derivation and re-gated, never hand-edited).

---

### Task 10: Procrustes anchor-quality read-out (pre-registered)

**Files:**
- Modify: `shared/scripts/procrustes_align.py:31-35` (SLOTS)
- Produces (gitignored): `languages/sanskrit/results/procrustes_results.json`, `models/sanskrit_procrustes_gemma_vectors.npz`

- [ ] **Step 1: Add the slot**

In `shared/scripts/procrustes_align.py`, change:

```python
SLOTS = {
    "sumerian": {"surface_key": "sumerian"},
    "hittite": {"surface_key": "hittite"},
    "greek": {"surface_key": "greek"},
}
```

to:

```python
SLOTS = {
    "sumerian": {"surface_key": "sumerian"},
    "hittite": {"surface_key": "hittite"},
    "greek": {"surface_key": "greek"},
    "sanskrit": {"surface_key": "sanskrit"},
}
```

- [ ] **Step 2: Run the fitter (sanskrit only — existing slots' results must not be touched)**

```bash
python shared/scripts/procrustes_align.py --slot sanskrit
```

Expected: two-variant fit (full/stable), val selection, test split untouched, results JSON written under `languages/sanskrit/results/`.

- [ ] **Step 3: Apply the pre-registered rule — verbatim, no adjustment**

Read the selected variant's **val cosine** and write the verdict against the bands exactly as pre-registered in the spec (existing slots for reference: sumerian 0.1157, hittite 0.0586, greek 0.1149):

- ≥ 0.20 ⇒ "anchor quality WAS a binding constraint for the other slots; the stronger-anchors lever stays live."
- ≤ 0.12 ⇒ "anchors were never the constraint; the stronger-anchors lever is retired, and with it the last named route to Plane A."
- 0.12–0.20 ⇒ "inconclusive, stated as such."

The verdict sentence goes into the journal (Task 11) verbatim against these bands.

- [ ] **Step 4: Full pytest + commit**

```bash
pytest -q
git add shared/scripts/procrustes_align.py
git commit -m "feat(shared): sanskrit slot in procrustes read-out (pre-registered anchor-quality test)"
```

---

### Task 11: Production export + docs (README, journal, root README)

**Files:**
- Produces (gitignored): `languages/sanskrit/final_output/*`
- Modify: `languages/sanskrit/README.md` (Status, Anchors, measured numbers)
- Modify: `docs/EXPERIMENT_JOURNAL.md` (new dated entry)
- Modify: `README.md` (suite table row; remove Vedic/Sanskrit from the future-slots table)

- [ ] **Step 1: Export**

```bash
python languages/sanskrit/scripts/10_export_production.py
ls -la languages/sanskrit/final_output/
```

Expected: `sanskrit_aligned_vectors.npz`, `sanskrit_aligned_gemma_vectors.npz`, `sanskrit_aligned_vocab.json`, metadata with `seed`/`test_size` keys.

- [ ] **Step 2: Journal entry**

Append a dated entry (2026-07-XX, run date) to `docs/EXPERIMENT_JOURNAL.md` containing, in this order:
1. Slot summary: corpus stats (01 output), MW entries (02), anchor count + token hit rate (06 `anchor_stats.json`), FastText vocab.
2. **Disclosure:** the negation-gloss deviation from the Greek 06 recipe (rule, rationale = survey finding A1 / MW privatives like ahiṃsā, user-approved 2026-07-14) and the thin non-ATF 05.
3. Word-level suite numbers (both targets, all strata, exact+syn, leak check, `gold_oov_candidates`).
4. The Procrustes read-out: chosen variant, val cosine, and the pre-registered verdict sentence from Task 10 Step 3 — verbatim, with the bands restated.

- [ ] **Step 3: README updates**

- `languages/sanskrit/README.md`: Status → shipped (commit hash); fill Anchors section (count, hit rate, split note: lemma-group union-find, 64/16/20, seed 42); add suite numbers.
- Root `README.md`: add the Sanskrit row to the suite table (same columns as the other five slots); delete the Vedic/Sanskrit line from the future-slots table.

- [ ] **Step 4: Full pytest + final commit**

```bash
pytest -q
git add languages/sanskrit/README.md docs/EXPERIMENT_JOURNAL.md README.md
git commit -m "feat(sanskrit): production export + docs — suite numbers and pre-registered anchor read-out"
```

---

## Self-Review Record

- **Spec coverage:** full DCS (T2/T4), MW anchors with SLP1→IAST digraph table (T3), FORM-stream FastText (T2/T5/T8), per-text DCS IDs preserved as `dcs-<text_id>-<chapter_id>` (T2), gloss hit-rate gate (T6), sed-clones with verification (T5/T7), lemma-group split seed 42 via cloned 09/09b (T7/T9), procrustes read-out with verbatim bands (T10), README/journal/root-README (T4/T11), fail-loud error handling (T2/T3 SystemExits, parse-loss accounting), all spec-listed tests present (T1/T2/T3/T6). Approved deviation (negation rule) is constrained to 06, disclosed in journal + README.
- **Known deliberate deltas vs spec text:** `mw_glosses.json` lives in `data/dictionaries/` as a list (Greek convention; recorded in T3); 05 is a new thin file rather than a clone (spec's own §Architecture describes 05 as "sanskrit_normalize on FORM stream", which the ATF clone does not implement); `extract_anchors` returns `(anchors, stats)` (needed by the spec-mandated gate).
- **Type consistency:** `normalize_sanskrit_token` (T1) used in T2 tests/T3/T5/T6; `parse_file` keys (T2) match T2 tests; `_entry` helper matches `mw_glosses.json` schema (`lemma_norm`/`gloss_first`/`glosses`) consumed by the sed-derived `build_mw_index`; `extract_anchors` tuple return consistent between T6 code, T6 tests, and T6 `main()`.
