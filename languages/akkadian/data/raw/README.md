# languages/akkadian/data/raw

Raw corpus data for the Old Babylonian Akkadian slot. Gitignored except this
file — regenerate by running the fetch scripts below
(`languages/akkadian/scripts/`).

## Sources

All sources are ORACC (oracc.museum.upenn.edu) project JSON archives,
downloaded per-project as `oracc_<project>.zip`. License not stated in-repo;
consult ORACC directly.

- **OB-period projects** (hbtin, saao/saa, blms, rinap/rinap1): `python
  languages/akkadian/scripts/01_scrape_oracc_ob.py` → `ob_literary/`,
  `ob_literary_texts.json`, `ob_literary_lemmas.json`.
- **Standard Babylonian supplement** (~1M tokens; needed for the full 3.0M
  corpus) — full project list in the script: `python
  languages/akkadian/scripts/01b_scrape_oracc_sb.py` → `sb/`, `sb_texts.json`,
  `sb_lemmas.json`.
- **OB letters** (saao/saa01, saao/saa17): `python
  languages/akkadian/scripts/02_scrape_oracc_letters.py` → `ob_letters/`,
  `ob_letters_texts.json`, `ob_letters_lemmas.json`.
- **DCCLT** (Digital Corpus of Cuneiform Lexical Texts; also yields the
  Sumerian↔Akkadian bridge pairs for the v2 cross-lingual experiment): `python
  languages/akkadian/scripts/03_scrape_dcclt.py` → `dcclt/`,
  `dcclt_texts.json`, `dcclt_lemmas.json` (and
  `data/processed/sumerian_akkadian_pairs.jsonl`).

Training also requires `shared/models/english_gemma_*.npz` and a symlink to
Sumerian's GloVe cache — see the slot README's "Quick start".

## Expected layout after fetching

```
data/raw/
├── ob_literary/oracc_<project>.zip, ob_literary_texts.json, ob_literary_lemmas.json
├── sb/oracc_<project>.zip, sb_texts.json, sb_lemmas.json
├── ob_letters/oracc_<project>.zip, ob_letters_texts.json, ob_letters_lemmas.json
├── dcclt/oracc_dcclt.zip, dcclt_texts.json, dcclt_lemmas.json
└── oracc_lemmas.json
```
