# languages/greek/data/raw

Raw corpus data for the Classical Greek slot. Gitignored except this file —
regenerate by running the fetch + parse scripts below
(`languages/greek/scripts/`). Both sources require manual download; the
parsers exit with instructions if the extract isn't found at the expected
path.

## Sources

- **Diorisis Ancient Greek Corpus** (Figshare dataset 12251468), v1.51, 821
  JSON files (Beta Code), Homer to 5th c. AD, ~10.2M tokens. Manual download:

  ```
  https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus_JSON_/12251468
  ```

  Extract to `/tmp/diorisis` (default `diorisis_root`), then:
  `python languages/greek/scripts/01_parse_diorisis.py` → writes
  `greek_texts.json`, `greek_lemmas.json` (glosses empty — Diorisis has none).

- **LSJ** (Liddell-Scott-Jones lexicon), Perseus Digital Library
  `github.com/PerseusDL/lexica`, `CTS_XML_TEI/perseus/pdllex/grc/lsj/`, ~28
  XML files, ~115k entries:

  ```bash
  git clone https://github.com/PerseusDL/lexica /tmp/lsj_repo
  cp /tmp/lsj_repo/CTS_XML_TEI/perseus/pdllex/grc/lsj/*.xml /tmp/lsj
  ```

  Extract XML files to `/tmp/lsj` (default `lsj_root`), then: `python
  languages/greek/scripts/02_parse_lsj.py` → writes
  `../dictionaries/lsj_glosses.json` (not under `data/raw`).

License: not stated in-repo for either source; consult Figshare/Diorisis and
PerseusDL directly.

## Expected layout after fetching

```
data/raw/
├── greek_texts.json
└── greek_lemmas.json
```
