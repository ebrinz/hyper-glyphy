# languages/sumerian/data/raw

Raw corpus data for the Sumerian slot. Gitignored except this file — regenerate
by running the fetch scripts below (`languages/sumerian/scripts/`).

## Sources

- **ETCSL** (Electronic Text Corpus of Sumerian Literature), Oxford Text
  Archive: `python languages/sumerian/scripts/01_scrape_etcsl.py` — downloads
  `etcsl.zip` (TEI P4 XML) and writes `etcsl_texts.json`.
- **CDLI** (Cuneiform Digital Library Initiative) bulk ATF dump,
  `github.com/cdli-gh/data` (git-lfs): `python
  languages/sumerian/scripts/02_scrape_cdli.py` — clones the repo to
  `cdli-data/` (needs `git lfs install` first) and writes `cdli_texts.json`
  from `cdli-data/cdliatf_unblocked.atf`.
- **ORACC** (oracc.museum.upenn.edu) project JSON archives — six Sumerian-heavy
  projects (epsd2/admin/ur3, epsd2/admin/ed3b, epsd2/admin/ed3a,
  epsd2/literary, etcsri, dcclt): `python
  languages/sumerian/scripts/03_scrape_oracc.py` — downloads
  `oracc/oracc_<project>.zip` per project and writes `oracc_texts.json` +
  `oracc_lemmas.json`.
- **ORACC blms** (bilingual Sumerian-Akkadian incantation/prayer texts,
  Udug-hul series): `python languages/sumerian/scripts/11_fetch_incantations.py`
  — downloads `incantations/oracc_blms.zip`, or reuses
  `languages/akkadian/data/raw/ob_literary/oracc_blms.zip` if already present.

License: not stated in-repo for any of the four sources above; consult OTA,
CDLI, and ORACC directly before redistribution.

## Expected layout after fetching

```
data/raw/
├── etcsl.zip, etcsl_texts.json
├── cdli-data/                      (git clone; cdliatf_unblocked.atf)
├── cdli_texts.json
├── oracc/oracc_<project>.zip       (6 files)
├── oracc_texts.json, oracc_lemmas.json
└── incantations/oracc_blms.zip     (optional)
```
