# languages/sanskrit/data/raw

Raw corpus data for the Sanskrit slot. Gitignored except this file —
regenerate by running the fetch commands below, then the parse scripts
(`languages/sanskrit/scripts/`).

## Sources

- **DCS** (Digital Corpus of Sanskrit, `github.com/OliverHellwig/sanskrit`),
  CC BY 4.0. Sparse-clone the `dcs/data/conllu` subtree:

  ```bash
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/OliverHellwig/sanskrit.git \
    languages/sanskrit/data/raw/dcs
  git -C languages/sanskrit/data/raw/dcs sparse-checkout set dcs/data/conllu
  ```

  270 text-named chapter directories, 15,900 `.conllu` files. Cite: Hellwig,
  O., *The Digital Corpus of Sanskrit (DCS)*, 2010–2024. Then:
  `python languages/sanskrit/scripts/01_parse_dcs.py`.

- **Monier-Williams** (Cologne CDSL 2020 digitization of the 1899 edition),
  CC BY-NC-SA 3.0 (non-commercial — narrower than DCS; per `mwheader.xml`,
  Copyright 2014 The Sanskrit Library and Thomas Malten):

  ```bash
  curl -o languages/sanskrit/data/raw/mwxml.zip \
    https://www.sanskrit-lexicon.uni-koeln.de/scans/MWScan/2020/downloads/mwxml.zip
  unzip -o languages/sanskrit/data/raw/mwxml.zip -d languages/sanskrit/data/raw/mw
  ```

  Then: `python languages/sanskrit/scripts/02_parse_mw.py`.

## Expected layout after fetching

```
data/raw/
├── dcs/dcs/data/conllu/...      (sparse git checkout)
├── mwxml.zip
└── mw/xml/mw.xml, mw.dtd, mwheader.xml, mw-meta2.txt
```

See `languages/sanskrit/README.md` ("Corpus") for full provenance notes.
