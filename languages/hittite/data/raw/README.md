# languages/hittite/data/raw

Raw corpus data for the Hittite slot. Gitignored except this file —
regenerate by running the fetch script below (`languages/hittite/scripts/`).

## Source

**TLHdig** 0.2.0-beta (Hethitologie Portal Mainz, via Zenodo record 15459134),
22k morphologically-annotated XML texts, CC-BY:

```bash
curl -sL "https://zenodo.org/records/15459134/files/TLHdig_0.2.0-beta.zip" \
  -o /tmp/tlhdig.zip && unzip -q /tmp/tlhdig.zip -d /tmp/tlhdig/
```

`01_parse_tlhdig.py` expects the extracted archive at
`/tmp/tlhdig/TLHbasisONLINE25.1_ZENODO` by default (override via the
`tlhdig_root` argument); it exits with a download-instructions error if that
path doesn't exist. Glosses in the source are German — translated to English
downstream at anchor-build time (06), no separate dictionary fetch needed.

Run: `python languages/hittite/scripts/01_parse_tlhdig.py`

## Expected layout after fetching

```
data/raw/
├── hittite_texts.json        [{p_number, lines, source: "TLHdig"}]
├── hittite_lemmas.json       [{form, cf, gw (German), pos, lang, ...}]
└── hittite_heterograms.json  {sumerograms: {...}, akkadograms: {...}}
```

(The `letters/`, `ob_literary/`, and `heterograms/` subdirectories present in
some checkouts are leftovers from the Akkadian bridge-data fetch, not produced
by this slot's own scripts.)
