# languages/sanskrit — Sanskrit Alignment

Cross-lingual embedding alignment for Sanskrit, mapping into both GloVe 300d
and whitened-EmbeddingGemma 768d English semantic spaces. Pipeline structure
mirrors `languages/greek/` (scripts 04–10 are sed-clones of the Greek
canonicals). See the [design spec](../../docs/superpowers/specs/2026-07-13-sanskrit-slot-design.md)
for full rationale — Sanskrit is the best-resourced slot buildable, and
doubles as the stronger-anchors experiment for the Procrustes anchor-quality
question (journal, 2026-07-13).

## Status

Pipeline scaffolded (01, 02, normalizer), real data fetched and parsed
(this task). Runs pending — 04–10 and the eval suite land in later tasks
(updated in Task 11).

## Corpus

- **DCS** (Digital Corpus of Sanskrit, github.com/OliverHellwig/sanskrit,
  CC BY 4.0). Sparse-cloned the `dcs/data/conllu` subtree:

  ```bash
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/OliverHellwig/sanskrit.git \
    languages/sanskrit/data/raw/dcs
  git -C languages/sanskrit/data/raw/dcs sparse-checkout set dcs/data/conllu
  ```

  Both the initial partial clone and the sparse-checkout `set` (which
  fetches the missing blobs) worked as documented; no fallback to a plain
  `git clone --depth 1` was needed. 270 text-named chapter directories
  (Aitareyopaniṣad, Atharvavedasaṃhitā, Ṛgveda, …), 15,900 `.conllu` files.
  Measured from the real `01_parse_dcs.py` run: **15,790 chapter files**
  (110 files produced zero non-empty lines and were skipped), **754,502
  lines**, **5,679,462 token-lemma records**, **90,184 unique lemmas**,
  parse loss **0/6,713,257 token lines (0.000%)**.
  Cite: Hellwig, O., *The Digital Corpus of Sanskrit (DCS)*, 2010–2024.

- **Monier-Williams** (Cologne CDSL 2020 digitization of the 1899 edition).

  ```bash
  curl -o languages/sanskrit/data/raw/mwxml.zip \
    https://www.sanskrit-lexicon.uni-koeln.de/scans/MWScan/2020/downloads/mwxml.zip
  unzip -o languages/sanskrit/data/raw/mwxml.zip -d languages/sanskrit/data/raw/mw
  ```

  `xml/mw.xml` (~64 MB), `mw.dtd`, `mwheader.xml`, `mw-meta2.txt`. Licensing
  per `mwheader.xml`: Copyright 2014 The Sanskrit Library and Thomas Malten,
  released under CC BY-NC-SA 3.0 (non-commercial — narrower than DCS's CC BY
  4.0; note this if the corpus is ever redistributed downstream). Measured
  from the real `02_parse_mw.py` run: **177,323 entries** after dedup by
  `lemma_norm`.

## Deliberate deviations from the Greek recipe

1. **FORM-stream tokenization.** FastText (07, not yet run) will train on
   the sandhi-resolved FORM stream from the conllu, not the lemma stream —
   this is the Greek convention, kept spec-locked to keep the anchor-quality
   comparison against the other five slots unconfounded. See the design
   spec, "Scope."
2. **06's negation-gloss rule.** MW glosses that begin with a negation-led
   segment (e.g. `ahiṃsā` → `"not injuring anything"`) fall through to the
   next gloss segment (`"harmlessness"`) when 06 selects the anchor's
   English content word, rather than taking the negation-led segment
   verbatim. Surveyed under approach A1 and user-approved; see the
   implementation plan (commit 1ba88d5) and the eventual journal entry for
   06's real run.

Spot-checking `mw_glosses.json`, a meaningful share of `gloss_first` values
are short fragments (`"of"`, `"or"`, `"see"`) or proper-name stubs (`"of a
king"`, `"of an author"`) left over after `<ns>`/`<ab>` tag exclusion —
15.1% of entries have a `gloss_first` that is a single stopword; 21.9% start
with `of`/`or`. This is expected given MW's proper-name-heavy entries and is
handled downstream: 06 selects the first English *content word* present in
the Gemma vocab cache, so stopword-only `gloss_first` values simply fail to
yield a content word rather than producing a garbage anchor. No entries had
a `gloss_first` ending in a typographic quote (’) or containing no letters
(0/177,323 both counts).

## Pipeline scripts

| Script | Purpose |
|--------|---------|
| `01_parse_dcs.py` | Parse DCS conllu → token-lemma records + per-chapter texts |
| `02_parse_mw.py` | Parse Monier-Williams XML (SLP1 → IAST) → Sanskrit-English gloss pairs |
| `sanskrit_normalize.py` | IAST canonicalization (NFC, lowercase) used by 01/02 and downstream |

Scripts 04–10 (deduplicate, normalize/tokenize, extract anchors, train
FastText, fuse, align+evaluate x2, export) are not yet created for this
slot — they land as sed-clones of the Greek canonicals in later tasks.

## Running

```bash
python languages/sanskrit/scripts/01_parse_dcs.py
python languages/sanskrit/scripts/02_parse_mw.py
```

## Tests

```bash
pytest languages/sanskrit/tests/ -v
```
