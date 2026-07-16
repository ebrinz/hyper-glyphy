# languages/sanskrit — Sanskrit Alignment

Cross-lingual embedding alignment for Sanskrit, mapping into both GloVe 300d
and whitened-EmbeddingGemma 768d English semantic spaces. Pipeline structure
mirrors `languages/greek/` (scripts 04–10 are sed-clones of the Greek
canonicals). See the [design spec](../../docs/superpowers/specs/2026-07-13-sanskrit-slot-design.md)
for full rationale — Sanskrit is the best-resourced slot buildable, and
doubles as the stronger-anchors experiment for the Procrustes anchor-quality
question (journal, 2026-07-13).

## Status

Shipped. Anchor extraction (commit a096673), pipeline clones 07–10 +
FastText/Ridge alignment run (commit 63f2af9), Procrustes anchor-quality
read-out (commit a5ecc08), production export (this commit). See the
[repo journal](../../docs/EXPERIMENT_JOURNAL.md), 2026-07-16 entry, for the
full measured writeup.

## Anchors

95,924 anchors extracted (Task 6); token-level DCS-lemma→MW join hit rate
94.9% (5,391,784 hits / 287,678 misses), 40% gate passed; 90,176 valid at
fit time. Split via the shared lemma-group union-find (`anchor_split.py`):
64/16/20 train/val/test, seed 42. Anchor extraction applies the
negation-gloss rule (see "Deliberate deviations" below): glosses hitting a
negator before an in-vocab content word skip to the next gloss segment.

## Word-level suite

Seed 42 lemma-group split (near-surface edges), 50,000 candidates, CSLS
retrieval. Leak check: 0.00% (0/19,185).

| Target | alpha | Dict top-1 | Interp top-1 | Zero-shot top-1 | Combined top-1 | Combined syn |
|--------|-------|:----------:|:-------------:|:----------------:|:---------------:|:------------:|
| GloVe 300d | 1.0 | 33.51% | 2.55% | 0.23% | 2.22% | 3.73% |
| Gemma whitened 768d | 1000 | 44.40% | 4.35% | 0.74% | 3.83% | 6.18% |

Gemma beats GloVe combined: +1.62pp top-1. Sanskrit also carries the
pre-registered Procrustes anchor-quality read-out (val cosine 0.1145,
≤0.12 band fired — see journal 2026-07-16 entry): with the best token-level
hit rate of any slot (94.9%), the fact that the val cosine still lands in
the same ~0.115 band as Sumerian (0.1157) and Greek (0.1149) indicates
anchors were never the binding constraint on the semi-orthogonal plane.

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

1. **FORM-stream tokenization.** FastText (07) trains on the sandhi-resolved
   FORM stream from the conllu, not the lemma stream — this is the Greek
   convention, kept spec-locked to keep the anchor-quality comparison
   against the other five slots unconfounded. See the design spec, "Scope."
   Real run: 768d skip-gram, window 10, min_count 2, epochs 10, sg 1;
   vocab 195,309.
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
| `04_deduplicate_corpus.py` | Deduplicate corpus lines |
| `05_clean_and_tokenize.py` | Normalize the DCS FORM stream for FastText — a thin IAST tokenizer, not the Greek/Sumerian ATF-cleaning clone |
| `06_extract_anchors.py` | Join DCS lemmas with MW glosses → anchor pairs; carries the negation-gloss rule and the 40% hit-rate gate |
| `07_train_fasttext.py` | Train 768d FastText skip-gram embeddings |
| `08_fuse_embeddings.py` | Zero-pad fusion [768d \| 000...768d] → 1536d |
| `09_align_and_evaluate.py` | Ridge regression → GloVe 300d |
| `align_09.py` | Shared helper module re-exporting 09's training/evaluation functions for reuse by 09b |
| `09b_align_gemma.py` | Ridge regression → whitened Gemma 768d |
| `10_export_production.py` | Dual-view production export (GloVe 300d + whitened Gemma 768d) |

## Running

Data-fetch steps (sparse-clone DCS, download/unzip MW) precede 01/02 — see
"Corpus" above for the exact commands. Full reproduction order:

```bash
python languages/sanskrit/scripts/01_parse_dcs.py
python languages/sanskrit/scripts/02_parse_mw.py
python languages/sanskrit/scripts/04_deduplicate_corpus.py
python languages/sanskrit/scripts/05_clean_and_tokenize.py
python languages/sanskrit/scripts/06_extract_anchors.py
python languages/sanskrit/scripts/07_train_fasttext.py
python languages/sanskrit/scripts/08_fuse_embeddings.py
python languages/sanskrit/scripts/09_align_and_evaluate.py
python languages/sanskrit/scripts/09b_align_gemma.py --mode whitened
python languages/sanskrit/scripts/10_export_production.py
python shared/scripts/procrustes_align.py --slot sanskrit
```

## Tests

```bash
pytest languages/sanskrit/tests/ -v
```
