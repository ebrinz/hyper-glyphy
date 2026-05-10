# languages/akkadian — Old Babylonian Akkadian Alignment

OB Akkadian aligned to whitened-EmbeddingGemma (768d, primary) and GloVe
(300d, secondary). Pipeline structure mirrors `languages/sumerian/` 1:1.

## Current results

| Metric | Whitened-Gemma 768d | GloVe 300d |
|--------|:---:|:---:|
| Top-1  | 21.66% | 9.63% |

See [`docs/EXPERIMENT_JOURNAL.md`](docs/EXPERIMENT_JOURNAL.md) for the v1 ship, v1.1 gap-closing pass, and the identified levers remaining.

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
python scripts/06_extract_anchors.py

# Train + align (requires shared/models/english_gemma_*.npz and Sumerian's GloVe -- symlink before running)
python scripts/07_train_fasttext.py
python scripts/08_fuse_embeddings.py
python scripts/09_align_and_evaluate.py     # GloVe target
python scripts/09b_align_gemma.py --mode whitened  # whitened-Gemma target

# Export
python scripts/10_export_production.py
```

## Lookup API

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('languages/akkadian/final_output')))
from akkadian_lookup import AkkadianLookup
lkp = AkkadianLookup(space="gemma")  # or "glove"
print(lkp.lookup("szarrum", k=10))
```

## Bridge data

`data/processed/sumerian_akkadian_pairs.jsonl` — 50,636 Sumerian<->Akkadian
word pairs from DCCLT lexical lists. Parsed but not yet used; v2 will
cross-validate Sumerian-Gemma and Akkadian-Gemma alignments through
these pairs.

## Spec & plan

- Spec: `docs/superpowers/specs/2026-05-09-akkadian-slot-design.md`
- Plan: `docs/superpowers/plans/2026-05-09-akkadian-slot.md`
