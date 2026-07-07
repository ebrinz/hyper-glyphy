# languages/akkadian — Old Babylonian Akkadian Alignment

OB Akkadian aligned to whitened-EmbeddingGemma (768d, primary) and GloVe
(300d, secondary). Pipeline structure mirrors `languages/sumerian/` 1:1.

## Current results (v1.3)

> **Pre-fix (leaked split)** — these numbers are invalidated as surface-variant leakage
> artifacts per the 2026-07-06 eval-integrity audit. Akkadian rerun under the
> leakage-free split: Gemma 0.14%, GloVe 0.09% top-1. Reruns of other metrics
> deferred pending eval redesign. See [`docs/EXPERIMENT_JOURNAL.md`](../../../../docs/EXPERIMENT_JOURNAL.md).

| Metric | Whitened-Gemma 768d | GloVe 300d |
|--------|:---:|:---:|
| Top-1  | **36.43%** | 27.79% |
| Top-5  | 59.18% | 43.23% |
| Top-10 | **66.51%** | 47.52% |

Pre-fix: top-10 (66.51%) exceeded Sumerian's (65.99%); top-1 gap to Sumerian
(36.43% vs 52.13%) was the remaining alignment-precision frontier.

Corpus 3.0M tokens, FastText vocab 45,769, ORACC anchor pool 24,415,
Ridge alpha 0.01 (Gemma) / 0.001 (GloVe). See
[`docs/EXPERIMENT_JOURNAL.md`](docs/EXPERIMENT_JOURNAL.md) for the v1 ship,
v1.1 three-lever pass, v1.2 four-lever pass (incl. one falsified hypothesis),
and the v1.3 alpha-sweep finding (+7.41pp Gemma top-1 from one constant change).

## Quick start

```bash
# Scrape (network)
python scripts/01_scrape_oracc_ob.py        # OB ORACC projects (~2M tokens)
python scripts/01b_scrape_oracc_sb.py       # Standard Babylonian supplement (~1M tokens; needed for 3.0M corpus)
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
