# languages/hittite — Hittite (TLHdig) Alignment

Hittite aligned to whitened-EmbeddingGemma (768d, primary) and GloVe
(300d, secondary). Pipeline structure mirrors `languages/akkadian/` with
two notable deltas:

1. **Source corpus is TLHdig** (Hethitologie Portal Mainz / Zenodo), not
   ORACC. ORACC has near-zero Hittite content; TLHdig publishes 22k
   morphologically-annotated XML texts under CC-BY.
2. **Glosses are German**, translated to English at anchor-build time
   via multilingual EmbeddingGemma + NLTK English wordlist filter — no
   explicit German-English dictionary required.

## Current results (v1 ship)

| Metric | Whitened-Gemma 768d | GloVe 300d |
|--------|:---:|:---:|
| Top-1  | **40.62%** | 35.40% |
| Top-5  | 50.99% | 41.88% |
| Top-10 | 55.02% | 43.20% |

Top-1 (40.62%) beats Akkadian's v1.3 (36.43%) from day one, despite ~20% the
corpus. GloVe top-1 (35.40%) matches Sumerian (35.70%) and beats Egyptian.
See [`docs/EXPERIMENT_JOURNAL.md`](docs/EXPERIMENT_JOURNAL.md) for the full
writeup including the German→English translation strategy and remaining
levers.

## Quick start

```bash
# Download TLHdig from Zenodo (~64MB zipped, 11GB extracted)
curl -sL "https://zenodo.org/records/15459134/files/TLHdig_0.2.0-beta.zip" \
  -o /tmp/tlhdig.zip && unzip -q /tmp/tlhdig.zip -d /tmp/tlhdig/

# Parse XML to JSON
python scripts/01_parse_tlhdig.py

# Process
python scripts/04_deduplicate_corpus.py
python scripts/05_clean_and_tokenize.py

# Anchors (Gemma encodes 3k German glosses; ~1 min)
python scripts/06_extract_anchors.py

# Train + align
python scripts/07_train_fasttext.py
python scripts/08_fuse_embeddings.py
python scripts/09_align_and_evaluate.py             # GloVe target
python scripts/09b_align_gemma.py --mode whitened   # Gemma target

# Optional: sweep Ridge alpha (recommended at v1)
python scripts/ridge_alpha_sweep.py

# Export
python scripts/10_export_production.py
```

## Lookup API

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('languages/hittite/final_output')))
from hittite_lookup import HittiteLookup
lkp = HittiteLookup(space="gemma")  # or "glove"
lkp.lookup("atanzi", k=10)  # "they eat" → morphological variants
```

## Heterogram-bridge data

`data/raw/hittite_heterograms.json` — aggregate Sumerogram (4,081 unique)
and Akkadogram (2,829 unique) counts from TLHdig. These feed into the v1
anchor extractor's bridge path: bridge anchors via existing
`languages/sumerian/` and `languages/akkadian/` aligned-Gemma spaces.

## Spec & process notes

No formal design spec (used the Akkadian framework directly with deltas
captured in the journal). Lessons from Akkadian's 7-iteration arc baked
into v1:

- Ridge α sweep on day one (Akkadian L7 lesson: +7.41pp from one constant)
- Subword inference with train-only OOV partition (Akkadian L5-refined)
- Lemma-surface expansion in anchor extraction (Akkadian L4)
- No speculative dictionary fetcher (Akkadian L6b falsification)
