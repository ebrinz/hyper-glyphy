# Egyptian (Hieroglyphic) Alignment

Cross-lingual embedding alignment for ancient Egyptian hieroglyphic transliterations, mapping into both GloVe 300d and whitened-EmbeddingGemma 768d English semantic spaces.

## Pipeline

Scripts 01-05 (corpus building) are not yet ported. The cleaned corpus and anchors were migrated from heiroglyphy V15.

| Script | Purpose |
|--------|---------|
| `06_extract_anchors.py` | Normalize heiroglyphy anchors to hyper-glyphy format |
| `07_train_fasttext.py` | Train 768d FastText skip-gram embeddings |
| `08_fuse_embeddings.py` | Zero-pad fusion [768d \| 000...768d] -> 1536d |
| `09_align_and_evaluate.py` | Ridge regression -> GloVe 300d |
| `09b_align_gemma.py` | Ridge regression -> whitened Gemma 768d |
| `10_export_production.py` | Dual-view production export |

## Running

```bash
# From repo root
python languages/egyptian/scripts/06_extract_anchors.py
python languages/egyptian/scripts/07_train_fasttext.py
python languages/egyptian/scripts/08_fuse_embeddings.py
python languages/egyptian/scripts/09_align_and_evaluate.py
python languages/egyptian/scripts/09b_align_gemma.py
python languages/egyptian/scripts/10_export_production.py
```

## Tests

```bash
pytest languages/egyptian/tests/ -v
```
