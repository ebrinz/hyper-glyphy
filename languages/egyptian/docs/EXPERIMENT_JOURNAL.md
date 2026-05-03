# Egyptian Alignment Experiment Journal

## 2026-05-03 — Phase Beta: Port to hyper-glyphy

Ported heiroglyphy V15 Egyptian alignment pipeline into hyper-glyphy monorepo.

**Baseline (heiroglyphy V15, GloVe 300d only):** 32.35% top-1, 41.47% top-5, 45.13% top-10.

**Changes from V15:**
- Dropped visual features (ResNet-50 768d, 0.59% match rate) in favor of pure zero-padding to match Sumerian pipeline structure.
- Added whitened-Gemma 768d as primary target (new for Egyptian).
- Standardized anchor format to hyper-glyphy convention.
- Added full pytest test suite.

**Data provenance:**
- Corpus: heiroglyphy `heiro_v5_getdata/data/processed/cleaned_corpus.txt` (100,729 lines, 789K tokens, BBAW/TLA sources)
- Anchors: heiroglyphy `heiro_v5_getdata/data/processed/english_anchors.json` (8,541 pairs from TLA/Ramses/BBAW)
- FastText model: heiroglyphy `heiro_v15/models/fasttext_mc5_w10` (768d, window=10, min_count=5, sg=1, epochs=10)
