# Egyptian Alignment Experiment Journal

> **2026-07-06 — All accuracy numbers in this journal are pre-fix (leaked split).**
> See the repo-wide journal entry ([`docs/EXPERIMENT_JOURNAL.md`](../../../../docs/EXPERIMENT_JOURNAL.md),
> 2026-07-06) for the full eval-integrity writeup. Egyptian's gloss-group split (union-find
> on shared gloss, since Egyptian anchors carry no lemma field) makes honest zero-shot
> exact-match ~0% by construction; eval redesign pending.

---

## 2026-07-06 — Pre-fix numbers recorded as historical; eval-integrity note added

Numbers from the 2026-05-04 ship (PCA-256, alpha 1.0 Gemma / 0.1 GloVe) are
pre-fix — see banner above and repo journal.

## 2026-05-04 — Phase Beta shipped: whitened-Gemma alignment complete

**Results (pre-fix, leaked split):**

| Metric | Whitened-Gemma 768d | GloVe 300d |
|--------|:---:|:---:|
| Top-1  | 34.57% | 33.42% |
| Top-5  | 40.76% | 41.75% |
| Top-10 | 43.73% | 45.38% |

Parameters: PCA-256, Ridge alpha 1.0 (Gemma) / 0.1 (GloVe). Valid anchors 6,060
of 8,170 total. Source: `languages/egyptian/final_output/metadata.json`.

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
