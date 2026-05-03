# Egyptian Phase Beta Design Spec

## Goal

Port heiroglyphy V15 Egyptian alignment pipeline into `languages/egyptian/` within hyper-glyphy, producing cross-comparable GloVe and whitened-Gemma alignments that match the Sumerian pipeline's structure 1:1.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Visual features | Drop (pure zero-pad) | 0.59% match rate is noise; makes fusion identical to Sumerian |
| Data migration | Copy essential artifacts | Self-contained, no dependency on heiroglyphy's version-folder layout |
| Script scope | Alignment pipeline only (06-10) | Corpus already built; scraping scripts can be backfilled later |
| Script numbering | Mirror Sumerian scheme | 1:1 correspondence across languages for cross-comparison |
| Abstraction | None yet (concrete first) | Extract shared base class in Phase gamma after two implementations exist |

## Target Directory Structure

```
languages/egyptian/
├── scripts/
│   ├── 06_extract_anchors.py      # Adapt TLA/Ramses anchor format to hyper-glyphy standard
│   ├── 07_train_fasttext.py       # FastText 768d skip-gram (identical params to Sumerian)
│   ├── 08_fuse_embeddings.py      # Zero-pad fusion [768d | 000...768d] -> 1536d
│   ├── 09_align_and_evaluate.py   # Ridge -> GloVe 300d (alpha=0.001)
│   ├── 09b_align_gemma.py         # Ridge -> whitened Gemma 768d (alpha=100, with sweep)
│   ├── 10_export_production.py    # Dual-view export (Gemma + GloVe)
│   └── egyptian_normalize.py      # Transliteration normalization
├── data/
│   ├── raw/                       # Copied from heiroglyphy (gitignored)
│   └── processed/                 # cleaned_corpus.txt, english_anchors.json (gitignored)
├── models/                        # FastText model, ridge weights (gitignored)
├── results/                       # Alignment results JSON (gitignored)
├── final_output/
│   ├── egyptian_aligned_vectors.npz        # GloVe-space (80,662 x 300, fp16)
│   ├── egyptian_aligned_gemma_vectors.npz  # Gemma-space (80,662 x 768, fp16)
│   ├── egyptian_aligned_vocab.pkl          # Shared vocab
│   ├── egyptian_lookup.py                  # EgyptianLookup class (dual-view API)
│   └── metadata.json                       # Accuracy metrics, config, schema
├── tests/
│   ├── test_06_anchors.py
│   ├── test_07_fasttext.py
│   ├── test_08_fusion.py
│   ├── test_09_alignment.py
│   ├── test_10_export.py
│   └── test_lookup.py
├── docs/
│   └── EXPERIMENT_JOURNAL.md
└── README.md
```

## Data Migration

Copy from heiroglyphy into `languages/egyptian/`:

| Source (heiroglyphy) | Destination (hyper-glyphy) | Notes |
|----------------------|---------------------------|-------|
| `heiro_v5_getdata/data/processed/cleaned_corpus.txt` | `data/processed/cleaned_corpus.txt` | 100,729 lines, 789K tokens |
| `heiro_v5_getdata/data/processed/english_anchors.json` | `data/processed/english_anchors.json` | 8,541 anchor pairs |
| `heiro_v15/models/fasttext_mc5_w10.vec` | `models/fasttext_egyptian.vec` | 768d, 10,833 vocab |
| `heiro_v15/models/fasttext_mc5_w10.model` | `models/fasttext_egyptian.model` | Full gensim model |

NOT copied:
- `visual_embeddings_768d.pkl` (dropping visuals)
- `gardiner_mapping.json` (not needed without visuals)
- GloVe (reused from existing location via computed path)

## Pipeline Design

### Script 06 -- Anchor Extraction

Reads the existing `english_anchors.json` from heiroglyphy and normalizes to hyper-glyphy's standard anchor format:

```json
{"egyptian": "nTr", "english": "god", "confidence": 0.87, "frequency": 234}
```

Applies quality filters matching Sumerian's conventions:
- Min frequency >= 5
- Exclude single-char English glosses
- Exclude numeric-only glosses
- Exclude named-entity clashes where identifiable

Output: `data/processed/english_anchors.json` (filtered, standardized format).

### Script 07 -- FastText Training

Identical hyperparameters to Sumerian:

| Parameter | Value |
|-----------|-------|
| vector_size | 768 |
| window | 10 |
| min_count | 5 |
| sg | 1 (skip-gram) |
| epochs | 10 |

Input: `data/processed/cleaned_corpus.txt`
Output: `models/fasttext_egyptian.model`, `models/fasttext_egyptian.vec`

The pre-trained model is copied from heiroglyphy so this script exists for reproducibility but doesn't need to run for initial results.

### Script 08 -- Fusion

Pure zero-padding concatenation:

```
[FastText 768d | zeros 768d] -> 1536d
```

Identical to Sumerian. No visual features.

Input: `models/fasttext_egyptian.vec`
Output: `models/fused_embeddings_1536d.npz`

### Script 09 -- GloVe Alignment

Ridge regression: fused 1536d -> GloVe 300d.

| Parameter | Value |
|-----------|-------|
| alpha | 0.001 |
| test_size | 0.2 |
| random_state | 42 |

Expected accuracy: ~32.35% top-1 (matching heiroglyphy V15, minus negligible 0.59% visual contribution).

Input: fused embeddings + anchors + GloVe (`languages/sumerian/data/processed/glove.6B.300d.txt`)
Output: `models/ridge_weights.npz`, `results/alignment_results.json`

### Script 09b -- Gemma Alignment

Ridge regression: fused 1536d -> whitened Gemma 768d.

| Parameter | Value |
|-----------|-------|
| alpha | 100 (starting point, matching Sumerian) |
| test_size | 0.2 |
| random_state | 42 |

Includes alpha sweep: `[0.01, 0.1, 1, 10, 100, 1000]` to find optimal alpha for Egyptian.

This is the NEW target -- heiroglyphy never had Gemma alignment.

Input: fused embeddings + anchors + `shared/models/english_gemma_whitened_768d.npz`
Output: `models/ridge_weights_gemma_whitened.npz`, `results/alignment_results_gemma_whitened.json`

### Script 10 -- Export

Projects all Egyptian vectors through both Ridge matrices. Dual-view output:

- `final_output/egyptian_aligned_vectors.npz` (GloVe-space, fp16)
- `final_output/egyptian_aligned_gemma_vectors.npz` (Gemma-space, fp16)
- `final_output/egyptian_aligned_vocab.pkl` (shared vocab)
- `final_output/metadata.json` (schema v2, both accuracy metrics)

Note: vocab serialized as pickle to match existing Sumerian pipeline convention (both SumerianLookup and heiroglyphy V15 use pickle for vocab).

### EgyptianLookup

Mirrors SumerianLookup API exactly:

- `find(english_word, top_k=10, space="gemma")` -> list of (egyptian_word, similarity)
- `find_both(english_word, top_k=10)` -> dict with "gemma" and "glove" keys
- `find_analogy(a, b, c, top_k=10, space="gemma")` -> vector arithmetic results

Lives at `final_output/egyptian_lookup.py`.

### egyptian_normalize.py

Handles Egyptian transliteration normalization (equivalent to `sumerian_normalize.py`). Strips diacritics variants, normalizes Unicode forms, handles Manuel de Codage conventions.

## Import Patterns

Same dual-root pattern as Sumerian scripts:

```python
_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # hyper-glyphy root
_LANG_ROOT = Path(__file__).parent.parent                 # languages/egyptian/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

## Shared Dependencies

- `shared/models/english_gemma_whitened_768d.npz` (already exists from Sumerian pipeline)
- GloVe at `languages/sumerian/data/processed/glove.6B.300d.txt` (accessed via computed path)

## Config Updates

- `pytest.ini`: add `languages/egyptian/tests` to testpaths
- `.gitignore`: already covers `languages/*/` patterns (no changes needed)

## Cross-Comparability Guarantees

- Same embedding dimensions at every stage (768 -> 1536 -> 300/768)
- Same Ridge parameters (alpha, test_size, random_state)
- Same evaluation methodology (top-K cosine nearest neighbor)
- Same export format (fp16 NPZ + pickle vocab + JSON metadata)
- Same metadata schema so comparative analysis can load both languages uniformly
- Same script numbering for 1:1 correspondence

## Test Infrastructure

Per-script test files matching Sumerian's pattern:
- Shape/dtype assertions on all intermediate artifacts
- Vocab consistency checks (no duplicates, correct count)
- Accuracy sanity thresholds (GloVe top-1 > 25%)
- Round-trip tests for export/load cycle
- EgyptianLookup API tests mirroring SumerianLookup tests

## Out of Scope (Phase gamma)

- Anomaly atlas for Egyptian
- Egyptian cosmogony case study
- Comparative cross-civilizational document
- Framework base class extraction (shared BaseLookup)
- Egyptian corpus-building scripts (01-05)
