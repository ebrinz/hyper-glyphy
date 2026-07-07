# languages/greek — Classical Greek Alignment

Cross-lingual embedding alignment for Classical Greek, mapping into both GloVe 300d
and whitened-EmbeddingGemma 768d English semantic spaces. Pipeline structure mirrors
`languages/akkadian/` and `languages/hittite/`.

## Status

Scaffold complete (G1–G4, commit cbb4e54, 2026-05-11). First alignment run deferred
pending eval redesign (see [repo journal](../../../../docs/EXPERIMENT_JOURNAL.md),
2026-07-06 entry).

## Corpus

- **Diorisis** (Figshare 12251468): 821 JSON files, Homer to 5th century AD.
  10,202,857 token-lemma records from 820 texts, 537,785 lines. ~10.2M tokens.
- **LSJ** (Perseus Digital Library `PerseusDL/lexica`): 27 XML files, 90,424 entries.
  LSJ join hit rate: 77% (7.76M hits / 2.30M misses).

## Anchors

106,260 anchors extracted (G4, commit cbb4e54). Lemma-group split applied (commit
59a8252): union-find groups anchors sharing a lemma, 64/16/20 train/val/test split,
alpha selected on validation.

## Pipeline scripts

| Script | Purpose |
|--------|---------|
| `01_parse_diorisis.py` | Parse Diorisis JSON → token-lemma records |
| `02_parse_lsj.py` | Parse Perseus LSJ XML → Greek-English gloss pairs |
| `04_deduplicate_corpus.py` | Deduplicate corpus lines |
| `05_clean_and_tokenize.py` | Normalize via `greek_normalize.py` (NFD, drop diacritics, lowercase) |
| `06_extract_anchors.py` | Join Diorisis lemmas with LSJ glosses → anchor pairs |
| `07_train_fasttext.py` | Train 768d FastText skip-gram embeddings |
| `08_fuse_embeddings.py` | Zero-pad fusion [768d | 000...768d] → 1536d |
| `09_align_and_evaluate.py` | Ridge regression → GloVe 300d |
| `09b_align_gemma.py` | Ridge regression → whitened Gemma 768d |

Note: `ridge_alpha_sweep.py` was retired in the 2026-07-06 eval-integrity fix; alpha
selection is now inline in `09`/`09b` using the validation split.

## Running

```bash
python languages/greek/scripts/01_parse_diorisis.py
python languages/greek/scripts/02_parse_lsj.py
python languages/greek/scripts/04_deduplicate_corpus.py
python languages/greek/scripts/05_clean_and_tokenize.py
python languages/greek/scripts/06_extract_anchors.py
python languages/greek/scripts/07_train_fasttext.py
python languages/greek/scripts/08_fuse_embeddings.py
python languages/greek/scripts/09_align_and_evaluate.py
python languages/greek/scripts/09b_align_gemma.py --mode whitened
```

## Tests

```bash
pytest languages/greek/tests/ -v
```
