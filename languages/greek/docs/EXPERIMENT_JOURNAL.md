# Greek Experiment Journal

## 2026-05-11 — G1–G4 scaffold: parser, LSJ glosses, clean, anchors (commit cbb4e54)

**Spec:** no formal design doc; followed the Hittite framework with Greek-specific deltas.

**Corpus:** Diorisis (Figshare 12251468, CC-BY) — 821 JSON files, Homer to 5th c. AD.
820 texts, 537,785 lines, 10,202,857 token-lemma records. ~10.2M tokens.

**Glosses:** Perseus LSJ XML (`PerseusDL/lexica`), 27 files, 90,424 entries. Translation
extraction via `<tr>` tags; first content word in `english_gemma_768d` vocab after
stripping stop words and hyphens.

**LSJ join hit rate:** 77% (7.76M hits / 2.30M misses).

**Anchors:** 106,260 total (8× Hittite, 4× Akkadian).

**Normalization (`greek_normalize.py`):** NFD decomposition, drop polytonic combining
marks (accents, breathings, iota subscript, diaeresis), final sigma normalization,
lowercase. 10/10 tests pass.

**Scripts added:** `01_parse_diorisis.py`, `02_parse_lsj.py`, `04_deduplicate_corpus.py`,
`05_clean_and_tokenize.py`, `06_extract_anchors.py`, `07_train_fasttext.py`,
`08_fuse_embeddings.py`, `09_align_and_evaluate.py`, `09b_align_gemma.py`,
`greek_normalize.py`, plus audit/coverage/align helpers copied from Hittite.

**Commits:** `cbb4e54` (G1–G4 scaffold), `3550cbb` (anchors carry contributing lemmas
for group split), `59a8252` (lemma-group split + val-selected alpha scripts).

**Status:** first alignment run deferred pending eval redesign. See repo journal
[`docs/EXPERIMENT_JOURNAL.md`](../../../../docs/EXPERIMENT_JOURNAL.md), 2026-07-06 entry.
