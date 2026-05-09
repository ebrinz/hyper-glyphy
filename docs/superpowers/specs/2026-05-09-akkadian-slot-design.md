# Akkadian Slot Design Spec

## Goal

Add a third language slot, `languages/akkadian/`, parallel to existing Sumerian and Egyptian slots. Produce cross-comparable GloVe and whitened-Gemma alignments for Old Babylonian Akkadian, with the pipeline's structure mirroring Sumerian 1:1 to maximize code reuse and cross-language comparability. Scaffold the Sumerian↔Akkadian bridge data (DCCLT lexical lists) for a v2 cross-lingual validation experiment, but do not run the experiment in v1.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Corpus scope | Old Babylonian (~2000–1600 BCE) | Temporally commensurate with Sumerian corpus (ETCSL+CDLI dominated by OB attestation); enables clean bridge experiment via shared scribal tradition |
| Anchor lexicon | eBL primary + ORACC project glosses fallback | Mirrors Sumerian's ePSD2-primary/ORACC-fallback architecture; eBL is the actively-maintained, machine-readable Akkadian analog to ePSD2 |
| Bridge data | DCCLT scaffolded in v1, experiment deferred to v2 | Data is the hard part; scrape-and-parse cost is low; experiment can run once both alignments are stable |
| Pipeline structure | Mirror Sumerian's `01-10` numbered scripts | 1:1 correspondence with existing slots; ~70% direct code reuse |
| Project codes | Discover at scrape time via ORACC index | ORACC sub-project slugs change; query language=`akk` + period contains `Old Babylonian` rather than hardcode stale list |
| Token-budget contingency | If OB cleaned corpus < 500k tokens, flag SB fallback in v1 docs | Standard Babylonian preserves OB linguistic stratum for canonical first-millennium copies; usable as FastText pretrain supplement only, not anchor data |
| Success criterion | Floor: above-chance alignment. Target: ≥30% top-1 (Egyptian league). Stretch: ≥45% top-1 (approaching Sumerian) | Sumerian's Workstream 2b showed one normalization fix can move top-1 by 32pp; committing to a single number pre-diagnostic is premature |

## Target Directory Structure

```
languages/akkadian/
├── __init__.py
├── README.md
├── data/
│   ├── raw/
│   │   ├── ob_literary/                   # OB Atra-Hasis, OB Gilgamesh, Hammurabi, omens (gitignored)
│   │   ├── ob_letters/                    # Mari letters, ARCHIBAB-derived where ATF-available (gitignored)
│   │   └── dcclt/                         # Lexical lists (Sumerian↔Akkadian bridge data, gitignored)
│   ├── dictionaries/
│   │   ├── ebl_lemmas.json                # primary anchor source (gitignored)
│   │   └── oracc_lemmas.json              # fallback, auto-extracted from ATF stream (gitignored)
│   └── processed/
│       ├── corpus_clean.txt               # FastText training corpus (gitignored)
│       ├── anchors.jsonl                  # extracted anchor records (gitignored)
│       └── sumerian_akkadian_pairs.jsonl  # parsed DCCLT pairs, for v2 bridge (gitignored)
├── docs/
│   └── EXPERIMENT_JOURNAL.md
├── final_output/
│   ├── akkadian_aligned_vectors.npz       # GloVe-space (fp16)
│   ├── akkadian_aligned_gemma_vectors.npz # Gemma-space (fp16)
│   ├── akkadian_aligned_vocab.pkl
│   ├── akkadian_lookup.py                 # AkkadianLookup class (dual-view API)
│   └── metadata.json
├── models/                                # FastText model, ridge weights (gitignored)
├── results/                               # Alignment result JSONs (gitignored)
├── scripts/
│   ├── __init__.py
│   ├── 01_scrape_oracc_ob.py              # OB literary projects from ORACC
│   ├── 02_scrape_oracc_letters.py         # OB letters from ORACC
│   ├── 03_scrape_dcclt.py                 # Lexical lists; outputs corpus + parsed pairs
│   ├── 04_deduplicate_corpus.py           # Direct copy from Sumerian
│   ├── 05_clean_and_tokenize.py           # Fork + adapt for Akkadian normalization
│   ├── 06_extract_anchors.py              # Fork + adapt; eBL fetch + ORACC fallback
│   ├── 07_train_fasttext.py               # Direct copy from Sumerian (768d skip-gram)
│   ├── 08_fuse_embeddings.py              # Direct copy (zero-padding 768+768=1536)
│   ├── 09_align_and_evaluate.py           # Direct copy (ridge → GloVe 300d)
│   ├── 09b_align_gemma.py                 # Direct copy (ridge → whitened Gemma 768d)
│   ├── 10_export_production.py            # Fork + rename → AkkadianLookup
│   ├── akkadian_normalize.py              # NFC, mimation, logogram-syllabic alternation
│   └── coverage_diagnostic.py             # Fork from Sumerian; add `logogram_unmatched` bucket
└── tests/
    ├── test_06_anchors.py
    ├── test_07_fasttext.py
    ├── test_08_fusion.py
    ├── test_09_alignment.py
    ├── test_10_export.py
    └── test_lookup.py
```

## Pipeline Scripts

| Script | Status | Notes |
|---|---|---|
| `01_scrape_oracc_ob.py` | new, mostly copied from Sumerian's `03_scrape_oracc.py` | OB literary ORACC sub-projects (eBL projects, OB omens, Hammurabi). Reuses ATF-fetch logic. |
| `02_scrape_oracc_letters.py` | new, mostly copied | OB letters where ORACC has ATF. |
| `03_scrape_dcclt.py` | new | Pulls DCCLT (`dcclt` ORACC project). Outputs running text to `data/raw/dcclt/` AND parsed `sumerian_akkadian_pairs.jsonl`. |
| `04_deduplicate_corpus.py` | direct copy | Text-agnostic. |
| `05_clean_and_tokenize.py` | fork + adapt | Akkadian-specific normalization (š/ḫ/ṣ/ṭ Semitic transcription, logogram-vs-syllabic readings, determinative handling). |
| `06_extract_anchors.py` | fork + adapt | Fetches eBL lemma list (new fetcher), falls back to ORACC project glosses. |
| `07_train_fasttext.py` | direct copy | 768d skip-gram, identical hyperparameters as starting point. |
| `08_fuse_embeddings.py` | direct copy | Zero-padding fusion (768 + 768 = 1536). |
| `09_align_and_evaluate.py` | direct copy | GloVe ridge alignment + top-k eval. |
| `09b_align_gemma.py` | direct copy | Whitened-Gemma ridge alignment (matching the Sumerian 52% pattern). |
| `10_export_production.py` | fork + rename | Produces `AkkadianLookup` class with same dual-view API. |
| `akkadian_normalize.py` | new | Akkadian-specific normalization helpers (analog to `sumerian_normalize.py` and `egyptian_normalize.py`). |
| `coverage_diagnostic.py` | fork from Sumerian | Adds `logogram_unmatched` bucket for the Akkadian-specific dual-encoding problem. |

Estimated reuse: ~70% direct copy from Sumerian, two scripts genuinely new (DCCLT scraper, eBL anchor fetcher), three scripts forked-and-adapted (clean/tokenize, anchor extraction, coverage diagnostic).

## Corpus Sources

ORACC's OB Akkadian content is split across multiple sub-projects whose codes shift over time. Rather than hardcoding slugs that may be stale, `01_scrape_oracc_ob.py` enumerates ORACC's project index and filters by `language=akk` and period containing `Old Babylonian`.

**Target text list for v1, by content:**

- **Code of Hammurabi** — canonical OB literary benchmark
- **OB Atra-Hasis** — Old Babylonian flood epic
- **OB Gilgamesh** — Pennsylvania, Yale, and other OB tablets
- **OB omens** — *šumma izbu*, *šumma ālu* OB recensions, extispicy
- **OB hymns and prayers** — Ishtar hymn, OB liturgical fragments
- **Mari letters** — where ATF-available
- **OB mathematical texts** — DCCMT (`dccmt` ORACC project, separate from DCCLT)
- **OB lexical lists** — routed to `data/raw/dcclt/` via the dedicated DCCLT scraper (not the OB-literary scraper)

**Bridge corpus**: DCCLT (`dcclt` project on ORACC) — confirmed live, ATF format. Parser pulls running text into `data/raw/dcclt/` and the structured Sumerian-Akkadian column pairs into `data/processed/sumerian_akkadian_pairs.jsonl`.

**Anchor lexicon**: eBL's API at `https://www.ebl.lmu.de/api/dictionary` returns JSON lemmas with period flags, glosses, and CDA-derived English translations. Fetcher caches to `data/dictionaries/ebl_lemmas.json`. ORACC project glosses (extracted from ATF as the existing Sumerian pipeline does) populate `data/dictionaries/oracc_lemmas.json` as fallback.

## Anchor Extraction

**Output schema**: matches Sumerian's `anchors.jsonl` — one `{lemma, gloss, source}` record per line.

**Filtering**: pull eBL lemmas where the period flag contains `Old Babylonian`. Drop lemmas with no English gloss (German-only entries logged as fallback candidates, not included in v1).

**Akkadian-specific normalization** (the place we will fight, by analogy to Sumerian's Workstream 2b):

- **Mimation**: OB lemmas are typically cited *with* mimation (`šarrum`); attested OB corpus tokens also have mimation, so keep it as the primary form. Add an alternation step trying the non-mimation form (`šarru`) as a fallback match.
- **Logogram ↔ syllabic equivalence**: Akkadian text encodes the same word two ways (`LUGAL` = `šar-ru-um`). eBL's `forms` field maps both — extractor registers both surface variants under one lemma. Structurally distinct from Sumerian (which mostly lacks this duality) and is the Akkadian-specific complexity.
- **Determinatives in `{braces}`**: same handling as Sumerian — strip during matching, preserve in raw store.
- **Diacritic encoding**: precomposed `š/ḫ/ṣ/ṭ` vs decomposed `s+́` etc. Run NFC normalization on both anchor and corpus side from day one (analog of Sumerian's 2b fix; Akkadian has more diacritics, so the pre-fix gap is likely larger).

**Coverage diagnostic from day one.** Ship `coverage_diagnostic.py` adapted from Sumerian's, with buckets: `valid_anchor`, `akkadian_vocab_miss`, `gloss_miss`, `normalization_recoverable`, `logogram_unmatched`. The last bucket is Akkadian-specific and is the early-warning signal for the logogram/syllabic dual-encoding problem.

**Anchor target size**: eBL has roughly 10–15k lemmas with OB attestation. After corpus-presence filtering and dropping no-gloss entries, expect 5–8k usable v1 anchors. Smaller than Sumerian's 13k, in the same ballpark as Egyptian's 5k — workable.

## v1 Deliverable

**v1 ships when:**

1. Pipeline runs end-to-end, producing `final_output/akkadian_aligned_gemma_vectors.npz` and an `AkkadianLookup` class with the same `.lookup(word, k=10)` interface as `SumerianLookup`.
2. Coverage diagnostic and top-k accuracy numbers are committed to both `languages/akkadian/docs/EXPERIMENT_JOURNAL.md` and the project-level journal.
3. DCCLT scraping is functional and `data/processed/sumerian_akkadian_pairs.jsonl` exists with parsed pairs — but no cross-lingual experiment is run.

**Success criterion (whitened-Gemma top-1):**

- **Floor**: alignment is meaningfully above chance.
- **Target**: ≥30% top-1, putting Akkadian in the same league as Egyptian (32%).
- **Stretch**: ≥45% top-1, approaching Sumerian's 52%.

No single point estimate is committed — Sumerian's Workstream 2b showed one normalization fix can move top-1 by 32pp, and committing to a number pre-diagnostic is premature.

## Out of v1 Scope

- Cross-lingual bridge experiment (DCCLT data extracted and parsed, but not used for cross-validation; deferred to v2).
- Standard Babylonian fallback execution (only triggers if cleaned OB corpus < 500k tokens; v1 only flags the contingency, doesn't run it).
- Diachronic OB → Classical Akkadian comparison.
- Production exports beyond the `AkkadianLookup` class (no PDF research artifact analogous to the Sumerian cosmogony document).
- Any abstraction extraction across the three slots (Sumerian, Egyptian, Akkadian) — concrete-first; abstract once a fourth slot proves the shape.

## Sequencing

Six logical phases, each independently committable:

1. **Scaffold** directory structure and copy unchanged scripts (`04`, `07`, `08`, `09`, `09b`).
2. **Scrape** scripts `01_scrape_oracc_ob.py`, `02_scrape_oracc_letters.py`, `03_scrape_dcclt.py` — bulk of the new code.
3. **Clean/tokenize** `05_clean_and_tokenize.py` with Akkadian normalization (mimation + logogram + NFC) and `akkadian_normalize.py` helper module.
4. **Anchors**: eBL fetcher, `06_extract_anchors.py` adaptation, and `coverage_diagnostic.py` with the `logogram_unmatched` bucket.
5. **Train/fuse/align**: run `07`-`09b` (mostly verify-and-tune; ridge alpha sweep if needed).
6. **Export & document**: `10_export_production.py` produces `AkkadianLookup`, write journal entry, commit final artifacts.
