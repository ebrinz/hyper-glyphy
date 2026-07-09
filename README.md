<p align="center">
  <img src="assets/banner.svg" alt="hyper-glyphy" width="100%"/>
</p>

<p align="center">
  <strong>Cross-lingual embedding alignment for ancient languages — five language slots, one shared pipeline</strong>
</p>

<p align="center">
  <a href="#results">Results</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#usage">Usage</a> &bull;
  <a href="#running-the-pipeline">Pipeline</a> &bull;
  <a href="#data-sources">Data Sources</a>
</p>

---

## Results

> Numbers before 2026-07-06 measured surface-variant memorization, not translation — see the [experiment journal](docs/EXPERIMENT_JOURNAL.md), 2026-07-06 entry.

The table below reports the **eval-redesign suite** (stratified CSLS, 50K candidates, lemma-group split): three regimes per slot — **dictionary** (in-sample: a fixed 1,000-anchor sample of the training set, measuring memorization of known glosses), **interpolation** (test anchors — always unseen lemmas — whose gold English gloss appears as a training target), **zero-shot** (test anchors whose gold gloss was never a training target) — plus a document-level panel (Sumerian ETCSL genre LOO and Hittite→Greek parallel retrieval). Gemma beats GloVe combined in 4 of 5 slots; Akkadian Gemma is anomalous (alpha-selection noise at near-zero val signal — see journal). Hittite zero-shot n is small (only 31% of test gold glosses are in-vocab of the 50K candidate set).

| Slot | Target | alpha | Dict top-1 | Interp top-1 | Zero-shot top-1 | Combined top-1 | Combined syn | Family / Script |
|------|--------|-------|:----------:|:------------:|:---------------:|:--------------:|:------------:|:----------------|
| [Sumerian](languages/sumerian/) | GloVe | 100 | 70.79% | 7.30% | 0.88% | 4.55% | 5.01% | language isolate / cuneiform |
| [Sumerian](languages/sumerian/) | Gemma | 1000 | 74.32% | 9.43% | 0.35% | 5.54% | 5.69% | language isolate / cuneiform |
| [Egyptian](languages/egyptian/) | GloVe | 1e-4 | 79.20% | 18.93% | 0.95% | 14.81% | 15.69% | Afroasiatic / hieroglyphic |
| [Egyptian](languages/egyptian/) | Gemma | 10 | 42.82% | 20.06% | 0.00% | 15.47% | 16.12% | Afroasiatic / hieroglyphic |
| [Akkadian](languages/akkadian/) | GloVe | 0.1 | 48.54% | 0.00% | 0.38% | 0.31% | 1.13% | East Semitic / cuneiform |
| [Akkadian](languages/akkadian/) | Gemma | 1e4 ¹ | 19.85% | 0.46% | 0.05% | 0.13% | 0.48% | East Semitic / cuneiform |
| [Hittite](languages/hittite/) | GloVe | 1e-4 | 63.36% | 6.77% | 0.00% ² | 4.39% | 4.39% | Indo-European (Anatolian) / cuneiform |
| [Hittite](languages/hittite/) | Gemma | 1e-4 | 79.61% | 10.53% | 0.00% ² | 6.83% | 8.05% | Indo-European (Anatolian) / cuneiform |
| [Greek](languages/greek/) | GloVe | 1e-4 | 39.05% | 4.06% | 0.40% | 3.31% | 5.67% | Indo-European / alphabetic |
| [Greek](languages/greek/) | Gemma | 0.1 | 52.49% | 5.33% | 0.68% | 4.37% | 7.42% | Indo-European / alphabetic |

¹ Akkadian Gemma alpha=1e4 is the grid ceiling (val-selection noise at ~0.06%); see journal 2026-07-09 entry.
² Hittite zero-shot n=144; 69% of test gold glosses are OOV of the 50K candidate vocab; see journal 2026-07-09 entry.

**Document-level panel.** Sumerian ETCSL genre leave-one-out (n=338, 5 genres, majority baseline 40.83%): gemma_aligned 63.31% (+22.5 pp), glove_aligned 60.95% (+20.1 pp), fused_unaligned 68.05% (+27.2 pp, projection cost ≈5 pp). Gate 1 PASS. Cross-language parallel retrieval (Hittite → Greek, pool=820): Kumarbi→Theogony rank 731/820, Illuyanka→Theogony 781/820, Ullikummi→Theogony 788/820, MRR 0.0013. Gate 2 FAIL — word-level alignment does not compose into cross-slot document retrieval; myth study proceeds via Plane B (native-space RSA). See [`docs/myth_study_plan.md`](docs/myth_study_plan.md) for the two-plane study design and go/no-go verdicts.

Both alignment targets are accessible via per-language `Lookup` classes (`space="gemma"|"glove"`).

### Research progress

Active experiment log: [`docs/EXPERIMENT_JOURNAL.md`](docs/EXPERIMENT_JOURNAL.md). Sumerian-specific historical findings: [`languages/sumerian/docs/EXPERIMENT_JOURNAL.md`](languages/sumerian/docs/EXPERIMENT_JOURNAL.md). Recent findings (newest first):

- **2026-07-09 — Eval redesign shipped: stratified CSLS suite + document-level panel; all five slots measured (Greek first run).** Three-stratum eval (dictionary/interpolation/zero-shot), CSLS retrieval, 50K candidates, leak-check 0.00% all five slots. Genre LOO PASS (gemma_aligned 63.31% vs 40.83% majority). Parallel retrieval FAIL (MRR 0.0013; myth study routes to Plane B native-space RSA). See journal 2026-07-09 entry and [`docs/myth_study_plan.md`](docs/myth_study_plan.md).
- **2026-07-06 — Eval integrity: lemma-group split + validation-selected alpha; all prior headline numbers invalidated.** Surface-variant train/test leakage (32–65% of test items had a same-gloss train anchor within edit distance 1) and test-set alpha tuning inflated every slot's accuracy. Fixed across all five language pipelines via a shared union-find group split (64/16/20 train/val/test) with alpha selected on validation. Akkadian rerun as evidence: GloVe 27.79% → 0.09%, whitened-Gemma 36.43% → 0.14% top-1; a three-way diagnostic (44.4% train-set accuracy, exact reproduction of the old number under the old split, 16.3% seen-gloss test rate) confirms the collapse is real zero-shot difficulty, not a bug. Remaining reruns deferred pending an eval redesign (seen/unseen strata, CSLS, restricted candidate vocab, document-level evaluation). See the journal entry for the full writeup.
- **2026-05-11 — Greek scaffold (G1–G4): parser, LSJ glosses, clean, anchors, FastText + alignment scripts ready; first run pending eval redesign.** 106,260 anchors, 10.2M-token Diorisis corpus (Homer to 5th c. AD). See [`languages/greek/`](languages/greek/).
- **2026-05-11 — Hittite v1 shipped:** Gemma top-1 40.62% (pre-fix), beats Akkadian's v1.3 from day one. TLHdig corpus (Zenodo), German glosses translated via multilingual EmbeddingGemma. See [`languages/hittite/docs/EXPERIMENT_JOURNAL.md`](languages/hittite/docs/EXPERIMENT_JOURNAL.md).
- **2026-05-11 — Akkadian v1.3:** Ridge alpha sweep (+7.41pp top-1, 29.02% → 36.43% Gemma). See [`languages/akkadian/docs/EXPERIMENT_JOURNAL.md`](languages/akkadian/docs/EXPERIMENT_JOURNAL.md).
- **2026-05-10 — Akkadian v1.1–v1.2:** Gap-closing iterations (+4.91pp + 7.36pp top-1). See slot journal.
- **2026-05-09 — Akkadian slot v1 shipped:** Third language slot, Gemma top-1 16.75% (pre-fix). See [`languages/akkadian/docs/EXPERIMENT_JOURNAL.md`](languages/akkadian/docs/EXPERIMENT_JOURNAL.md).
- **2026-04-20 — Anomaly Atlas Interpretive Findings:** Standalone ~9,500-word document + PDF with embedded cuneiform font, surfacing 15-20 atlas findings across six themes. See [`languages/sumerian/docs/anomaly_atlas_findings.md`](languages/sumerian/docs/anomaly_atlas_findings.md) (markdown) / [`languages/sumerian/docs/anomaly_atlas_findings.pdf`](languages/sumerian/docs/anomaly_atlas_findings.pdf) (PDF with cuneiform).
- **2026-04-19 — Sumerian Cosmogony document:** A methodology-driven ~14,000-word case study on the Anunnaki cosmogonic cycle, using the 52%-top-1 whitened-Gemma alignment for geometric translation of five pivotal terms (`abzu`, `zi`, `nam`, `namtar`, `me`). See [`languages/sumerian/docs/sumerian_cosmogony.md`](languages/sumerian/docs/sumerian_cosmogony.md).
- **2026-04-19 — Workstream 2b (STRETCH tier shipped):** Normalization fix landed. Whitened-Gemma top-1 **19.85% → 52.13% (+32.28pp)**. Training anchors 1,572 → 6,867. Coverage diagnostic's `normalization_recoverable` bucket cleared from 7,651 to 0. The 2b-pre diagnostic's attribution held to the bit — a ~20-line unicode-normalization fix delivered the largest single top-1 gain in the project's history.
- **2026-04-19 — Workstream 2b-pre:** Coverage diagnostic attributed 64.85% of the 11,798 `sumerian_vocab_miss` anchors to a simple ASCII-normalization gap between the anchor extractor and the corpus tokenizer (subscripts → ASCII, strip determinative braces, drop hyphens). Inference-based alternatives (FastText subword inference, morpheme composition) scored 10.7% and 1.8% Tier-2 top-5 accuracy respectively — not the next lever to pull.
- **2026-04-18 — Workstream 2a:** Anchor audit baselined valid-anchor survival at 14.05% (1,951/13,886). 84.96% of all dropout is `sumerian_vocab_miss`; every other bucket combined is under 1%.
- **2026-04-16 — Phase B:** Dual-view Sumerian lookup. Whitened EmbeddingGemma and GloVe now coexist as parallel alignment targets; downstream code toggles via `space="gemma"|"glove"`.
- **2026-04-16 — Phase A retry #2:** BERT-whitening (Su et al. 2021) applied to the EmbeddingGemma target unlocked +2.54pp top-1 over GloVe. Centering + whitening is mandatory for any contextual-encoder alignment target.

## How It Works

```
Sumerian corpus (ETCSL + CDLI + ORACC)
        |
  ATF cleaning & tokenization
        |
  FastText skip-gram (768d)
        |
  Zero-padding fusion (768d + 768d = 1536d)
        |
  Ridge Regression ──┬── whitened-EmbeddingGemma 768d ── Nearest-neighbor retrieval
                     └── GloVe 300d                    ── Nearest-neighbor retrieval
```

The approach follows a cross-lingual embedding alignment strategy with a dual target:

1. **Train monolingual embeddings** on a large Sumerian corpus using FastText.
2. **Fuse** text embeddings with zero-padding (dimensionality regularization).
3. **Learn a linear mapping** from the fused Sumerian space into both (a) whitened EmbeddingGemma 768d (primary target) and (b) GloVe 300d (secondary target), using anchor word pairs from ePSD2 and ETCSL co-occurrence.
4. **Evaluate** by checking if the nearest English neighbor of a projected Sumerian vector is the correct translation; both target spaces are queryable via one `SumerianLookup` class.

## Usage

```python
from languages.sumerian.final_output.sumerian_lookup import SumerianLookup

lookup = SumerianLookup(
    gemma_vectors_path="languages/sumerian/final_output/sumerian_aligned_gemma_vectors.npz",
    glove_vectors_path="languages/sumerian/final_output/sumerian_aligned_vectors.npz",
    vocab_path="languages/sumerian/final_output/sumerian_aligned_vocab.pkl",
    gemma_english_path="shared/models/english_gemma_whitened_768d.npz",
    glove_english_vectors=glove_vectors,
    glove_english_vocab=glove_vocab,
)

# Default space is the whitened-Gemma 768d manifold:
lookup.find("king")                     # -> [("ul3", 0.67), ("asal", 0.51), ...]

# Query the GloVe 300d manifold:
lookup.find("king", space="glove")      # -> [("ul3", 0.68), ("se2", 0.60), ...]

# Both spaces at once:
lookup.find_both("fate")                # -> {"gemma": [...], "glove": [...]}

# Vector analogy in either space:
lookup.find_analogy("king", "queen", "god", space="gemma")

# Weighted blend of concepts:
lookup.find_blend({"sun": 0.7, "power": 0.3}, space="gemma")
```

## Running the Pipeline

### Prerequisites

```bash
pip install -r requirements.txt
```

### Full Pipeline

```bash
# 1. Scrape corpora (ETCSL ~5MB, CDLI ~230MB, ORACC ~700MB)
python languages/sumerian/scripts/01_scrape_etcsl.py
python languages/sumerian/scripts/02_scrape_cdli.py
python languages/sumerian/scripts/03_scrape_oracc.py

# 2. Process corpus
python languages/sumerian/scripts/04_deduplicate_corpus.py
python languages/sumerian/scripts/05_clean_and_tokenize.py
python languages/sumerian/scripts/06_extract_anchors.py

# 3. Download GloVe (862MB, or symlinks from heiroglyphy)
python shared/scripts/download_glove.py

# 4. Train and evaluate
python languages/sumerian/scripts/07_train_fasttext.py     # ~30-60 min
python languages/sumerian/scripts/08_fuse_embeddings.py
python languages/sumerian/scripts/09_align_and_evaluate.py
python languages/sumerian/scripts/10_export_production.py
```

### Tests

```bash
pytest languages/sumerian/tests/ shared/tests/ --ignore=languages/sumerian/tests/test_integration.py -v
```

## Data Sources

| Source | Content | Size |
|--------|---------|------|
| [ETCSL](https://etcsl.orinst.ox.ac.uk/) | Sumerian literary compositions with English translations | 36K lines, 35K with translations |
| [CDLI](https://cdli.ucla.edu/) | Bulk Sumerian transliterations (ATF format) | 96K texts, 1.4M lines |
| [ORACC](https://oracc.museum.upenn.edu/) | Lemmatized Sumerian with English glosses | 90K texts, 4.3M lemmas, 2.8K unique glosses |
| [GloVe 6B](https://nlp.stanford.edu/projects/glove/) | Pre-trained English word vectors | 400K words, 300d |

## Architecture

Originated as a port of [heiroglyphy](https://github.com/ebrinz/heiroglyphy) V15 to Sumerian; has since diverged in significant ways (dual-target alignment, whitening, normalization fix, anomaly atlas):

| Parameter | Value |
|-----------|-------|
| FastText dimensions | 768 |
| FastText window | 10 |
| FastText min_count | 5 |
| FastText algorithm | skip-gram |
| Fusion | 768d text + 768d zero-padding → 1,536d |
| Alignment | Ridge Regression (one per target) |
| Ridge alpha (GloVe) | 0.001 (post-Workstream 2b; pre-2b was 100 due to an underdetermined system) |
| Ridge alpha (whitened-Gemma) | 100 |
| Target spaces | GloVe 6B 300d + whitened EmbeddingGemma 768d |
| Training anchors | 6,867 (post-Workstream 2b; was 1,572 pre-fix) |

### Key findings

- **Normalization drift was the dominant blocker.** Workstream 2b found that 64.85% of anchor-dropout was a ~20-line unicode-normalization gap between ORACC citation forms and the ATF corpus (subscripts → ASCII, strip determinatives, drop hyphens). Fixing it 3×-multiplied top-1 on whitened Gemma (19.85% → 52.13%). See [`languages/sumerian/docs/EXPERIMENT_JOURNAL.md`](languages/sumerian/docs/EXPERIMENT_JOURNAL.md).
- **Whitening is mandatory for contextual-encoder targets.** BERT-whitening (Su et al. 2021) applied to raw EmbeddingGemma is the difference between an alignment that works and one that fails the baseline.
- **Atlas-driven anomaly analysis** surfaces specific Sumerian words where the alignment's geometry genuinely diverges from English translation conventions — see [`languages/sumerian/docs/anomaly_atlas_findings.md`](languages/sumerian/docs/anomaly_atlas_findings.md) and its [PDF rendering](languages/sumerian/docs/anomaly_atlas_findings.pdf).

## Project Structure

```
hyper-glyphy/
├── languages/
│   ├── sumerian/          # language isolate / cuneiform — shipped (pre-fix)
│   │   ├── scripts/       # Pipeline 01-10 + analysis/ + normalization + audit tools
│   │   ├── docs/          # EXPERIMENT_JOURNAL, anomaly atlas, cosmogony (Sumerian-specific)
│   │   ├── data/          # Raw and processed data (gitignored)
│   │   ├── models/        # Trained FastText caches (gitignored)
│   │   ├── results/       # Audit + diagnostic reports (committed)
│   │   ├── final_output/  # Production aligned vectors + SumerianLookup API
│   │   └── tests/
│   ├── egyptian/          # Afroasiatic / hieroglyphic — shipped (pre-fix)
│   ├── akkadian/          # East Semitic / cuneiform — shipped (pre-fix)
│   ├── hittite/           # Indo-European (Anatolian) / cuneiform — shipped (pre-fix)
│   └── greek/             # Indo-European / alphabetic — scaffold complete, run pending
├── framework/
│   └── analysis/          # Language-agnostic anomaly atlas + analysis lenses
├── shared/
│   ├── scripts/           # English target embedding tools (Gemma, GloVe, whitening)
│   │   └── anchor_split.py  # Union-find lemma-group split (post eval-integrity fix)
│   ├── models/            # Whitened-Gemma caches (gitignored)
│   └── tests/
└── docs/                  # Repo-wide journal, roadmap, research artifacts
    ├── ROADMAP.md
    ├── EXPERIMENT_JOURNAL.md   # Cross-language experiment log (newest first)
    └── RESEARCH_VISION.md
```

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for queued workstreams.

### Currently shipped (all numbers pre-fix — see Results banner above)

| Slot | Gemma top-1 | GloVe top-1 | Family / Script |
|------|:-----------:|:-----------:|:----------------|
| [Sumerian](languages/sumerian/) | 52.13% | 35.70% | language isolate / cuneiform |
| [Egyptian](languages/egyptian/) | 34.57% | 33.42% | Afroasiatic / hieroglyphic |
| [Akkadian](languages/akkadian/) | 36.43% | 27.79% | East Semitic / cuneiform |
| [Hittite](languages/hittite/) | 40.62% | 35.40% | Indo-European (Anatolian) / cuneiform |
| [Greek](languages/greek/) | — | — | Indo-European / alphabetic — scaffold complete, first run pending |

### Future language slots

Targeting a long-term roster of cross-comparable ancient-language alignments, each
using the same dual-target Ridge pipeline (GloVe 300d + whitened-Gemma 768d):

| Slot | Period | Family / Script | Why it's interesting |
|------|--------|-----------------|----------------------|
| **Greek** | Archaic → Koine, ~800 BCE – 600 CE | Indo-European / alphabetic | Massive corpus (Perseus, TLG); a high-quality calibration slot where modern semantic encoders already perform well — sanity-check for the pipeline. |
| **Ugaritic** | ~1400–1200 BCE | NW Semitic / alphabetic cuneiform | Tiny corpus but liturgically dense (Baal Cycle, KTU); sister branch to Akkadian, enables NW↔East Semitic alignment comparison. |
| **Elamite** | ~2300 BCE – 5th c. BCE | language isolate / cuneiform (partially deciphered) | The "embedding archaeology" case — most ambitious slot. Even partial alignment would be a research finding. |
| **Vedic** | ~1500–500 BCE (oral) | Old Indo-Aryan / Devanagari (late manuscripts) | Closest living-tradition ancient corpus; Grassmann's *Wörterbuch zum Rig-Veda* gives a clean anchor lexicon; pairs with Hittite for IE comparison. |
| **Aramaic** | ~1100 BCE – present (target: Imperial + Biblical) | NW Semitic / alphabetic (square script + variants) | Bridge language across the Persian/Hellenistic/Roman Near East; massive epigraphic corpus from Persepolis to the Cairo Geniza. |

The DCCLT bridge data (50k Sumerian↔Akkadian pairs) and similar bilingual lexical
traditions (Hittite-Akkadian, Ugaritic-Akkadian, Aramaic-Akkadian) make
cross-lingual validation experiments a recurring possibility across these slots.

## License

MIT
