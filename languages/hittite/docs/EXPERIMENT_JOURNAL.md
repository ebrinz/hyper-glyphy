# Hittite Experiment Journal

## 2026-05-11 — v1 ship: TLHdig Hittite aligned to whitened-Gemma 768d

**Spec:** brief (no formal design doc; followed the Akkadian framework with deltas
captured in this entry)
**Source corpus:** TLHdig 0.2.0-beta from Zenodo (record 15459134, CC-BY)
**Lessons applied:** all 7 levers from Akkadian's v1.0→v1.3 arc baked into v1.

### Headline numbers (whitened-Gemma 768d primary, GloVe 300d secondary)

| Metric | Hittite v1 | Akkadian v1.3 | Sumerian | Egyptian |
|--------|:---:|:---:|:---:|:---:|
| Top-1 (Gemma) | **40.62%** | 36.43% | 52.13% | n/a |
| Top-5 (Gemma) | 50.99% | 59.18% | 61.97% | n/a |
| Top-10 (Gemma) | 55.02% | 66.51% | 65.99% | n/a |
| Top-1 (GloVe) | 35.40% | 27.79% | 35.70% | 32.35% |
| Top-5 (GloVe) | 41.88% | 43.23% | 44.61% | 41.47% |
| Top-10 (GloVe) | 43.20% | 47.52% | 47.93% | 45.13% |
| FastText vocab | 31,412 | 45,769 | — | — |
| Corpus tokens | 591k | 3,000k | 2,800k | 789k |
| Anchors (training set) | 6,056 | 8,710 | 6,867 | 5,360 |

**Hittite Gemma top-1 (40.62%) beats Akkadian's v1.3 final (36.43%) from
day one,** despite ~20% the corpus and ~50% the anchor pool. GloVe top-1
(35.40%) ~matches Sumerian (35.70%) and beats Egyptian and Akkadian.

### What's different about Hittite vs the Mesopotamian slots

**Source corpus: TLHdig, not ORACC.** Hittite scholarship lives at the
Hethitologie Portal Mainz, not ORACC. TLHdig publishes their XML on Zenodo:
22k texts, lemmatized with full morphological analysis, CC-BY. Far richer
than ORACC's per-word `f` records — each TLHdig word carries up to 5
alternative parses in `mrp1..mrp5` attributes with format
`LEMMA@GLOSS@FEATURES@CATEGORY@`.

**Glosses are GERMAN, not English.** TLHdig is a German-tradition project;
the gloss field is German. Akkadian's `gw` was English, so we could feed
it directly to the anchor extractor. Hittite needs a translation step.

**The translation strategy: multilingual Gemma + English wordlist filter.**
For each unique German gloss meeting the >=5 occurrence threshold:
1. Encode with EmbeddingGemma (multilingual) → 768d vector
2. Find nearest English Gemma vocab entry, restricted to NLTK English words
3. Use that English word as the anchor's `english` field

This avoids any explicit German-English dictionary. The English-word filter
is necessary because the Gemma vocab cache was built from GloVe 400k vocab,
which contains many non-English tokens (German, Dutch, etc.). Without it,
"Gott" maps to "Gott" itself. Even with NLTK filtering, translation is
imperfect (`König → kral`, `Brot → brot`, both spuriously in NLTK words).
The Ridge regression nevertheless learns a strong alignment — the SEMANTIC
target vector lands meaningfully in Gemma space even when the literal
English label is weird.

**Heterograms as bonus anchors.** Hittite scribes wrote in trilingual mode
within the same line: Sumerograms (`LUGAL`), Akkadograms (`ŠARRU`),
determinatives (`<d>D</d>`), plus phonetic Hittite. The TLHdig XML tags
all three with `<sGr>`, `<aGr>`, `<d>`. We extract heterograms (4,081
Sumerograms, 2,829 Akkadograms) and bridge them via our existing
Sumerian/Akkadian aligned-Gemma spaces — for each heterogram occurring
frequently in TLHdig, look it up in the source-language aligned vector,
take the top-1 English neighbor (gated by cosine >= 0.5).

Result: 48 Sumerogram bridge anchors + 35 Akkadogram bridge anchors,
modest absolute count but free-with-the-data and theoretically clean
(scribes themselves tagged these as Sumerian/Akkadian equivalences).

### Lessons applied from Akkadian's 7-iteration arc

1. **Alpha sweep on day one.** `ridge_alpha_sweep.py` was scaffolded and
   run BEFORE declaring v1 numbers. Found α=0.01 optimal (matches
   Akkadian's optimum). Without this we'd have shipped at α=100 and
   reported ~28% top-1 — would've spent days iterating before discovering
   the lever.

2. **No speculative dictionary fetcher.** The German→English translation
   step was bounded to "use Gemma + a wordlist," not "scrape CHD/HED."
   Akkadian's L6b (eBL pivot) and L6b (DCCLT bridge) experiments showed
   speculative cross-language data sources rarely pay off.

3. **Subword inference at eval (train-only OOV partition).** Inherited
   from Akkadian's L5-refined: OOV anchors are training-only, test set
   drawn from in-vocab anchors only. 4,180 OOV anchors expanded training
   beyond the 7,570 in-vocab pool.

4. **Lemma-surface expansion in 06_extract_anchors.** Akkadian's L4
   global-surface-map approach is applied implicitly: each lemma's `cf`
   and `form` both get registered as anchors with the gloss's translated
   English target.

### Open questions for v1.1+

- **German→English translation quality.** NLTK words contains many archaic
  / borrowed entries (`brot`, `tempe`, `kral`). A frequency-curated English
  wordlist (top-10k from COCA, or SCOWL English) would likely lift top-1
  by +2-5pp.
- **Per-anchor Gemma encoding instead of vocab lookup.** Skip the
  "translate to English word" step entirely; use the Gemma-encoded German
  gloss vector as the Ridge target directly. Would require modifying
  09b_align_gemma.py to handle per-anchor target vectors.
- **Heterogram anchor weighting.** Bridge anchors have high theoretical
  reliability (scribal equivalences) but low absolute count (83). Could be
  weighted higher in sample-weighted Ridge.
- **No coverage-diagnostic run yet.** The audit_anchors + coverage_diagnostic
  scripts are ported but haven't been executed against the v1 artifacts.
  Should be a quick follow-up to identify remaining miss buckets.

### Files

| Script | Status |
|--------|--------|
| `01_parse_tlhdig.py` | new (TLHdig XML → texts/lemmas/heterograms JSON) |
| `04_deduplicate_corpus.py` | single-source TLHdig variant |
| `05_clean_and_tokenize.py` | routed via `hittite_normalize` |
| `06_extract_anchors.py` | new (German→English via multilingual Gemma + NLTK filter + heterogram bridge) |
| `07_train_fasttext.py` | direct copy |
| `08_fuse_embeddings.py` | direct copy |
| `09_align_and_evaluate.py` | inherited from Akkadian with L5-refined partition |
| `09b_align_gemma.py` | inherited; α=0.01 |
| `10_export_production.py` | adapted (HittiteLookup, JSON vocab) |
| `audit_anchors.py` | ported from Akkadian |
| `coverage_diagnostic.py` | ported from Akkadian |
| `ridge_alpha_sweep.py` | ported; ran at v1 ship (lesson from Akkadian L7) |
| `hittite_normalize.py` | new (NFC + ORACC→ATF + `=` clitic + `-` syllable + vowel-dedup) |

Commits: `400a229` (H1 scaffold), `305223b` (H2 normalize), `e3cc75a`
(H3 TLHdig parser), `8a0fe52` (H4 clean+anchors), `cf4fa4f` (H5 train+align+export).
