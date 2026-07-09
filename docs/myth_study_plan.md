# Myth Study Plan: Cross-Civilization Semantic Comparison of Creation Myths and Magical Texts

*Status: planning document — study not yet executed. Gate results from the eval-redesign pipeline are incorporated below.*

---

## 1. Purpose

This study applies the aligned semantic spaces built in this project to a comparative question: do creation myths, magical incantation texts, and hymnic/spiritual texts from different ancient civilizations exhibit measurably convergent structure, and if so, is that convergence a property of meaning shared across languages or an artifact introduced by the English-mediation layer used in alignment?

The project has trained FastText embeddings for five ancient languages (Sumerian, Egyptian, Akkadian, Hittite, Classical Greek), fused them to 1536-dimensional vectors, and aligned each into a shared whitened-Gemma 768d space via Ridge regression. Genre-level document retrieval in that shared space has been validated (Gate 1, below). The myth study is the first interpretive use of that infrastructure: not a translation claim, not an argument that aligned embeddings recover historical transmission, but a structured inquiry into whether the geometry of meaning differs systematically between text types and whether that difference is consistent across cultures that did not share a language.

This is meaningful precisely because the comparison is indirect. Cross-civilization convergence in semantic structure — even at the document level, even mediated by English — would be a non-trivial finding, since the null model (random same-genre pairs) provides an explicit baseline. Absence of convergence is equally informative.

---

## 2. Research Questions

**(a) Cosmogonic affinity.** Do creation/flood-narrative texts across Sumerian, Hittite, and Greek (and Akkadian, once per-text corpus is available) show measurably higher pairwise affinity in aligned space than genre-matched controls drawn from the same language slots? Concretely: is the mean cosine similarity of cosmogonic cross-language pairs significantly above the bootstrap null distribution of randomly matched narrative pairs?

**(b) Magical vocabulary geometry.** Do incantation and ritual binding texts — Hittite ritual tablets alongside whatever binding-spell comparanda can be assembled from Sumerian and Greek — occupy a geometrically distinctive region compared to narrative compositions? The hypothesis is that operational magical texts share a functional vocabulary (commanding, naming, binding, apotropaic address) that produces a measurable fingerprint even after alignment.

**(c) Kumarbi–Theogony as positive control.** The Kumarbi Cycle (CTH 344) is the documented source for significant portions of Hesiod's Theogony — theogonic succession, the castration motif, the stone-swallowing episode. This known transmission relationship is the study's positive control: the expected outcome is that the Illuyanka texts and the Kumarbi documents cluster with Theogony more strongly than a random Hittite–Greek pair would. Note that because Plane A (direct aligned-space cosine) failed its gate, this control is now routed through Plane B (native-space RSA); see Method and Go/No-Go sections.

**(d) Translation-delta consistency.** Which text classes — myth, magic, hymn — show the largest per-pair divergence between aligned-space similarity and native-space second-order similarity? Is that divergence consistent in direction and magnitude across the five language slots, or does it vary with language? A theme that converges in native space but not in aligned space is a translation-loss finding (the English projection suppresses the relevant geometry), not a cultural-divergence finding. The plan must track these two interpretations separately.

Question (d) was added to address the open question raised in `docs/RESEARCH_VISION.md` (line 111): *"Is the alignment layer (Ridge regression into GloVe space) itself distorting the geometry we want to measure? Should we compare manifolds in their native spaces instead?"* The two-plane design operationalizes that question.

---

## 3. Candidate Text Sets

**Sumerian.** The ETCSL corpus (36,496 lines across ~338 compositions in the genre-tagged subset) is the primary source. Confirmed compositions in the data:

- c141: Inanna's Descent to the Netherworld — cosmogonic/mythic narrative, opening line verified ("From the great heaven she set her mind on the great below")
- c174: The Flood Story (Ziusudra) — confirmed present, flood/creation narrative mentioning An, Enlil, Enki, Ninhursaga and the fashioning of humanity
- Enki narratives (c.1.x range, approximately 10–15 compositions): Enki and the World Order, Enki and Ninhursaga, etc. — require genre-tag filtering to identify the cosmogonic subset
- Incantation/magical compositions: present in ETCSL as a separate genre; the exact composition count for the magical slot will be determined during text roster construction

**Hittite.** The TLHdig corpus (20,713 texts) provides per-document access. Confirmed present:

- KBo 3.7: Illuyanka myth (the Dragon Myth, first recension) — confirmed in corpus
- KUB 17.5: Illuyanka myth (second recension) — confirmed in corpus
- KUB 33.x ritual range: 71 texts present; covers a wide range of purification, oracle, and festival rituals suitable for the magical comparandum slot
- **Not present: KUB 33.120 (Kumarbi Cycle, CTH 344).** The main Kumarbi tablet is absent from TLHdig. This is a corpus gap that directly affects the positive control; see Go/No-Go.

**Greek.** The Diorisis corpus (820 texts, 88 authors) is the Greek source. Confirmed present:

- Theogony: Hesiod (0020) — Theogony (001) — confirmed as an independent text document
- Works and Days: Hesiod (0020) — available as comparandum (cosmogonic/didactic)
- Homeric Hymns: present (multiple entries)
- **Not present: Greek Magical Papyri (PGM).** Diorisis covers literary Greek; the PGM are not included. The magical/incantation slot for Greek requires an external corpus and is a study prerequisite, not a current asset. This absence must be noted in any comparison of the magical text class.

**Egyptian.** The Egyptian corpus (heiroglyphy V15, migrated as cleaned transliteration text at `languages/egyptian/data/processed/cleaned_corpus.txt`, ~5.4 MB) is a monolithic running-text file without per-document segmentation or genre metadata. It contains word-level anchor pairs (8,541 anchors) but no text identifiers or boundaries. Coffin Texts and Book of the Dead spells are the natural comparanda for Egyptian cosmogonic and magical material, but they cannot be addressed as separate documents in the current corpus state. Egyptian occupies the same status as Akkadian for the document-level study: the embedding infrastructure exists (including `languages/egyptian/models/fused_embeddings_1536d.npz`), but per-text document centroids require a corpus preprocessing step that has not been done. Egyptian is therefore listed as a prerequisite slot, not an immediately available asset.

**Akkadian.** No per-text corpus is currently available. ORACC and CDLI dumps are word-level only, without the per-text identifiers needed to build document centroids. The Akkadian Epic of Creation (Enuma Elish) and Atrahasis are the obvious comparanda; they become available once a per-text Akkadian corpus is assembled. This is an explicit prerequisite.

---

## 4. Method: Two-Plane Convergence Design

The study uses a two-plane architecture that was designed specifically to separate English-mediated convergence (Plane A) from native-space structural convergence (Plane B), and to measure how much the alignment projection reshapes the geometry (translation delta).

### Plane A — Aligned Space (English-Mediated Direct Cosine)

Document centroids are computed using Smooth Inverse Frequency (SIF) weighting over aligned word vectors in the shared whitened-Gemma 768d space. For each composition in the text roster, a centroid is built from its token embeddings using the aligned production exports (`languages/*/final_output/`). Cross-language pairwise cosine similarities between theme-matched texts (cosmogonic, magical, hymnic) are compared against a genre-matched null distribution generated by bootstrapping over same-genre non-parallel pairs from the same language slots.

Thematic concept fingerprints augment the pairwise matrix: each document centroid is ranked against a curated English concept list (water, chaos, serpent, name, fate, bind, create, mountain, flood, sky) in aligned space. This produces a per-document concept profile that can be compared cross-linguistically without requiring direct pairwise cosine between documents.

Gemma–GloVe agreement is used as a per-claim confidence signal. When both aligned spaces (whitened-Gemma 768d and GloVe 300d) assign a pair elevated similarity, the finding is treated as robust. Divergence between the two spaces flags a claim for scrutiny.

**Current status of Plane A:** Gate 2 (parallel retrieval) failed. The cross-language retrieval geometry is non-discriminative: centroids are healthy within each language but cross-language cosines form a tight, low-variance blob (mean 0.252, std 0.024). Direct cross-language cosine similarities cannot distinguish thematically matched pairs from random same-genre pairs at the document level on current maps. Plane A cross-language cosine comparisons are therefore NO-GO pending a map quality improvement (e.g., Procrustes remap with more stable anchors, or a new alignment strategy). Within-language comparisons in aligned space remain valid and unaffected.

### Plane B — Native Space (Second-Order RSA)

Cross-language cosine in native FastText/fused spaces is undefined: each language's 1536d fused embeddings span an independent coordinate system. Plane B does not compare documents directly across languages. Instead, it compares similarity *structures*.

For each language slot with per-text document centroids, a document-by-document similarity matrix is computed in that language's native fused space (`languages/*/models/fused_embeddings_1536d.npz`). Convergence is measured by Spearman rank correlation over the upper triangle of matched-text pairs across two languages' matrices. If Hittite texts agree with Greek texts about which compositions are geometrically similar to each other — without any English mediation — that is a structural convergence signal that does not depend on the quality of the alignment maps.

Plane B is unaffected by the Gate 2 failure because it never performs cross-language cosine. It is the study's primary plane until Plane A maps are improved.

**Method prerequisites for Plane B:** the matched-text roster must be identical across both planes (same document IDs in both the native-space and aligned-space matrices) so that the translation delta is well-defined. Native-space document centroids require the fused embedding caches (`languages/*/models/fused_embeddings_1536d.npz`), which exist for all five languages; what is missing for Egyptian and Akkadian is the per-text segmentation layer, not the embeddings themselves.

### Translation Delta

For each language with available per-text data, the aligned-space pairwise matrix (Plane A, within-language) is correlated with the native-space matrix (Plane B). The per-pair delta is `s_aligned − s_native_second_order` (after rank-normalizing both matrices to make them comparable). A per-document distortion score aggregates the row-wise delta for each text.

**Interpretation rule (mandatory):** a theme (e.g., magical texts) that shows within-language structural convergence in Plane B but fails to show cross-language convergence in Plane A must be interpreted as a potential translation-loss finding — the English projection may be suppressing the relevant geometry — not as evidence of cultural divergence. Any claim of cultural non-convergence must demonstrate that the native-space signal is also absent before that conclusion can be drawn.

---

## 5. Controls

**Positive control — Kumarbi → Theogony (Plane B).** The documented transmission from the Hittite Kumarbi Cycle (CTH 344) to Hesiod's Theogony is the study's positive control. Because KUB 33.120 is absent from TLHdig, the Kumarbi side of this pair must be reconstructed from fragments present in the KUB 33.x range. If structural similarity between any KUB 33.x texts and Theogony exceeds the genre-null distribution in Plane B, this is evidence that Plane B can detect known transmission relationships. If it does not, the positive control is inconclusive (absent source text), not falsified.

**IE relatedness gradient.** Hittite and Greek are both Indo-European. Their native-space RSA matrices should show elevated structural similarity relative to cross-family comparisons (Hittite–Sumerian, Greek–Egyptian), all else equal. This gradient acts as a sanity check on Plane B sensitivity: if IE relatedness produces no elevation, structural similarity signals elsewhere should be interpreted cautiously.

**Random same-genre null.** All cross-language similarity claims are tested against a bootstrap distribution of randomly matched same-genre non-parallel pairs from the same language slots. This is the primary statistical control for both planes.

**Future: Sanskrit IE triangle.** Adding an Atharvaveda slot (Sanskrit) would create a three-way IE comparison (Hittite, Greek, Sanskrit), strengthening the IE gradient test and providing a magical-text comparandum with clear genre labels.

---

## 6. Go/No-Go Dependencies

**Gate 1 — Genre discriminability (PASS)**

Criterion: ETCSL genre leave-one-out nearest-centroid accuracy must beat the majority baseline (40.83%, n=338 compositions, 5 genres) by ≥15 percentage points in at least one aligned space.

Measured results:
- `gemma_aligned`: 63.31% (+22.5 pp above baseline) — **clears gate**
- `glove_aligned`: 60.95% (+20.1 pp above baseline) — clears gate
- `fused_unaligned`: 68.05% (+27.2 pp above baseline) — clears gate (projection cost: ~5 pp from 68.05 to 63.31)

Verdict: **PASS.** Document-level genre structure is real and recoverable in aligned space. This validates Plane A within-language genre separation and Plane B native-space document structure.

**Gate 2 — Parallel retrieval (FAIL)**

Criterion: parallel-text retrieval MRR ≥ 0.1, with the Kumarbi/Illuyanka positive-control pair ranking in the top quartile of its pool (top ~205 of 820).

Measured results:
- Kumarbi (KUB 33.120/CTH 344): **absent from TLHdig** — pair dropped
- Akkadian pairs: both dropped — no per-text IDs in available ORACC dumps
- Illuyanka (KBo 3.7 + KUB 17.5) → Theogony: rank 781 of 820, MRR 0.0013 — far below gate

Controller diagnostics showed healthy within-language centroids (Hittite 304, Greek 71, English 6699 in-vocab tokens), mean-centering had no material effect (rank 781 → 773), and cross-language similarities were a non-discriminative blob (mean 0.252, std 0.024). The failure is in the alignment maps' cross-language geometry, not in corpus coverage.

Verdict: **FAIL.**

**Consequence for study design:** Plane A cross-language cosine claims are on hold. The study proceeds via Plane B (native-space RSA) as the primary method, which is unaffected by map quality. Cross-language conclusions must be grounded in structural convergence (Plane B) rather than direct distance (Plane A). Plane A is reinstated if alignment maps are improved via Procrustes remap or a stronger anchor set.

**Word-level suite context (for calibration).** Dictionary strata accuracy ranges from 39–79% at top-1 across slots. Zero-shot is ~0–1% by construction (anchor split prevents gloss-group leakage). Gemma beats GloVe combined in 4 of 5 language slots. Known anomalies: Akkadian alpha-selection noise, Hittite candidate vocabulary covers only ~31% of its gold glosses. These constrain how much semantic precision can be expected from individual word-level embeddings, which propagates into document centroid quality.

---

## 7. Out of Scope / Future

**Sanskrit (Atharvaveda).** The Atharvaveda is the natural magical-text comparandum for an Indo-European triangle (alongside Greek and Hittite). It also provides a creation-hymn slot (Nasadiya Sukta in the Rigveda). Sanskrit is listed as a future slot that strengthens both the IE gradient control and the magical-text comparison. It is not currently in the pipeline.

**Mayan.** Popol Vuh is a cosmogonic text with no genetic relationship to the Eurasian cluster. It is a candidate document-level null control only — expected to show low structural convergence with all five current slots. It is out of scope until the study design for the five primary slots is finalized.

**Procrustes remap.** A Procrustes remap of native embeddings using a more stable anchor set (e.g., Swadesh-core concepts only, with coverage checks against actual in-vocab rates) is the most direct route to reinstating Plane A. This is a pipeline task, not a study task, and is listed in the roadmap as a prerequisite for future Plane A claims.

**Egyptian per-text segmentation.** Coffin Text spells and Book of the Dead chapters require the Egyptian corpus to be segmented into individual text units with IDs. The current cleaned corpus (heiroglyphy V15) is a monolithic file. Segmentation is a pipeline prerequisite before Egyptian can join the document-level study as a first-class slot rather than a future addition.

**Greek Magical Papyri.** PGM are not in the Diorisis corpus. Importing and aligning them requires a separate ingestion pipeline. Until then, the Greek magical-text slot is empty and any magical-text comparison is Hittite ritual vs Sumerian incantation only.

**Mayan and other document-level null controls.** After the five-slot study is complete, geographically and genetically isolated corpora (Mayan, Nahuatl) serve as stress tests: high cross-language RSA with isolated traditions would suggest the method is detecting corpus structure artifacts rather than genuine cultural affinity.
