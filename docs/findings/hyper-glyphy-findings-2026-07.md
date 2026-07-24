---
title: "Cross-Lingual Embedding Alignment for Six Ancient Languages: Pre-Registered Findings from the 2026-07 Arc"
author: "hyper-glyphy research program"
date: 2026-07-24
abstract: |
  We report the results of one research arc of the hyper-glyphy project, which
  aligns monolingual embeddings for six ancient languages — Sumerian, Egyptian,
  Akkadian, Hittite, Greek, and Sanskrit — into shared English target spaces.
  The arc rests on an honest-evaluation reset (2026-07-06) that invalidated all
  prior headline numbers as surface-variant leakage artifacts and replaced them
  with a stratified CSLS suite under a lemma-group split. On that foundation we
  ran a pre-registered anchor-quality experiment: the Sanskrit slot, built on
  the strongest lexicon join in the project (94.9%), landed its Procrustes
  validation cosine at 0.1145, inside the pre-registered ≤ 0.12 band — anchors
  were never the constraint, and the stronger-anchors lever and Plane A were
  retired. Suite v2 (shared gloss filters, plateau alpha rule) then improved
  combined test accuracy in all six slots, fixed the Akkadian-Gemma
  alpha-selection pathology, and left the whitened-Gemma target ahead of GloVe
  in six of six slots. The capstone K=5 myth study delivered the first
  adequately-powered Plane-B test in the study's history — a real null
  (sumerian–sanskrit, ρ=−0.33, exhaustive p=0.77) — alongside the study's first
  pre-registered positive, a band-edge result for Vṛtra↔Illuyanka (90.55th
  percentile) that carries an explicit midrank/tie-block caveat. Every one of
  these verdicts was banded before measurement; the four verdict sentences are
  reproduced verbatim from the experiment journal.
---

```{=latex}
% U+21D2 (⇒) appears in the verbatim retirement verdict but is absent from
% the Times New Roman body font; render it via the pinned monofont (Menlo),
% which covers it, so the verdict text stays byte-verbatim in this source.
\catcode8658=\active
\def⇒{{\ttfamily\char8658}}
```

# Abstract {-}

We report the results of one research arc of the hyper-glyphy project, which
aligns monolingual embeddings for six ancient languages — Sumerian, Egyptian,
Akkadian, Hittite, Greek, and Sanskrit — into shared English target spaces.
The arc rests on an honest-evaluation reset (2026-07-06) that invalidated all
prior headline numbers as surface-variant leakage artifacts and replaced them
with a stratified CSLS suite under a lemma-group split. On that foundation we
ran a pre-registered anchor-quality experiment: the Sanskrit slot, built on
the strongest lexicon join in the project (94.9%), landed its Procrustes
validation cosine at 0.1145, inside the pre-registered ≤ 0.12 band — anchors
were never the constraint, and the stronger-anchors lever and Plane A were
retired. Suite v2 (shared gloss filters, plateau alpha rule) then improved
combined test accuracy in all six slots, fixed the Akkadian-Gemma
alpha-selection pathology, and left the whitened-Gemma target ahead of GloVe
in six of six slots. The capstone K=5 myth study delivered the first
adequately-powered Plane-B test in the study's history — a real null
(sumerian–sanskrit, ρ=−0.33, exhaustive p=0.77) — alongside the study's first
pre-registered positive, a band-edge result for Vṛtra↔Illuyanka (90.55th
percentile) that carries an explicit midrank/tie-block caveat. Every one of
these verdicts was banded before measurement; the four verdict sentences are
reproduced verbatim from the experiment journal.

# Introduction

The hyper-glyphy project asks a narrow, testable question: how much of the
lexical and thematic structure of an ancient language survives a linear map
from a monolingual embedding space into a modern English semantic space? Six
language slots — Sumerian, Egyptian, Akkadian, Hittite, Classical Greek, and
Sanskrit — share one pipeline: FastText embeddings trained per language,
fused to a common dimensionality, and aligned by Ridge regression into two
English targets (GloVe 300d and a whitened EmbeddingGemma 768d space). Three
of the six slots are written in cuneiform script (the divine determinative
𒀭^[Cuneiform sign provenance, per the Anomaly Atlas convention: U+1202D,
Unicode character name CUNEIFORM SIGN AN — the divine determinative, standard
in ORACC cuneiform encoding.] will be familiar from any Sumerian or Hittite
tablet); the others span
hieroglyphic transliteration, polytonic Greek, and IAST-romanized Sanskrit.

This paper covers one arc of work, July 2026, and its through-line is
methodological: **pre-registration as method**. Every major claim in this
arc — the anchor-quality verdict, both myth-study read-outs, and the
document-level gates — was banded before measurement, and each verdict
sentence below is byte-copied from the experiment journal entry that recorded
it. Where a result sits at the edge of its band, the paper says so in the
same breath.

The arc's foundation is the eval-integrity reset of 2026-07-06. A repo-wide
review found two structural flaws in every slot's evaluation: surface-variant
train/test leakage — anchor extraction deliberately registers multiple
surfaces per lemma with the same gloss, and a random split over *pairs* let
(šarrum, "king") train while (šarru, "king") tested — and alpha tuned on the
test set. Measured on the exact shipped splits, the fraction of test items
with a same-gloss train anchor within edit distance 1 ranged from 29.5%
(Egyptian) to 65.2% (Greek, projected). The fix — a shared union-find
lemma-group split (64/16/20 train/val/test) with alpha selected on
validation — collapsed the Akkadian rerun from 27.79% to 0.09% top-1 (GloVe)
and from 36.43% to 0.14% (whitened Gemma). A three-way diagnostic (44.4%
accuracy on the model's own training anchors; a near-reproduction of the old
number — 27.59% vs 27.79% shipped — when the new code is re-run under the old
random-pair split, quantifying the leakage; only 16.3% of lemma-split test
glosses appearing anywhere in train) confirmed the collapse was real
zero-shot difficulty, not a bug. All prior headline numbers were declared invalidated, and every number
in this paper post-dates that reset.

Two further disciplines carried through the arc. First, a stratified
evaluation (Section 2.3) that separates memorization from generalization
instead of averaging them. Second, disclosure: every deviation from a shared
recipe, every user-accepted trade-off, and every band-edge caveat is recorded
in the journal at the moment of measurement, and reproduced here rather than
smoothed over.

The arc's four results, in order: (i) suite v2, a cleanup pass (shared gloss
filters, a plateau alpha rule) that improved combined accuracy in all six
slots and fixed a known selection pathology (Section 3.1); (ii) the Sanskrit
slot as a pre-registered anchor-quality experiment whose verdict retired an
entire improvement lever (Section 3.2); (iii) the document-level gates that
routed the myth study away from cross-language cosine (Section 3.3); and
(iv) the K=5 myth study, the first adequately-powered test of cross-language
thematic geometry in this project, returning a real null beside a band-edge
positive control (Section 3.4).

# Data and methods

## Language slots

Table 1 summarizes the six slots. Corpus and lexicon facts are taken from the
per-slot data manifests (`languages/<slot>/README.md` and
`languages/<slot>/data/raw/README.md`).

| Slot | Family / script | Corpus (source, size) | Lexicon (anchor source) | Anchors (v2) |
|------|-----------------|-----------------------|--------------------------|:------------:|
| Sumerian | language isolate / cuneiform | ETCSL (36K lines) + CDLI (96K texts, 1.4M lines) + ORACC (90K texts, 4.3M lemmas) | ePSD2 glosses + ETCSL co-occurrence | 13,048 |
| Egyptian | Afroasiatic / hieroglyphic | migrated pre-built from the predecessor heiroglyphy V15 project | TLA / Ramses / BBAW pairs (8,541 raw) | 4,568 |
| Akkadian | East Semitic / cuneiform | ORACC OB + SB projects + letters + DCCLT, 3.0M tokens | ORACC guide-word glosses | 24,116 |
| Hittite | Indo-European (Anatolian) / cuneiform | TLHdig 0.2.0-beta (Zenodo, 22k XML texts, CC-BY) | German guide words, translated via multilingual EmbeddingGemma; heterogram bridge (sux 28 / akk 39) | 11,651 |
| Greek | Indo-European / alphabetic | Diorisis v1.51 (821 JSON files, ~10.2M tokens, Homer to 5th c. AD) | Perseus LSJ (90,424 entries; join hit rate 77.1%) | 105,920 |
| Sanskrit | Indo-European (Indo-Aryan) / Devanagari (IAST) | DCS (15,790 chapter files, 5,679,462 token-lemma records, 90,184 unique lemmas; CC BY 4.0) | Monier-Williams, Cologne CDSL 2020 (177,323 entries; CC BY-NC-SA 3.0; join hit rate 94.9%) | 92,275 |

Table: The six language slots. Anchor counts are suite-v2 values
(journal, 2026-07-19); the Egyptian count is the 06-output basis, not
directly comparable to its v1 post-filter figure (4,152).

## Pipeline

Each slot trains 768-dimensional FastText skip-gram embeddings on its
cleaned corpus, zero-pads them to a fused 1536-dimensional representation,
and learns one Ridge regression per target space: (a) whitened
EmbeddingGemma 768d — BERT-whitening (Su et al. 2021) applied to the raw
encoder output, which earlier work found mandatory for any
contextual-encoder alignment target — and (b) GloVe 6B 300d. Both aligned
views are exposed through per-language `Lookup` classes
(`space="gemma"|"glove"`).

## Word-level evaluation suite

All word-level numbers use the stratified CSLS suite introduced 2026-07-09.
Anchors are split by a shared union-find lemma-group procedure (anchors merge
on shared lemma or shared surface; gloss fallback for Egyptian, which has no
lemma key), 64/16/20 train/val/test, seed 42, with near-surface edges merged
so that no test item has a same-gloss train anchor within edit distance 1
(leak check 0.00% in all slots). Retrieval is CSLS (k=10 mutual-neighbor
reranking) over a 50,000-candidate English vocabulary, reporting exact-match
and WordNet-synonym-credited top-1/5/10. Three regimes are reported per
slot: **dictionary** (a fixed 1,000-anchor sample of the training set —
in-sample by design, measuring memorization of known glosses),
**interpolation** (unseen test lemmas whose gold gloss appears as a training
target), and **zero-shot** (test lemmas whose gold gloss was never a training
target). Headline accuracies are conditioned on the gold gloss being present
in the 50K candidate vocabulary; a **gold-OOV** column reports how many test
items that restriction excludes (shown as excluded/evaluated).

Suite v2 (2026-07-19) made two recipe changes, holding the eval data fixed.
First, a shared `gloss_filters` module became the single source of
anchor-English selection across all six slots: negator rejection (*not, no,
without, never*, plus German equivalents for Hittite),
cross-reference-starter rejection (*see, cf, vid*), a 14-word scaffold-word
list, and a Unicode word regex, with a MIN_HIT_RATE 0.40 gate unchanged from
v1. Second, alpha selection moved to a plateau rule (**alpha-v2**): select on
validation top-5 CSLS, define the plateau as within 100/n_val percentage
points of the maximum, and break ties toward the lowest alpha (v1 selected on
val top-1 with an arbitrary tie-break). All twelve results configurations
(6 slots × 2 targets) record `alpha_selection=val_top5_csls_v2`.

## The Procrustes anchor-quality read-out

To test whether anchor quality — rather than map family or corpus scale —
was the binding constraint on cross-language document geometry, each slot
also fits a semi-orthogonal Procrustes map (W = UVᵀ of XᵀY, 1536→768, no
scale), with full and stable-monosemous anchor variants selected on
validation cosine. The isometry check matters here: the fused 1536d
embeddings have exact rank 768 (FastText 768d plus a projected component),
so a 1536→768 semi-orthogonal map can capture the data subspace exactly
(isometry ρ ≈ 1.0, no projection loss), and a low read-out cannot be blamed
on the projection.

The read-out was pre-registered in the 2026-07-13 Sanskrit design spec,
before the Sanskrit slot was built. Bands, restated: validation cosine
≥ 0.20 ⇒ anchor quality was a binding constraint and the stronger-anchors
lever stays live; ≤ 0.12 ⇒ anchors were never the constraint; between 0.12
and 0.20 ⇒ inconclusive. Sanskrit was built to be the stronger-anchors
condition: its token-level DCS-lemma→MW join hit rate of 94.9% is the best
of any slot (the others' joins are substantially thinner).

## Myth study design

The myth study asks whether cross-language *thematic* geometry — not
word-level translation — is detectable between the document collections of
different slots. Two planes were defined in the study plan. **Plane A** is
direct cross-language cosine in the aligned spaces; it was gated on two
document-level criteria (Section 3.3). **Plane B** is native-space
second-order representational similarity analysis (RSA): within each slot,
theme-centroid profiles are computed from the slot's own fused embeddings
(no cross-language cosine is ever taken), and the *structure* of those
profiles is compared across slots by Spearman correlation of the
theme-pair similarity ladders, with exhaustive permutation tests.

Five themes are defined — cosmogonic, hymnic, wisdom, royal_control,
magical — with five documents per filled theme per slot. Greek lacks a
magical theme (the Greek Magical Papyri are not in the Diorisis literary
corpus) and Hittite lacks wisdom; a slot pair's ladder uses only shared
themes, so ladder size K ranges from 3 to 5. Power is set by K: a K=3
ladder yields 3 upper-triangle values and 6 exhaustive permutations
(minimum attainable p = 0.1667 — structurally unpowered at α = 0.05); K=4
yields 24 permutations (min p ≈ 0.042); K=5 yields 120 (min p ≈ 0.008).
The Sanskrit slot is the only one that fills all five themes, making
sumerian–sanskrit the study's first K=5 ladder. The Sanskrit roster (DCS
chapter-level documents, pinned 2026-07-23) comprises Ṛgveda cosmogonic
hymns (RV 10.129, 10.90, 10.121, 10.190, plus a merged Vṛtra document =
RV 1.32 + 1.80 + 2.12, 862 tokens), the five longest non-cosmogonic RV
hymns, five principal Upaniṣads as wisdom documents, five Atharvaveda
royal-consecration chapters, and five long Atharvaveda incantation
chapters (excluding, by rule, the wedding and funerary books 14 and 18).

Two rules were pre-registered before the run. **Read-out 1 (K=5 ladder
RSA):** if the shared ladder reaches its maximum, exhaustive permutation
p ≤ 0.05 with positive Spearman ρ counts as the study's first
adequately-powered Plane-B positive, stated as such; p > 0.05 is a real,
reportable null, stated as such. **Read-out 2 (Vṛtra positive control):**
the Vṛtra profile correlation's percentile in a 1000-draw same-genre null
at ≥ the 90th percentile ⇒ "supports the IE combat-myth link"; ≤ the 75th
⇒ "fails, consistent with the Kumarbi-control finding"; between ⇒
inconclusive. The earlier doc-level Kumarbi control (Hittite succession
narratives vs the Theogony, 2000-draw null) carries the same band logic
and serves as the reference point.

# Results

## Word-level suite v2

Table 2 reports the current (suite v2) word-level numbers for all six slots
and both targets.

| Slot | Target | alpha | Dict top-1 | Interp top-1 | Zero-shot top-1 | Combined top-1 | Combined syn | Gold OOV |
|------------|----------|--------|:---------:|:---------:|:----------:|:----------:|:---------:|:----------:|
| Sumerian | GloVe | 10 | 77.99% | 7.20% | 0.20% | 4.44% | 5.32% | 145/1240 |
| Sumerian | Gemma | 1000 | 75.60% | 10.13% | 1.22% | 6.61% | 7.42% | 145/1240 |
| Egyptian | GloVe | 1e-3 | 66.29% | 25.15% | 0.00% | 19.07% | 19.73% | 87/451 |
| Egyptian | Gemma | 1e-3 | 70.52% | 26.02% | 0.92% | 19.96% | 21.51% | 87/451 |
| Akkadian | GloVe | 10 | 38.47% | 0.00% | 0.24% | 0.18% | 0.69% | 95/2184 |
| Akkadian | Gemma | 0.01 † | 55.24% | 0.18% | 0.31% | 0.27% | 0.55% | 95/2184 |
| Hittite | GloVe | 1e-4 | 67.99% | 8.90% | 0.00% ‡ | 5.01% | 5.01% | 850/419 |
| Hittite | Gemma | 1e-3 | 78.35% | 13.98% | 0.00% ‡ | 7.88% | 8.59% | 850/419 |
| Greek | GloVe | 0.1 | 37.10% | 4.43% | 0.59% | 3.60% | 6.24% | 2145/15632 |
| Greek | Gemma | 1.0 | 50.11% | 6.07% | 0.68% | 4.91% | 8.12% | 2145/15632 |
| Sanskrit | GloVe | 1e-4 | 35.16% | 3.47% | 0.09% | 2.95% | 4.79% | 1331/14217 |
| Sanskrit | Gemma | 1e4 § | 35.90% | 5.49% | 0.46% | 4.71% | 7.28% | 1331/14217 |

Table: Suite v2 (2026-07-19), top-1 exact % per stratum plus
synonym-credited combined and the gold-OOV column
(excluded/evaluated). † Akkadian Gemma v1's alpha=1e4 flat-noise pick
(dict 19.85%) is fixed under v2's plateau rule (see below). ‡ Hittite
zero-shot n=0 hits at both alphas; gold OOV 850/419 — the majority of
Hittite test items are OOV of the 50K candidate vocabulary. § Sanskrit
Gemma alpha=1e4 is a real interior signal max, not a flat-noise pick
(user-accepted trade, 2026-07-18; see text).

Three observations. First, **combined test accuracy on the primary
whitened-Gemma target improved in all six slots** relative to suite v1
(Egyptian 15.47% → 19.96%, the biggest v2 winner; Sumerian 5.54% → 6.61%;
Akkadian 0.13% → 0.27%; Hittite 6.83% → 7.88%; Greek 4.37% → 4.91%;
Sanskrit 3.83% → 4.71%). The anchor-count guardrail passed in all six
slots; v2's stricter filters reject between 52 (Sumerian) and 3,649
(Sanskrit, −3.8%) of the raw joins.

Second, the **memorization/generalization trade** is now explicit rather
than hidden in an average. The dictionary stratum (in-sample memorization)
runs 35–78% top-1 across slots at 50K candidates, while zero-shot remains
near zero everywhere (0.00–1.22%): the linear map retrieves known glosses
well and does not generalize to unseen lemma families at these corpus
scales. Sanskrit's Gemma alpha illustrates the trade directly: alpha=1e4 is
a genuine interior maximum of the validation signal that gives up in-sample
dictionary accuracy (44.40% v1 → 35.90% v2) in exchange for improvement in
every test stratum — a trade accepted deliberately and recorded as such
(journal, 2026-07-19).

Third, the **Akkadian-Gemma pathology is fixed, not papered over**. This
was the pre-registered acid test for the alpha-v2 rule: v1's Gemma
dictionary accuracy of 19.85% came from alpha=1e4, a flat-noise pick that
won by a single anchor over an entirely flat 0.1–1000 validation plateau.
The plateau rule, applied to the *same* flat sweep, ties toward the lowest
alpha and picks 0.01 — dictionary jumps to 55.24% and combined from 0.13%
to 0.27%, with no change to the eval data (Figure 1). With this fix, the
whitened-Gemma target beats GloVe on combined accuracy in **six of six
slots**.

![Akkadian Gemma validation sweep with the v1 top-1 pick (α=10^4^, a
flat-noise selection; dictionary 19.9%) and the v2 plateau pick (α=0.01;
dictionary 55.2%). Same sweep, same eval data — only the selection rule
changed.](figures/fig2_akkadian_alpha.pdf){width=90%}

## The anchor-quality experiment: Sanskrit and the retirement verdict

The Sanskrit slot (shipped 2026-07-16) was built to answer a standing
question with a pre-registered read-out: would a stronger anchor set move
the Procrustes plane? The slot's inputs are the strongest in the project —
DCS parsed with 0.000% loss (0 of 6,713,257 token lines), a 94.9%
token-level lemma→lexicon join against Monier-Williams (5,391,784 hits /
287,678 misses), 95,924 v1 anchors with 90,176 valid at fit time. Its
word-level v1 suite behaved like the other slots' (Gemma combined 3.83%,
beating GloVe's 2.22%; dictionary 44.40%), i.e. a normal member of the
family rather than an outlier.

The Procrustes read-out came back at validation cosine **0.1145** (full
variant 0.11451, n=57,834, chosen over stable 0.11422; n_fit 70,324;
isometry ρ 1.0000). That falls in the pre-registered **≤ 0.12** band, and
the pre-registered verdict was applied verbatim per the spec:

> "anchors were never the constraint ⇒ retire the stronger-anchors lever,
> and with it the last named route to Plane A."

The reference slots make the verdict legible (Figure 2): sumerian 0.1157,
greek 0.1149, hittite 0.0586, sanskrit 0.1145. Three slots with very
different corpus and lexicon quality — Sumerian's composite cuneiform
corpus, Greek's 77.1% LSJ join, Sanskrit's 94.9% MW join — converge at
~0.115, which suggests a structural ceiling on the semi-orthogonal plane
rather than an anchor-quality effect; Hittite's lower 0.0586 remains
unexplained by this test. Suite v2's re-measurement (recorded for the
record, not re-litigated: sanskrit 0.1198, sumerian 0.1117, greek 0.1163,
hittite 0.0666) left all four slots ≤ 0.12; the v1 retire verdict was
pre-registered on the v1 recipe and stands. Sanskrit's v2 value, 0.1198,
sits closest of the four to the 0.12 band edge — worth noting factually.

![Procrustes convergence: per-slot validation cosines under the v1 and v2
recipes against the pre-registered bands (retire ≤ 0.12, binding ≥ 0.20).
Sumerian, Greek, and Sanskrit converge at ~0.115 despite very different
anchor quality; no slot approaches the 0.20
band.](figures/fig1_procrustes.pdf){width=90%}

## Document-level gates

Two document-level panels gate cross-language claims. **Gate 1**
(within-language): Sumerian ETCSL genre leave-one-out over 338
compositions, 5 genres, majority baseline 40.83% — gemma_aligned 63.31%
(+22.5 pp), glove_aligned 60.95% (+20.1 pp), fused_unaligned 68.05%
(+27.2 pp; projection cost ≈ 5 pp). The criterion (≥ 15 pp over majority in
at least one aligned space) is a **PASS**: the aligned spaces retain real
within-language document structure.

**Gate 2** (cross-language): parallel retrieval, Hittite → Greek over a
pool of 820 Greek documents, with the Kumarbi succession narrative vs
Hesiod's Theogony as positive control. Criterion: MRR ≥ 0.1 with the
positive control in the top quartile. Measured on the Ridge plane:
Kumarbi→Theogony rank 731/820, Illuyanka→Theogony 781/820,
Ullikummi→Theogony 788/820, MRR 0.0013 — **FAIL**, with all three pairs in
the bottom 11% of the pool. Word-level alignment does not compose into
cross-slot document retrieval. Re-measured on the semi-orthogonal
Procrustes plane (2026-07-13): ranks 526/798/796, MRR 0.0015 — FAIL again;
Plane A stays no-go under both map families.

A follow-up hubness diagnostic (2026-07-19, suite-v2 space) tested the
anti-hub hypothesis for the Gate-2 failure and returned a **partial**
verdict. All three control pairs target the same Greek document (the
Theogony), whose SIF centroid is a genuine norm outlier — L2 norm 4.02
against a pool mean of 4.69 (std 0.32), the 1.8th percentile, suggesting
near-cancellation from lexical diversity. But its mean cosine to the rest
of the pool sits only at the 15th percentile (0.853 vs pool mean 0.885) —
below median, not bottom-decile — and ranks recomputed in the v2 space
(340/770/733) are better than v1's, with Kumarbi→Theogony no longer
bottom-decile at all. The Theogony carries a real geometric anomaly, but
not a clean "far from everything" anti-hub signature; some pair-specific
factor remains unaccounted for.

## The myth study at K=5

With Plane A gated off, the myth study runs entirely on Plane B —
native-space second-order RSA — and suite v2 supplied the spaces for a full
re-run with Sanskrit as the fourth slot (2026-07-24; six slot pairs, zero
dropped documents; every pinned roster ID resolved). Table 3 and Figure 3
give all six pair RSAs.

| Pair | K | ρ | p | n_perms | Themes dropped |
|------|:-:|:----:|:----:|:-------:|----------------|
| sumerian–hittite | 4 | 0.3143 | 0.25 | 24 | wisdom |
| sumerian–greek | 3 | 1.0 | 0.1667 | 6 | wisdom, magical |
| **sumerian–sanskrit** | **5** | **−0.3333** | **0.7667** | **120** | none |
| hittite–greek | 3 | 0.5 | 0.5 | 6 | magical |
| hittite–sanskrit | 4 | −0.3714 | 0.8333 | 24 | wisdom |
| greek–sanskrit | 3 | −0.5 | 0.8333 | 6 | wisdom, magical |

Table: All six slot-pair ladder RSAs (exhaustive permutation tests). The
sumerian–sanskrit pair is the study's first K=5 ladder.

![Slot-pair RSA matrix: shared-ladder K, Spearman ρ, and exhaustive
permutation p for the six pairs. The sumerian–sanskrit cell is the study's
first adequately-powered (K=5) test.](figures/fig3_rsa_matrix.pdf){width=90%}

**Read-out 1.** Bands, restated: at the ladder maximum, exhaustive p ≤ 0.05
with positive ρ would have been the study's first adequately-powered
Plane-B positive; p > 0.05 is a real, reportable null. Measured:
sumerian–sanskrit K=5, ρ=−0.3333, exhaustive p=0.7667 (n_perms=120) —
p > 0.05, verdict verbatim:

> "a real, reportable null — thematic geometry does not detectably align."

This is the first test in the study with enough permutations to clear
α = 0.05, and it comes back negative-signed and null. The hittite–sanskrit
K=4 ladder (ρ=−0.3714, p=0.8333) is also null under the same rule. The IE
gradient does not rescue a relatedness story: the mean ρ over the three
Indo-European pairs is −0.1238 against 0.327 for the non-IE pairs — both
Sanskrit IE pairs are negative, so adding the IE triangle's third leg does
not elevate the gradient; if anything the non-IE pairs read higher on this
measurement.

**Read-out 2.** Bands, restated: Vṛtra profile-correlation percentile in
the 1000-draw same-genre null ≥ 90th ⇒ supports; ≤ 75th ⇒ fails; between ⇒
inconclusive. Measured (Figure 4): vs Illuyanka (K=4 ladder), ρ=0.4,
percentile 90.55 ⇒ verdict verbatim:

> "supports the IE combat-myth link"

— the study's first pre-registered positive, and on the phylogenetically
closest pair (both Indo-European, both featuring a storm-god/serpent combat
myth). Stated plainly, as the journal states it: this is one sub-control
clearing the band by a narrow margin (90.55 vs the 90th-percentile line),
not a strong positive. Vs Theogony (K=3 ladder), ρ=0.5, percentile 62.3 ⇒
verdict verbatim:

> "fails, consistent with the Kumarbi-control finding."

For that ladder, ~75.4% of the null ties at ρ=+0.5 — the Vṛtra-vs-Theogony
profile sits exactly at the null's tied mode, the same pattern the Kumarbi
control showed at K=3.

![Vṛtra positive control: observed profile-correlation percentile against
the 1000-draw same-genre null for both sub-controls, with the
pre-registered 90th/75th-percentile bands. Vṛtra↔Illuyanka (ρ=0.4) clears
the support band at the 90.55th percentile; Vṛtra↔Theogony (ρ=0.5) lands at
the 62.3rd and fails. Caveat, quoted from the journal: "the 90.55th
percentile is a midrank value: the .55 margin itself reflects null draws
tying the observed ρ=0.4 (the tie mass at the observed value is not
separately recorded in this run), so the clearance over the 90th-percentile
band is within one tie-block's width. The pre-registered verdict is defined
on the midrank percentile and stands; the discreteness caveat applies with
full force."](figures/fig4_vrtra_control.pdf){width=90%}

**Kumarbi control stability.** The doc-level positive control
(Kumarbi/Ullikummi/Illuyanka vs Theogony, 2000-draw null) is pure Plane B —
it runs entirely on native fused embeddings, which suite v2 did not touch —
and re-running it on the v2 spaces reproduced the v1 numbers bit-for-bit:
kumarbi–theogony ρ=0.5, percentile 58.98; ullikummi–theogony ρ=0.5, 58.98;
illuyanka–theogony ρ=1.0, 90.15. Both named plan controls sit at the null's
42.6%-tied mode; the non-plan-named Illuyanka pair reaches the discreteness
ceiling at ρ=+1.0 (~19.7% of the null tied there). Verdict, unchanged and
verbatim: "FAIL — named controls at null mode; per plan, cross-language
doc-level claims narrow to within-language planes." The bit-for-bit
reproduction is itself informative: the Kumarbi result is a property of the
native embeddings, not an artifact of the (now-changed) Ridge recipe.

**Supporting measurements.** The translation delta (Spearman correlation
between native-space and aligned-space document-distance rankings, per
slot) is high everywhere: Sumerian 0.8838 (39 docs), Hittite 0.8514 (18),
Greek 0.9044 (12), Sanskrit 0.9378 (25, the highest of the four) — aligning
distorts within-language document geometry only mildly. Cosmogonic concept
fingerprints (within-language centroid profiles over ten concepts, compared
across slots in aligned space) are all measurable (`fingerprint_status` ok
in all four slots); the six cross-slot correlations run from
sumerian–greek 0.8788 down to hittite–sanskrit 0.6242, but per the results
JSON's baseline note, all must be read against the mismatched-theme
cross-slot baseline (pair means 0.55–0.85, maxima up to 0.96) — none of the
six is dramatically elevated above its own pair's mismatched baseline.

# Discussion and limitations

**What the maps support — and what they cannot do.** The arc's positive
findings are all within-language: aligned spaces classify Sumerian genre at
+22.5 pp over the majority baseline, preserve document geometry at
Spearman 0.85–0.94 against the native spaces, and retrieve memorized
glosses at 35–78% top-1 over 50K candidates. Everything cross-language
failed or nulled at adequate power: cross-slot document retrieval is
worse than chance for the known-parallel controls under both map
families; the first K=5 RSA ladder is a signed-negative null; and the
matched-theme fingerprint correlations do not separate from their
mismatched baselines. The consistent picture is that these linear maps
transport lexical memorization and coarse within-language structure, and
do not carry cross-language thematic geometry at the scales measured. The
Procrustes convergence of three dissimilar slots at ~0.115 says the
constraint is structural, not a fixable input-quality problem — which is
exactly why the pre-registered experiment was allowed to retire the lever.

**Band-edge honesty.** The arc's single pre-registered positive clears its
band by 0.55 percentile points, and that margin is narrower than it looks:
the 90.55th percentile is a midrank value, so the clearance is within one
tie-block's width of the 90th-percentile line (the tie mass at the observed
ρ=0.4 was not separately recorded in this run). The verdict is defined on
the midrank percentile and stands; a reader should nevertheless weight it
as a band-edge result, supported by one sub-control while the other
(vs Theogony) fails.

**OOV conditioning.** All headline accuracies are conditioned on the gold
gloss being inside the 50K-candidate vocabulary. This is disclosed per slot
(Table 2's gold-OOV column) and is severe for Hittite, where 850 test
items are excluded and only 419 evaluated — a consequence of
German→English gloss translation producing specialized vocabulary. Combined
accuracies are therefore not comparable across slots without reading that
column.

**Ladder discreteness and power.** Spearman ρ over K=3 ladders takes only
the values {±1, ±0.5, 0}, exhaustive permutation floors are coarse (min
p = 0.1667 at K=3), and large fractions of the null mass tie at single
values (75.4% at ρ=+0.5 for the Vṛtra–Theogony ladder). Only the K=5 pair
is adequately powered at α = 0.05, and even it has a p-floor of ~0.008.
K=3 results in Table 3 — including the eye-catching sumerian–greek ρ=1.0 —
are structurally unpowered and must not be read as evidence.

**Construct and chronology caveats.** The "magical" theme compares
different text-type constructs across slots (Sumerian bilingual incantation
tablets vs Hittite SISKUR ritual prescriptions), and the Sumerian magical
documents are Late-Babylonian-period bilinguals, chronologically later than
the mainly Old Babylonian ETCSL literary corpus. Both facts are recorded in
the results JSON and bound the interpretation of any ladder that includes
the magical theme.

**Single-experiment scope.** The retirement verdict binds the
stronger-anchors lever on the semi-orthogonal Procrustes plane as
operationalized here; it does not claim that no anchor improvement of any
kind could ever matter elsewhere, and Hittite's low 0.0586/0.0666 cosine
remains an open observation rather than an explained one.

# Reproducibility

All computed artifacts — trained FastText models, fused vectors,
Ridge/Procrustes weights, English embedding caches, processed corpora,
anchors, eval results, and production exports (~45GB) — are mirrored in a
companion Hugging Face dataset repo,
**ebrinz/hyper-glyphy-artifacts**, whose layout mirrors this repository
exactly. A fresh clone bolts on in one step:

```bash
bash shared/scripts/fetch_artifacts.sh   # hf download + recreate local symlinks
```

This reproduces the published numbers exactly without retraining (FastText
training is non-deterministic, so a retrain gives slightly different
vectors). The pre-v2 artifact state is pinned as the `suite-v1` tag on the
HF repo, from which the archived v1 table reproduces. The exact
environment that produced the committed artifacts is pinned in
`requirements.lock.txt` (Python 3.12.3; gensim 4.4.0, numpy 2.3.5,
scikit-learn 1.7.0); the gensim FastText `.model` artifacts are
version-sensitive pickles and must be loaded under the lockfile
environment. Raw third-party corpora are not mirrored (licensing varies by
source); each slot documents its fetch steps in
`languages/<slot>/data/raw/README.md`. MW-derived Sanskrit files in the
mirror carry CC BY-NC-SA 3.0 (Cologne CDSL); everything computed in the
project is CC BY 4.0. Myth-study numbers in this paper are read directly
from the git-tracked `shared/results/myth_study.json`; all verdict
sentences are byte-copies from `docs/EXPERIMENT_JOURNAL.md` (entries
2026-07-16, 2026-07-19, 2026-07-24).

# References {-}

- **ETCSL** — The Electronic Text Corpus of Sumerian Literature, Oxford
  Text Archive. <https://etcsl.orinst.ox.ac.uk/>
- **CDLI** — Cuneiform Digital Library Initiative, bulk ATF dump
  (`github.com/cdli-gh/data`). <https://cdli.ucla.edu/>
- **ORACC** — The Open Richly Annotated Cuneiform Corpus (incl. ePSD2,
  DCCLT, blms). <https://oracc.museum.upenn.edu/>
- **TLA** — Thesaurus Linguae Aegyptiae
  (<https://thesaurus-linguae-aegyptiae.de/>); with Ramses
  (<http://ramses.ulg.ac.be/>) and BBAW (Berlin-Brandenburgische Akademie
  der Wissenschaften), via the predecessor heiroglyphy project.
- **TLHdig** — TLHdig 0.2.0-beta, Hethitologie Portal Mainz, Zenodo record
  15459134 (CC-BY).
- **Diorisis** — The Diorisis Ancient Greek Corpus (JSON), v1.51, Figshare
  dataset 12251468.
- **LSJ** — Liddell-Scott-Jones lexicon, Perseus Digital Library
  (`github.com/PerseusDL/lexica`).
- **DCS** — Hellwig, O., *The Digital Corpus of Sanskrit (DCS)*,
  2010–2024. `github.com/OliverHellwig/sanskrit` (CC BY 4.0).
- **Monier-Williams** — *A Sanskrit-English Dictionary* (1899), Cologne
  Digital Sanskrit Lexicon (CDSL) 2020 digitization, The Sanskrit Library
  and Thomas Malten (CC BY-NC-SA 3.0).
- **GloVe** — GloVe 6B pre-trained English word vectors (400K words,
  300d), Stanford NLP. <https://nlp.stanford.edu/projects/glove/>
- **EmbeddingGemma** — multilingual text-embedding model (768d), used
  whitened as the primary alignment target; whitening per BERT-whitening
  (Su et al., 2021).
