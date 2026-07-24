# Arc Findings Paper (LaTeX-born PDF) — Design

**Date:** 2026-07-24
**Status:** Approved
**Goal:** Close the 2026-07 arc with a paper-structured findings document —
markdown source rendered to a LaTeX-born PDF via the house pandoc/XeLaTeX
pipeline — covering the honest-eval foundation, the Sanskrit slot and its
pre-registered anchor-quality retirement, suite v2, and the K=5 myth study
(powered null + split Vṛtra verdict) as the capstone.

## Background

The arc's results live in `docs/EXPERIMENT_JOURNAL.md` entries (2026-07-16,
-19, -24), the README tables, and the git-tracked `shared/results/*.json`.
The repo has one prior findings PDF (Anomaly Atlas, 2026-04-20): markdown →
pandoc → XeLaTeX with `languages/sumerian/docs/templates/cuneiformy-pandoc.tex`
(fontspec, cuneiform auto-switch), both source and PDF committed, README
dual-linked. Toolchain present (pandoc 3.9, xelatex). This document needs
IAST diacritics, polytonic Greek, and occasional cuneiform in one PDF.

## Scope

- Full arc, paper register: abstract / intro / data & methods / results /
  discussion & limitations / reproducibility / source references.
  Target ~15–20 rendered pages.
- Tables + exactly 4 generated figures (below). Hand-written prose;
  numeric integrity enforced by the review gate, not templating.
- Out of scope (recorded): new analyses or measurements of any kind; any
  change to results JSONs or the journal; number-injection templating;
  arXiv submission mechanics; changes to the Sumerian template.

## Deliverables

1. `docs/findings/hyper-glyphy-findings-2026-07.md` — the paper source.
2. `docs/findings/hyper-glyphy-findings-2026-07.pdf` — built artifact
   (committed, Anomaly Atlas precedent).
3. `docs/templates/hyper-glyphy-pandoc.tex` — project-level template,
   extended copy of the Sumerian one (which stays untouched): same
   structure, body font chosen at plan time for combined IAST + polytonic
   Greek coverage (candidates verified against installed fonts; cuneiform
   auto-switch preserved).
4. `docs/templates/build_findings.sh` — one-command build
   (`pandoc --template … --pdf-engine=xelatex …`), idempotent.
5. `shared/scripts/findings_figures.py` — deterministic matplotlib script
   reading ONLY tracked inputs (`shared/results/myth_study.json`, the
   per-slot `final_output/metadata.json`, and numbers pinned as module
   constants where a JSON is gitignored — each such constant carries a
   provenance comment naming its journal entry), emitting the 4 figures to
   `docs/findings/figures/*.pdf` (committed).

## The 4 figures

1. **Procrustes convergence:** per-slot val cosines v1 and v2 (sumerian /
   hittite / greek / sanskrit) against the pre-registered 0.12/0.20 bands —
   the ~0.115 structural ceiling made visible.
2. **Akkadian alpha fix:** Gemma val sweep with v1's top-1 pick (α=10⁴) vs
   v2's plateau pick (α=0.01) and the dictionary-stratum consequence
   (19.9 → 55.2).
3. **RSA ladder matrix:** the six slot pairs — ladder K, ρ, and exhaustive
   p — as an annotated matrix; the powered sumerian–sanskrit K=5 cell
   visually distinguished.
4. **Vṛtra control:** observed profile-correlation percentile vs the
   1000-draw null for both sub-controls, with the 90/75 bands drawn and
   the midrank/tie-block caveat in the caption.

Figure styling follows the dataviz guidance (loaded at implementation
time); fonts in figures must render IAST correctly.

## Paper outline (section contract)

- **Abstract** — the arc in ~200 words: honest eval, six slots, anchor
  lever retired by a pre-registered experiment, suite v2 improvements,
  first powered Plane-B null, first pre-registered positive (band-edge).
- **1 Introduction** — project premise; the 2026-07-06 honest-eval reset
  as the methodological foundation; pre-registration as the arc's
  discipline.
- **2 Data & methods** — six slots table (corpus, lexicon, sizes);
  pipeline sketch; stratified CSLS suite (strata definitions, candidate
  restriction + gold-OOV conditioning); dual Ridge targets; the Procrustes
  read-out protocol and its bands; myth-study planes, theme ladders,
  permutation power, and both pre-registered rules.
- **3 Results** — 3.1 suite v2 table (per-slot, both targets, incl. gold
  OOV) + fig 2 + the memorization/generalization trade stated; 3.2 the
  anchor-quality experiment: sanskrit numbers, the ≤0.12 verdict VERBATIM,
  fig 1; 3.3 document-level: Gate 1 pass, Gate 2 measured fail, the
  hubness diagnostic (partial; Theogony centroid-norm 1.83rd pctile);
  3.4 myth K=5: fig 3, fig 4, both read-out sentences VERBATIM with bands
  restated, Kumarbi v1→v2 stability, translation deltas, fingerprints.
- **4 Discussion & limitations** — what within-language structure supports
  vs what cross-language geometry cannot do; band-edge honesty (tie-block
  width); OOV conditioning of headline accuracies; ladder discreteness;
  single-sub-control nature of the positive.
- **5 Reproducibility** — GitHub (commit-pinned), HF mirror + `suite-v1`
  tag, `requirements.lock.txt`, one-command artifact bolt-on.
- **References** — data-source citations (DCS, CDSL/MW, Diorisis, Perseus
  LSJ, ORACC, ETCSL, TLHdig, TLA, GloVe, EmbeddingGemma), house style.

All verdict sentences appear verbatim as measured; no claim in the paper
may exceed the journal's wording.

## Build & fonts

XeLaTeX via pandoc, template as in Deliverables. Font requirement: one
body font covering Latin + IAST diacritics + polytonic Greek (candidates
checked at plan time from installed system fonts; the template's cuneiform
auto-switch handles cuneiform spans). The build script fails loudly on
pandoc/xelatex errors; the rendered PDF gets a page-by-page visual check
for fallback boxes (missing glyphs) as an explicit verification step.

## Verification

- `findings_figures.py`: deterministic (no timestamps/randomness); smoke
  test asserts the 4 files are produced and non-empty
  (`shared/tests/test_findings_figures.py`); figure values read from
  tracked JSONs at test time where available.
- Numeric integrity: the task review traces every number in the paper to
  the tracked results JSONs / journal entries (the discipline that caught
  three real errors this arc). Verdict sentences byte-checked.
- Full pytest green per commit; build script run is part of the
  implementation tasks, not left to the reader.

## Docs & workflow

Branch `findings-pdf`; spec → plan (pin fonts, figure data paths, exact
numbers table for the writer) → subagent-driven execution. On ship: README
Recent-findings bullet + dual md/PDF link (Anomaly Atlas pattern), journal
one-liner noting the paper's existence, memory update.
