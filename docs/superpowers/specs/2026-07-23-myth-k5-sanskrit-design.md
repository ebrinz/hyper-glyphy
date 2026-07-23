# Myth Study K=5 — Sanskrit Fourth Slot & Suite-v2 Re-measurement

**Date:** 2026-07-23
**Status:** Approved
**Goal:** Add Sanskrit (Rigveda / Atharvaveda / Upaniṣads) as the myth
study's fourth slot with a full five-theme roster — creating the study's
first K=5 theme ladders (exhaustive permutation min p = 1/120 ≈ 0.008) —
re-measure the whole study on the suite-v2 embedding spaces, and read out
two pre-registered results: the powered ladder RSA and a new Indra-Vṛtra ↔
Illuyanka positive control.

## Background

The myth study (docs/myth_study_plan.md; shipped 2026-07-12) measures
cross-tradition thematic structure via Plane B (native-space RSA over theme
ladders), a doc-level positive control (Kumarbi ↔ Theogony), translation
delta, and concept fingerprints. Its documented weakness is ladder
coarseness: existing slot pairs share at most K=4 themes (Hittite-Sumerian,
min p 0.042); most share K=3 (min p 0.167 — unfalsifiable). Sanskrit is the
only corpus in the project that can fill all five themes (cosmogonic,
hymnic, wisdom, royal_control, magical), and DCS's chapter granularity is
hymn-level, so rosters pin cleanly. All three per-slot inputs the study
needs already exist for Sanskrit under suite v2
(`fused_embeddings_1536d.npz`, `sanskrit_aligned_gemma_vectors.npz`,
`english_gemma_whitened_768d.npz`). Suite v2 (journal 2026-07-19) changed
every slot's embeddings, so the v1 myth numbers are stale regardless; this
project re-measures everything on v2.

## Scope

- Surgical extension of `shared/scripts/myth_study.py` (Approach A) +
  Sanskrit registration in `shared/scripts/doc_eval.py`.
- Curated five-theme Sanskrit roster (exact DCS IDs pinned in the
  implementation plan from the `sanskrit_texts.json` inventory).
- Full re-run on suite-v2 spaces, all four slots, 6 slot pairs.
- Two pre-registered read-outs (below), journal entry, roster JSON refresh.

**Out of scope (recorded):** slot-registry refactor (Approach B — deferred
to Coptic prep, when a fifth slot would amortize it); Plane A parallel
retrieval (retired); any doc_eval benchmark change; Coptic; word-level
suite changes.

## Code changes (Approach A — surgical)

All in `shared/scripts/myth_study.py` unless noted:

1. `SLOTS = ("sumerian", "hittite", "greek", "sanskrit")`.
2. `slot_pairs = list(itertools.combinations(SLOTS, 2))` replaces the
   hardcoded 3-pair list (yields 6 pairs; deterministic order from SLOTS).
3. `build_roster()` gains a `sanskrit` branch (constants + rules below),
   including a `SANSKRIT_MERGES`-style merge for the Vṛtra document
   (modeled on `HITTITE_MERGES`).
4. IE-gradient block: keeps its existing three named pair keys and adds
   `rho_sanskrit_hittite`, `rho_sanskrit_greek`, `rho_sanskrit_sumerian`;
   the IE-vs-non-IE contrast statement is recomputed over the enlarged set
   (IE pairs: hittite-greek, sanskrit-hittite, sanskrit-greek).
5. New `vrtra_control` result block using the existing `doc_profile` +
   `percentile_in_null` machinery: profile of the merged Vṛtra doc vs
   (a) Illuyanka (Hittite) and (b) Theogony (Greek), each against the same
   same-genre bootstrap null construction as the Kumarbi control
   (`N_NULL_DRAWS = 1000`, seed unchanged).
6. `doc_eval.py::_slot_documents`: register
   `sanskrit → languages/sanskrit/data/raw/sanskrit_texts.json` with
   `normalize_sanskrit_token` (generic p_number+lines path, like
   hittite/greek).
7. Concept fingerprints: no code change — Sanskrit is picked up by the
   existing per-slot loop once in `SLOTS`; its whitened-Gemma cache must
   contain all 10 concept words (verified at plan time; if any is missing
   the existing `fingerprint_status` mechanism records it).

## Sanskrit roster (five themes)

DCS chapters are hymn-granular (`p_number = dcs-<text_id>-<chapter_id>`),
so rosters pin chapter-level p_numbers; multi-chapter works are grouped by
`text_name`. Selection rules (exact IDs pinned in the plan against the real
inventory; counts follow the existing per-theme convention of ~5 docs):

- **cosmogonic:** RV 10.129 (Nasadiya), 10.90 (Puruṣa), 10.121
  (Hiraṇyagarbha), 10.190, plus a merged `vrtra` doc = RV 1.32 + companion
  Vṛtra/Indra-combat hymns (candidates 1.80, 2.12; final list pinned in
  the plan). The merge mirrors `HITTITE_MERGES` concatenation.
- **hymnic:** 5 longest RV hymns outside the cosmogonic roster.
- **wisdom:** 5 longest Upaniṣads present in DCS (Aitareya confirmed
  present; remainder pinned from inventory), each grouped across its
  chapters into one doc by `text_name`.
- **royal_control:** 5 royal charms/hymns (AV royal-consecration material,
  e.g. AV 3.3/3.4/4.8 class; pinned in the plan).
- **magical:** 5 longest AV incantation chapters not already used by
  royal_control.

Gate: any pinned roster doc whose in-vocab token count is zero (the
existing `doc_centroid → None → dropped_docs` path) is a stop-and-surface
before analysis, not a silent drop.

## Pre-registered read-outs (written before any run; no post-hoc adjustment)

1. **K=5 ladder RSA.** The K=5 pair is **sanskrit-sumerian** (the only two
   slots filling all five themes; Hittite lacks wisdom, so sanskrit-hittite
   caps at K=4, min p 0.042 — reported alongside). For these pairs: if the
   shared theme ladder reaches its maximum (K=5 resp. K=4), an exhaustive
   permutation p ≤ 0.05 with positive Spearman ρ = the study's first
   adequately-powered Plane-B positive, stated as such; p > 0.05 = a real,
   reportable null ("thematic geometry does not detectably align"), stated
   as such. No lever/verdict implications beyond the study itself.
2. **Vṛtra positive control.** Percentile of the Vṛtra profile correlation
   in the same-genre null: **≥ 90th** ⇒ "supports the IE combat-myth
   link"; **≤ 75th** ⇒ "fails, consistent with the Kumarbi-control
   finding"; between ⇒ "inconclusive". Both sub-controls (vs Illuyanka, vs
   Theogony) reported; the verdict sentence applies per sub-control,
   verbatim to the journal against these bands.

## Re-run semantics

Entire study recomputed on suite-v2 exports for all four slots.
`shared/results/myth_study.json` and `myth_study_roster.json` are
overwritten (v1 values remain in the journal's 2026-07-12 entries and
reproduce from the HF `suite-v1` tag). The journal entry reports: per-pair
RSA (ladder K, ρ, exhaustive p), both positive-control families (Kumarbi
re-measured on v2 + the new Vṛtra bands verbatim), translation delta,
fingerprints, dropped-doc accounting, and an explicit v1→v2 comparison
paragraph for the previously-measured quantities.

## Error handling

Repo convention, fail loudly: missing Sanskrit npz inputs → the study's
existing load errors; pinned roster IDs absent from `sanskrit_texts.json`
→ raise with the missing IDs listed (new check in the sanskrit roster
branch); zero-in-vocab pinned docs → stop-and-surface.

## Testing

- `shared/tests/test_myth_study_roster.py` (new): sanskrit roster branch
  determinism on a fixture corpus (pinned IDs found, vrtra merge
  concatenates, missing-ID raise); pair enumeration = 6 and contains all
  sanskrit pairs; existing-slot roster entries unchanged on the fixture.
- Existing myth-study behavior rides on unchanged functions; full pytest
  green per commit.
- Analysis integrity: seed and `N_NULL_DRAWS` unchanged; the run is
  deterministic given the spaces.

## Docs & workflow

Branch `myth-k5`; spec → plan (pin DCS roster IDs + verify concept-word
coverage in the whitened cache) → subagent-driven execution → run →
journal + README research-progress bullet + memory. HF mirror: results
JSONs are gitignored; `shared/results/*.json` ARE tracked (myth_study.json
precedent) — commit the refreshed study JSONs with the run task.
