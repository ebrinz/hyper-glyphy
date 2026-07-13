# Sanskrit Sixth Slot — Design

**Date:** 2026-07-13
**Status:** Approved
**Goal:** Ship Sanskrit as the sixth language slot on the proven pipeline (DCS
corpus, Monier-Williams anchors, FastText + fused 1536d, dual-target Ridge,
stratified CSLS suite), and use it — as the best-resourced slot buildable — to
answer the anchor-quality question the Procrustes remap raised, under a
pre-registered interpretation rule.

## Background

The Procrustes remap (journal 2026-07-13) showed Gate 2's cross-language
failure is not map geometry: semi-orthogonal maps with zero projection loss
still fail, and val cosines sit at 0.06–0.12 across slots. The remaining
named lever is "a stronger anchor set." Sanskrit is the strongest anchor
situation available — DCS is large AND gold-lemmatized, Monier-Williams is a
high-quality digitized lexicon with native English glosses — so the slot
doubles as the stronger-anchor experiment: if the best-resourced slot also
lands in the 0.06–0.12 band, anchor quality was never the constraint.

## Scope

- Full DCS corpus (Vedic through Classical — matches the Greek precedent of
  one slot spanning the whole tradition).
- Monier-Williams (Cologne CDSL) as the anchor lexicon — native English
  glosses, no translation hop.
- Word-level pipeline through production export, per-text corpus saved with
  DCS text IDs (Atharvaveda/Rigveda hymn granularity preserved for the
  future myth-study rerun), plus the Procrustes fit for the anchor read-out.
- **Tokenization is the Greek convention: FastText trains on the inflected
  FORM stream** (sandhi-resolved word forms from the conllu), anchors keyed
  on lemma. Deliberate, to keep the anchor-quality comparison against the
  other five slots unconfounded.

**Out of scope (recorded):** myth-study K=5 rerun (own project);
lemma-stream FastText variant (a future, explicitly-labeled second-pass
experiment); Nasadiya/creation-theme roster work; any doc-level benchmark.

## Architecture

```
DCS conllu (git clone, CC BY)          MW XML (Cologne CDSL)
        │                                      │
01_parse_dcs.py                        02_parse_mw.py
  → data/raw/sanskrit_texts.json         → data/raw/mw_glosses.json
    [{p_number: DCS text id,               {headword_iast: [english glosses]}
      lines, source}]                      (SLP1 keys → IAST at parse time)
  → data/raw/sanskrit_lemmas.json
    [{form, cf: lemma, pos, lang:"san"}]
        │
04_deduplicate_corpus.py … 10_export_production.py   (Greek-canonical clones)
  05: sanskrit_normalize on FORM stream → corpus for FastText
  06: DCS lemma ↔ MW headword join → english_anchors.json
  07: FastText 768d on inflected forms   08: fuse → 1536d
  09/09b: dual-target Ridge (GloVe / whitened-Gemma), lemma-group split seed 42
  10: production export
        │
shared/scripts/eval_suite (CSLS, 50k)  +  procrustes_align SLOTS += sanskrit
```

New code is three files: `01_parse_dcs.py`, `02_parse_mw.py`,
`sanskrit_normalize.py`. Scripts 04–10 are sed-clones of the Greek
canonicals (the repo's verified pattern), with slot-name substitutions only.

## Data sources & acquisition

- **DCS:** documented `git clone --depth 1` of the official DCS GitHub
  repository into `languages/sanskrit/data/raw/` (exact URL pinned in the
  implementation plan). Openly licensed (CC BY). The conllu export carries
  sandhi-resolved FORM and gold LEMMA per token, with per-text IDs.
- **Monier-Williams:** Cologne CDSL digitization (~180k entries), XML
  download into `data/raw/` (exact URL pinned in the plan). `key1`
  headwords are SLP1.
- Acquisition is a documented manual step in the slot README, matching the
  Greek/Diorisis precedent — no speculative fetcher script.

## Normalization — `sanskrit_normalize.py`

- Canonical form: lowercase IAST, Unicode NFC.
- DCS conllu is already IAST; normalize case/NFC only.
- MW SLP1 → IAST at parse time via a small deterministic in-repo mapping
  table (no new dependency); covers the full SLP1 set including digraph
  expansions (A→ā, i/I, u/U, f→ṛ, F→ṝ, x→ḷ, X→ḹ, E→ai, O→au, M→ṃ, H→ḥ,
  K→kh, G→gh, C→ch, J→jh, W→ṭh, Q→ḍh, T→th, D→dh, P→ph, B→bh, N→ṅ, Y→ñ,
  R→ṇ, w→ṭ, q→ḍ, S→ś, z→ṣ, L→ḻ, ~→m̐).

## Anchor extraction — 06

Byte-for-byte the Greek/LSJ recipe:

1. Load DCS lemma records; for each lemma `cf`, normalized-IAST lookup into
   MW glosses.
2. From the first MW gloss, take the first English content word present in
   the `english_gemma_768d` cache vocab as the anchor's `english`.
3. Schema matches the other slots: `{sanskrit, english, confidence,
   frequency, source, lemmas: [cf]}` — the `lemmas` field drives
   `group_split`'s union-find.
4. Anchor validity at fit time requires the lemma form in FastText vocab
   (trained on the FORM stream), same as Greek.

**Gloss hit-rate gate (PGM lesson):** 06 reports the DCS-lemma→MW hit rate
before any FastText compute. Under ~40% (Hittite territory) is a
stop-and-surface, not a silent continue.

## Evaluation

- **Word-level:** standard suite — lemma-group split (seed 42, near-surface
  edges), val-selected alpha from the widened grid, stratified CSLS over 50k
  candidates (dictionary / interpolation / zero-shot), leak check expected
  0.00%. Results join the README suite table and journal.
- **Anchor-quality read-out:** after export, add
  `"sanskrit": {"surface_key": "sanskrit"}` to `procrustes_align.SLOTS` and
  run the fitter (same two-variant protocol, val selection, test split
  untouched).

**Pre-registered interpretation rule (written before any fit; gates
interpretation only — the slot ships on the word-level suite regardless):**

- Procrustes val cosine **≥ 0.20** ⇒ anchor quality WAS a binding constraint
  for the other slots; the stronger-anchors lever stays live.
- **≤ 0.12** (the existing slots' band or below) ⇒ anchors were never the
  constraint ⇒ retire the stronger-anchors lever, and with it the last
  named route to Plane A.
- **Between 0.12 and 0.20** ⇒ inconclusive, stated as such.
- No threshold adjustment after measurement; the verdict wording goes in the
  journal verbatim against these bands.

## Error handling

Fail loudly, repo convention: missing DCS clone / MW XML → clear error
naming the documented fetch step; dim asserts in cloned 09/09b unchanged;
unparseable conllu lines are counted and reported in 01's output summary
(parse-loss %), never silently dropped.

## Testing

All under `languages/sanskrit/tests/`:

- `test_01_dcs.py` — conllu fixture → texts/lemmas JSON shape,
  sandhi-resolved FORM extraction, parse-loss accounting.
- `test_02_mw.py` — SLP1→IAST table (full digraph set), gloss
  extraction from an MW XML fixture.
- `test_normalize.py` — IAST canonicalization idempotence, NFC.
- `test_06_anchors.py` — join logic, content-word gloss selection,
  schema including the `lemmas` field.
- Cloned 04–10 verified by sed-diff against the Greek canonicals (the
  eval-redesign T7/T8 pattern), not re-tested line by line.

## Docs & workflow

- Slot README (sources, licenses, fetch steps, pipeline order), journal
  entry with measured suite numbers and the read-out verdict, root README
  table row (and removal of Vedic from the future-slots table on ship).
- Branch `sanskrit-slot`; spec → plan → subagent-driven execution.
