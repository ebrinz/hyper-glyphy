# hyper-glyphy Experiment Journal

Cross-language experiment log. Reverse chronological — newest at the top.

## Recent findings (newest first)

- **2026-05-10 — Akkadian v1.2 second gap-closing pass: +7.36pp top-1 (21.66% → 29.02%).** Four more levers after v1.1: L4 global lemma-surface expansion (+3.73pp, the largest single lift in v1.2), L5-refined subword inference with training-only OOV partition (+1.19pp), L6a corpus expansion #2 to 62 ORACC projects / 3M tokens (+1.74pp), and L6b DCCLT bridge anchor bootstrapping (FALSIFIED, -3.55pp, reverted). Top-10 reached 57.08% (Sumerian 65.99%) — coverage gap nearly closed. Top-1 gap to Sumerian (29.02% vs 52.13%) is now an alignment-precision problem, not coverage. See [slot journal](../languages/akkadian/docs/EXPERIMENT_JOURNAL.md).

- **2026-05-10 — Akkadian v1.1 gap-closing pass: +4.91pp top-1 (16.75% → 21.66%).** Executed three improvement levers (mimation wiring, FastText min_count change, SB pretraining corpus). The dominant lever was L3 (corpus expansion); Sumerian's W2b-style normalization win did not exist for Akkadian (the per-slot coverage diagnostic was attribution-decisive). Identified remaining levers (subword inference at eval time, lemma-surface expansion) projected to add another +5-12pp. See [slot journal](../languages/akkadian/docs/EXPERIMENT_JOURNAL.md).

- **2026-05-09 — Akkadian slot v1 shipped:** Third language slot. OB-period scope, ORACC-only anchor lexicon (eBL pivoted — see slot journal). Whitened-Gemma top-1 **16.75%** (vs Sumerian 52.13%, Egyptian 32.35%). 50,636 DCCLT bridge pairs scaffolded for v2 cross-lingual experiment. The corpus is smaller (712k tokens vs 2.8M) and anchor coverage thinner (44% vs 65% valid) than Sumerian — see [`languages/akkadian/docs/EXPERIMENT_JOURNAL.md`](../languages/akkadian/docs/EXPERIMENT_JOURNAL.md) for the full writeup and identified levers (normalization audit is highest-leverage). Design: [spec](superpowers/specs/2026-05-09-akkadian-slot-design.md), [plan](superpowers/plans/2026-05-09-akkadian-slot.md).
