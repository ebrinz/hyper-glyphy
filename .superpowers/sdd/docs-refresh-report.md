# Docs Refresh Report — 2026-07-06

Branch: `worktree-agent-a12d357dc007fe13f`

---

## Commits (4 new)

| SHA | Subject |
|-----|---------|
| `11213eb` | docs(readme): fix identity, links, tables, and research progress |
| `1846f4b` | docs(roadmap): update statuses for all shipped slots + fix links/naming |
| `bb0365b` | docs(languages): add pre-fix notes, fix stale numbers, add Greek docs |
| `40ec497` | docs(research-vision): reframe current work for five slots + eval-integrity context |

---

## Findings: fixed vs deferred

| # | Finding | Status | Notes |
|---|---------|--------|-------|
| 1 | README broken links (anomaly_atlas, sumerian_cosmogony) | **FIXED** | Paths corrected to `languages/sumerian/docs/`; `test -f` verified |
| 2 | Banner alt "Cuneiformy", tagline, project tree rooted at cuneiformy/ | **FIXED** | Alt → "hyper-glyphy"; tagline updated; tree rewritten as five-slot |
| 3 | Main results table (32.35/Egyptian stale, two-slot) | **FIXED** | Five-slot pre-fix table; Egyptian 34.57/33.42 from metadata.json |
| 4 | "Currently shipped" table (wrong Egyptian numbers, Greek unstarted) | **FIXED** | Table updated; Egyptian 34.57%/33.42%; Greek = scaffolded, run pending |
| 5 | Missing research progress entries (Akkadian v1–v1.3, Hittite v1, Greek G1–G4) | **FIXED** | Entries added; Sumerian historical journal cross-linked |
| 6 | README experiment journal link (historical Sumerian findings) | **FIXED** | `languages/sumerian/docs/EXPERIMENT_JOURNAL.md` now in header line |
| 7 | ROADMAP item 2 stale status (Egyptian "queued, blocked") | **FIXED** | "partially shipped 2026-05-04"; atlas/cosmogony noted as outstanding |
| 8 | ROADMAP item 3 stale blocker | **FIXED** | References atlas/cosmogony deliverables, not "item 2" wholesale |
| 9 | ROADMAP item 4 stale (Akkadian/Hittite/Greek status, wrong corpus) | **FIXED** | Shipped slots listed; Greek scaffolded; TLHdig named correctly |
| 10 | ROADMAP "Shipped" list ends at Phase α | **FIXED** | Items 11–15 added: Egyptian β, Akkadian v1–v1.3, Hittite v1, Greek scaffold, eval-integrity fix |
| 11 | ROADMAP target tree (hattusian/, jiaguwen/, comparative/ don't exist) | **FIXED** | Tree reflects actual five dirs; aspirational slots removed from tree |
| 12 | ROADMAP "Hittite Dictionary project" → TLHdig | **FIXED** | In item 4 prose and item 13 of Shipped list |
| 13 | ROADMAP journal links point to sumerian journal | **FIXED** | Intro and Shipped section both point to `docs/EXPERIMENT_JOURNAL.md` (repo-wide) + Sumerian-specific |
| 14 | Egyptian journal: missing 2026-05-04 shipped-results entry + 2026-07-06 note | **FIXED** | Entry added with PCA-256/alpha params from metadata.json; invalidation banner at top |
| 15 | Egyptian README: no results section | **FIXED** | Pre-fix results table + eval-integrity note added |
| 16 | Hittite journal: Egyptian column stale (n/a / 32.35); missing invalidation note | **FIXED** | Column updated to 34.57/33.42 (pre-fix labeled); banner added at top |
| 17 | Invalidation pointer note at top of Sumerian and Akkadian journals | **FIXED** | Banners added to both |
| 18 | Greek: no README, empty docs/ | **FIXED** | README (pipeline status, corpus, anchors, scripts) + docs/EXPERIMENT_JOURNAL.md stub (G1–G4 entry) created |
| 19 | Akkadian README: missing 01b_scrape_oracc_sb.py; no pre-fix note | **FIXED** | Step added with note; pre-fix banner above results |
| 20 | Hittite README: references deleted ridge_alpha_sweep.py | **FIXED** | Step removed from quick-start; note added in spec section that it was retired |
| 21 | RESEARCH_VISION: stale "Cuneiformy", 11.24% accuracy, Akkadian/Greek as future | **FIXED** | Phase 1 rewritten for five slots; eval-integrity framing added; Phase 3 annotated with shipped/scaffolded status |
| 22 | Sumerian anchor count 8,558 vs 8,584 (metadata) | **DEFERRED** | Sumerian journal not otherwise touched; cosmetic-only; "if touched" condition not met |
| 23 | ROADMAP "Hattusian (Hittite)" naming | **FIXED** | Handled by item 11 (target tree rewrite) and item 9 (item 4 rewrite) |

**Fixed: 22 / Deferred: 1** (finding 22 — cosmetic, condition "if touched" not met)

---

## Verification outputs

### `grep -rn "32.35" README.md docs/ languages/*/README.md languages/*/docs/` (excluding superpowers/)

*Zero hits* in user-facing docs. Remaining hits in superpowers/plans and sumerian/docs/sumerian_cosmogony.md are all explicitly historical (dated experimental entries or historical baseline comparisons in research documents dated before the Egyptian ship).

### `grep -rn "ridge_alpha_sweep" README.md docs/ROADMAP.md docs/RESEARCH_VISION.md languages/*/README.md`

*Zero hits* in user-facing top-level docs and language READMEs.

### `grep -rn "Cuneiformy\|cuneiformy" README.md`

*Zero hits.*

### `python -m pytest -q`

**294 passed, 6 warnings** — suite unchanged.

---

## Anything in the findings file that turned out to be wrong

- Finding 3 says "Egyptian 5,360 anchors" in the old table — confirmed stale (metadata shows 6,060 valid / 8,170 total).
- Finding 4 says Egyptian shows "GloVe 32.35% under a Gemma column header" — confirmed in pre-edit README (the single-column "Whitened-Gemma top-1" with 32.35% which is actually the GloVe V15 baseline).
- All file path claims in the findings were accurate (anomaly_atlas_findings.md, sumerian_cosmogony.md both confirmed in `languages/sumerian/docs/`).
- Finding 12 says corpus is "via Hittite Dictionary project" — found in ROADMAP item 4 as written; corrected to TLHdig (Hethitologie Portal Mainz / Zenodo).
- Greek anchor count 106,260 and 10.2M-token corpus: confirmed from commit cbb4e54 message.
