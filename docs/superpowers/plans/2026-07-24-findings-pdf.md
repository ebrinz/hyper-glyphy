# Arc Findings PDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the paper-structured arc findings document as markdown + a LaTeX-born PDF (pandoc/XeLaTeX house pipeline) with 4 deterministic matplotlib figures.

**Architecture:** Project-level pandoc template (extended copy of the Sumerian one) + one-command build script; a figure script reading tracked JSONs (gitignored values pinned as provenance-commented constants); a hand-written paper whose every number is review-traced to committed sources; a final build + page-by-page glyph check.

**Tech Stack:** pandoc 3.9, XeLaTeX (installed), matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-07-24-findings-pdf-design.md`

## Global Constraints

- Branch `findings-pdf`. Repo root. Full `pytest -q` green before every commit (baseline 390).
- Fonts (pinned 2026-07-24 against installed system fonts): body `Times New Roman` (macOS-guaranteed Latin Extended Additional for IAST + Greek Extended for polytonic), mono `Menlo`, cuneiform `NotoSansCuneiform-Regular.ttf` copied to `docs/fonts/` (template Path `./docs/fonts/`). The Sumerian template `languages/sumerian/docs/templates/cuneiformy-pandoc.tex` is NOT modified.
- Figure palette (validated via the dataviz six-checks script, light surface, 2026-07-24): blue `#0072B2`, orange `#E69F00`, green `#009E73`, pink `#CC79A7`; contrast WARN on orange/pink is relieved by direct labels (mandatory). Diverging (fig 3): `#0072B2` ↔ neutral `#B0B0B0` ↔ `#E69F00`. One axis per chart; thin marks; direct labels selective; no rainbow.
- Figures deterministic: no timestamps (`metadata={"CreationDate": None}` on savefig), no randomness, fixed sizes; matplotlib default DejaVu fonts (cover IAST + Greek in figures).
- Every number in the paper traces to: the README v2/v1 tables, `docs/EXPERIMENT_JOURNAL.md` entries (2026-07-16/-19/-24), or `shared/results/myth_study.json` — the writer cites nothing from memory; the reviewer re-traces. Verdict sentences appear VERBATIM; no claim may exceed the journal's wording.
- Committed artifacts: paper md, built PDF, 4 figure PDFs, template, build script, cuneiform ttf copy, figure script + smoke test. Nothing else.

---

### Task 1: Template, fonts, build script

**Files:**
- Create: `docs/templates/hyper-glyphy-pandoc.tex` (copy + 1 edit)
- Create: `docs/fonts/NotoSansCuneiform-Regular.ttf` (copy)
- Create: `docs/templates/build_findings.sh`

**Interfaces:**
- Produces: `bash docs/templates/build_findings.sh <input.md> <output.pdf>` — pandoc + xelatex build with mainfont/monofont vars set; consumed by Task 4.

- [ ] **Step 1: Copy assets**

```bash
mkdir -p docs/templates docs/fonts docs/findings/figures
cp languages/sumerian/docs/templates/cuneiformy-pandoc.tex docs/templates/hyper-glyphy-pandoc.tex
cp languages/sumerian/docs/fonts/NotoSansCuneiform-Regular.ttf docs/fonts/
```

In `docs/templates/hyper-glyphy-pandoc.tex`, the cuneiform font line already reads `[Path=./docs/fonts/]` (repo-root relative) — verify with `grep -n "docs/fonts" docs/templates/hyper-glyphy-pandoc.tex`; if it points elsewhere, set it to `[Path=./docs/fonts/]`. No other template edits.

- [ ] **Step 2: Write the build script**

`docs/templates/build_findings.sh`:

```bash
#!/bin/bash
# Build a findings PDF from markdown via the house pandoc/XeLaTeX template.
# Usage (from repo root): bash docs/templates/build_findings.sh <in.md> <out.pdf>
set -euo pipefail
IN=${1:?usage: build_findings.sh <in.md> <out.pdf>}
OUT=${2:?usage: build_findings.sh <in.md> <out.pdf>}
pandoc "$IN" -o "$OUT" \
  --template=docs/templates/hyper-glyphy-pandoc.tex \
  --pdf-engine=xelatex \
  -V mainfont="Times New Roman" \
  -V monofont="Menlo" \
  -V fontsize=11pt \
  -V geometry:margin=1.1in \
  --toc --number-sections
echo "built: $OUT"
```

`chmod +x docs/templates/build_findings.sh`

- [ ] **Step 3: Glyph-proof build (throwaway input)**

Create `/tmp/glyph_test.md` containing exactly:

```markdown
# Glyph test

IAST: ṛṣi kṛṣṇa ahiṃsā Bṛhadāraṇyakopaniṣad saṃsāra ḹ m̐
Greek: ἀθάνατος ῥέει ᾧ ῆ Θεογονία
Mixed: Vṛtra ↔ Illuyanka (ρ=0.4, p ≤ 0.05, −0.333)
```

Run: `bash docs/templates/build_findings.sh /tmp/glyph_test.md /tmp/glyph_test.pdf`
Expected: clean build. Then extract and check text round-trips:

```bash
pdftotext /tmp/glyph_test.pdf - | head -8
```

Expected: the IAST and Greek strings appear intact (no missing-glyph boxes render as gaps/`?` in extraction). Also open/screenshot the PDF and LOOK at it — tofu boxes fail this step. Record the check output in your report. Delete the two /tmp files after.

- [ ] **Step 4: Full pytest + commit**

```bash
pytest -q
git add docs/templates/hyper-glyphy-pandoc.tex docs/templates/build_findings.sh docs/fonts/NotoSansCuneiform-Regular.ttf
git commit -m "feat(docs): project-level findings PDF template + build script (IAST/Greek/cuneiform fonts)"
```

---

### Task 2: `findings_figures.py` + smoke test

**Files:**
- Create: `shared/scripts/findings_figures.py`
- Test: `shared/tests/test_findings_figures.py`

**Interfaces:**
- Produces: `python shared/scripts/findings_figures.py [--outdir docs/findings/figures]` → writes `fig1_procrustes.pdf`, `fig2_akkadian_alpha.pdf`, `fig3_rsa_matrix.pdf`, `fig4_vrtra_control.pdf`. `main(outdir: Path) -> list[Path]` importable for the test.

- [ ] **Step 1: Write the failing smoke test**

`shared/tests/test_findings_figures.py`:

```python
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
_spec = spec_from_file_location("findings_figures",
                                str(_ROOT / "shared" / "scripts" / "findings_figures.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_four_figures_produced(tmp_path):
    paths = _mod.main(tmp_path)
    assert len(paths) == 4
    names = {p.name for p in paths}
    assert names == {"fig1_procrustes.pdf", "fig2_akkadian_alpha.pdf",
                     "fig3_rsa_matrix.pdf", "fig4_vrtra_control.pdf"}
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
```

Run: `pytest shared/tests/test_findings_figures.py -v` — expect FAIL (module absent).

- [ ] **Step 2: Implement**

`shared/scripts/findings_figures.py`:

```python
"""
Deterministic figures for the arc findings paper (spec 2026-07-24).

Inputs: shared/results/myth_study.json (git-tracked) read live; values whose
source JSONs are gitignored are pinned as constants with provenance comments
naming the journal entry that published them. No timestamps, no randomness.

Palette: dataviz-validated (2026-07-24, light surface) — direct labels
mandatory (contrast relief for orange/pink).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent.parent
MYTH_JSON = _ROOT / "shared" / "results" / "myth_study.json"

BLUE, ORANGE, GREEN, PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
NEUTRAL = "#B0B0B0"
_SAVE_KW = dict(bbox_inches="tight", metadata={"CreationDate": None})

# --- Pinned values (source JSONs gitignored; provenance = journal entries) ---
# Procrustes val cosines: journal 2026-07-13 (v1 sum/hit/grk), 2026-07-16 (v1 san),
# 2026-07-19 + myth-K5 run (v2). languages/*/results/procrustes_results.json are gitignored.
PROCRUSTES = {  # slot: (v1, v2)
    "sumerian": (0.1157, 0.1117),
    "hittite": (0.0586, 0.0666),
    "greek": (0.1149, 0.1163),
    "sanskrit": (0.1145, 0.1198),
}
BANDS = (0.12, 0.20)  # pre-registered interpretation bands (spec 2026-07-13)

# Akkadian-Gemma alpha sweep, suite v2 (journal 2026-07-19; alignment results gitignored).
AKK_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4, 1e5]
AKK_TOP5_V2 = [1.18, 1.18, 1.24, 1.24, 1.18, 1.24, 1.18, 1.24, 0.56, 0.00]
AKK_V2_PICK, AKK_V2_DICT = 1e-2, 55.24
AKK_V1_PICK, AKK_V1_DICT = 1e4, 19.85   # v1 top-1 noise pick (journal 2026-07-09/-19)


def fig1_procrustes(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    slots = list(PROCRUSTES)
    xs = range(len(slots))
    for x, slot in zip(xs, slots):
        v1, v2 = PROCRUSTES[slot]
        ax.plot([x, x], [v1, v2], color=NEUTRAL, lw=1, zorder=1)
        ax.scatter([x], [v1], color=BLUE, s=42, zorder=2)
        ax.scatter([x], [v2], color=ORANGE, s=42, zorder=2)
        ax.annotate(f"{v2:.4f}", (x, v2), textcoords="offset points",
                    xytext=(8, 4), fontsize=8)
    for band, label in zip(BANDS, ("retire band (≤0.12)", "binding band (≥0.20)")):
        ax.axhline(band, color=NEUTRAL, lw=1, ls="--")
        ax.annotate(label, (len(slots) - 0.5, band), fontsize=8,
                    va="bottom", ha="right", color="#555555")
    ax.set_xticks(list(xs), [s.capitalize() for s in slots])
    ax.set_ylabel("Procrustes val cosine")
    ax.set_ylim(0, 0.24)
    ax.scatter([], [], color=BLUE, label="suite v1")
    ax.scatter([], [], color=ORANGE, label="suite v2")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    out = outdir / "fig1_procrustes.pdf"
    fig.savefig(out, **_SAVE_KW)
    plt.close(fig)
    return out


def fig2_akkadian_alpha(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(AKK_ALPHAS, AKK_TOP5_V2, color=BLUE, lw=2, marker="o", ms=4,
            label="val top-5 CSLS (v2 selector signal)")
    ax.axvline(AKK_V2_PICK, color=GREEN, lw=1.5)
    ax.annotate(f"v2 pick α={AKK_V2_PICK:g}\ndict 55.24%",
                (AKK_V2_PICK, 1.30), fontsize=8, ha="center", color=GREEN)
    ax.axvline(AKK_V1_PICK, color=PINK, lw=1.5)
    ax.annotate(f"v1 pick α={AKK_V1_PICK:g}\ndict 19.85%",
                (AKK_V1_PICK, 1.30), fontsize=8, ha="center", color=PINK)
    ax.set_xscale("log")
    ax.set_xlabel("Ridge α")
    ax.set_ylabel("val top-5 CSLS (%)")
    ax.set_ylim(0, 1.55)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    out = outdir / "fig2_akkadian_alpha.pdf"
    fig.savefig(out, **_SAVE_KW)
    plt.close(fig)
    return out


def fig3_rsa_matrix(outdir: Path) -> Path:
    r = json.load(open(MYTH_JSON))
    pairs = r["slot_pair_rsa"]
    slots = ["sumerian", "hittite", "greek", "sanskrit"]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for i, a in enumerate(slots):
        for j, b in enumerate(slots):
            if j <= i:
                continue
            entry = pairs.get(f"{a}-{b}") or pairs.get(f"{b}-{a}")
            rho, p, k = entry["rho"], entry["p"], len(entry["ladder"])
            t = max(-1.0, min(1.0, rho or 0.0))
            color = BLUE if t < 0 else ORANGE
            alpha = 0.15 + 0.6 * abs(t)
            ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                       color=color, alpha=alpha))
            weight = "bold" if (a, b) == ("sumerian", "sanskrit") else "normal"
            ax.text(j, i, f"K={k}\nρ={rho:+.2f}\np={p:.3f}",
                    ha="center", va="center", fontsize=8, weight=weight)
    # highlight the powered K=5 cell
    i, j = slots.index("sumerian"), slots.index("sanskrit")
    ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                               fill=False, edgecolor="#333333", lw=2))
    ax.set_xticks(range(len(slots)), [s.capitalize() for s in slots])
    ax.set_yticks(range(len(slots)), [s.capitalize() for s in slots])
    ax.set_xlim(-0.6, len(slots) - 0.4)
    ax.set_ylim(len(slots) - 0.4, -0.6)
    ax.set_title("Plane-B theme-ladder RSA (upper triangle; bold = powered K=5)",
                 fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    out = outdir / "fig3_rsa_matrix.pdf"
    fig.savefig(out, **_SAVE_KW)
    plt.close(fig)
    return out


def fig4_vrtra_control(outdir: Path) -> Path:
    r = json.load(open(MYTH_JSON))
    sc = r["vrtra_control"]["sub_controls"]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.axvspan(90, 100, color=GREEN, alpha=0.12)
    ax.axvspan(0, 75, color=PINK, alpha=0.10)
    ax.axvline(90, color=GREEN, lw=1, ls="--")
    ax.axvline(75, color=PINK, lw=1, ls="--")
    ys = {"vs_illuyanka": 1.0, "vs_theogony": 0.4}
    labels = {"vs_illuyanka": "Vṛtra ↔ Illuyanka", "vs_theogony": "Vṛtra ↔ Theogony"}
    for name, y in ys.items():
        pct = sc[name]["percentile"]
        ax.scatter([pct], [y], color=BLUE, s=60, zorder=3)
        ax.annotate(f"{labels[name]}  {pct:.2f}", (pct, y),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8)
    ax.annotate("supports (≥90)", (95, 0.05), fontsize=7, ha="center", color="#3a7d44")
    ax.annotate("fails (≤75)", (37, 0.05), fontsize=7, ha="center", color="#9c4a6e")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.1, 1.4)
    ax.set_yticks([])
    ax.set_xlabel("midrank percentile in 1000-draw same-genre null")
    ax.spines[["top", "right", "left"]].set_visible(False)
    out = outdir / "fig4_vrtra_control.pdf"
    fig.savefig(out, **_SAVE_KW)
    plt.close(fig)
    return out


def main(outdir: Path | str = _ROOT / "docs" / "findings" / "figures") -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return [fig1_procrustes(outdir), fig2_akkadian_alpha(outdir),
            fig3_rsa_matrix(outdir), fig4_vrtra_control(outdir)]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(_ROOT / "docs" / "findings" / "figures"))
    args = ap.parse_args()
    for p in main(args.outdir):
        print("wrote", p)
```

- [ ] **Step 3: Test green, generate the real figures, eyeball them**

```bash
pytest shared/tests/test_findings_figures.py -v
python shared/scripts/findings_figures.py
```

Open each of the 4 PDFs in `docs/findings/figures/` and LOOK: no label
collisions, IAST renders in fig 4 labels (Vṛtra), K=5 cell highlighted,
band annotations legible. Fix layout constants if needed (report any
adjustments). Verify determinism: run twice, `shasum` both sets — identical.

- [ ] **Step 4: Full pytest + commit**

```bash
pytest -q
git add shared/scripts/findings_figures.py shared/tests/test_findings_figures.py docs/findings/figures/
git commit -m "feat(docs): deterministic findings figures (procrustes, alpha fix, RSA matrix, vrtra control)"
```

---

### Task 3: The paper source

**Files:**
- Create: `docs/findings/hyper-glyphy-findings-2026-07.md`

**Interfaces:**
- Consumes: Task 2's figure paths (referenced as `figures/figN_*.pdf`).
- Produces: the complete paper source, buildable by Task 1's script.

- [ ] **Step 1: Write the paper**

Structure and content contract are the spec's §Paper outline — follow it
section by section. Authoritative sources for every number (read them; cite
nothing from memory):

- Suite v2 + archived v1 tables and footnotes: root `README.md` Results section.
- Arc narrative + all measured details: `docs/EXPERIMENT_JOURNAL.md` entries
  dated 2026-07-16 (Sanskrit slot + retirement verdict), 2026-07-19 (suite
  v2 + A5 + A6), 2026-07-24 (myth K=5). Verdict sentences must be copied
  VERBATIM from these entries.
- Myth-study numbers: `shared/results/myth_study.json` (tracked) directly.
- Data-source facts (corpora, licenses, sizes): per-slot `languages/*/README.md`
  and `data/raw/README.md` manifests.
- Reproducibility section facts: root README "Resources & Reproducibility"
  (HF repo, suite-v1 tag, lockfile, fetch script).

Pandoc specifics: YAML header with `title`, `date: 2026-07-24`,
`abstract: |` block; figures as
`![caption](figures/fig1_procrustes.pdf){width=90%}`; tables as pipe tables
(pandoc renders them via longtable/booktabs already in the template);
fig 4's caption MUST carry the midrank/tie-block caveat (copy the journal's
sentence). Length target 4,500–6,500 words (~15–20 rendered pages with
tables/figures). Cuneiform spans (if any are quoted from the Sumerian
material) use the template's cuneiform font mechanism — check how
`anomaly_atlas_findings.md` marks them and copy that convention; if none
are quoted, that is fine too.

- [ ] **Step 2: Self-check before handoff**

Grep your draft for each verdict sentence and diff it against the journal's
text (byte-match required). Confirm every table cell against the README
tables. List in your report each number you could NOT trace to a committed
source (there should be none — remove or source any stragglers).

- [ ] **Step 3: Full pytest + commit**

```bash
pytest -q
git add docs/findings/hyper-glyphy-findings-2026-07.md
git commit -m "docs: arc findings paper source (full arc, paper register)"
```

---

### Task 4: Build, glyph check, ship links

**Files:**
- Create: `docs/findings/hyper-glyphy-findings-2026-07.pdf` (built, committed)
- Modify: `README.md` (Recent-findings bullet + dual md/PDF link), `docs/EXPERIMENT_JOURNAL.md` (one-line note)

- [ ] **Step 1: Build**

```bash
bash docs/templates/build_findings.sh \
  docs/findings/hyper-glyphy-findings-2026-07.md \
  docs/findings/hyper-glyphy-findings-2026-07.pdf
```

Expected: clean build, no LaTeX errors. If pandoc reports missing
characters, STOP and surface (font gap — do not silently accept tofu).

- [ ] **Step 2: Page-by-page glyph + layout check**

Open the PDF and check EVERY page: no tofu boxes (IAST, Greek), figures
placed and legible, tables not overflowing margins, TOC correct, section
numbers sane. `pdftotext` spot-check the abstract and the Vṛtra section for
intact diacritics. Record page count and any fixes made (fix in source/
template, rebuild, recheck).

- [ ] **Step 3: Ship links**

- `README.md`: Recent-findings bullet (newest-first) announcing the paper,
  linking both `docs/findings/hyper-glyphy-findings-2026-07.md` and the
  `.pdf` (match the Anomaly Atlas dual-link phrasing at README.md:86).
- `docs/EXPERIMENT_JOURNAL.md`: one line under the 2026-07-24 entry noting
  the findings paper and its path.

- [ ] **Step 4: Full pytest + commit**

```bash
pytest -q
git add docs/findings/hyper-glyphy-findings-2026-07.pdf README.md docs/EXPERIMENT_JOURNAL.md
git commit -m "docs: build arc findings PDF + ship links"
```

Then: final whole-branch review (the reviewer re-traces paper numbers to
committed sources — the load-bearing gate for this branch) →
finishing-a-development-branch → memory update.

---

## Self-Review Record

- **Spec coverage:** deliverables 1–5 → T1 (template/fonts/script), T2 (figures + smoke test), T3 (paper), T4 (built PDF + links); 4 figures exactly as spec'd with palette validated and direct-label relief; paper outline enforced as T3's contract with verbatim-verdict + no-memory-citation rules; fonts pinned (Times New Roman/Menlo/Noto Cuneiform); verification = smoke test + determinism shasum + glyph checks + numbers-traced review; out-of-scope respected (no analysis changes, Sumerian template untouched).
- **Placeholder scan:** T3 delegates prose to committed sources by exact path + section contract (the established pattern for run-dependent/writing tasks); no TBDs.
- **Type consistency:** `main(outdir) -> list[Path]` matches test; figure filenames consistent between T2 code, test, and T3's references; build script signature consistent T1↔T4.
