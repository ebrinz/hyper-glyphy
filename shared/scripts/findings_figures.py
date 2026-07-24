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
        ax.annotate(label, (len(slots) - 0.05, band), fontsize=8,
                    va="bottom", ha="left", color="#555555")
    ax.set_xticks(list(xs), [s.capitalize() for s in slots])
    ax.set_xlim(-0.6, len(slots) + 1.5)
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
