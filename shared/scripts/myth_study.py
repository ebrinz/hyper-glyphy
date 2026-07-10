"""
Myth study, first executable phase: Plane B (native-space second-order RSA)
as the primary instrument, with translation delta and second-order concept
fingerprints. Plane A direct cross-language cosine is NO-GO (Gate 2 FAIL).

Slots in scope: sumerian, hittite, greek. Egyptian/Akkadian are prerequisites.
Honest note: the Greek magical slot is EMPTY — the Greek Magical Papyri (PGM)
are not in the Diorisis corpus.

See: docs/myth_study_plan.md
Usage: python -m shared.scripts.myth_study run
"""
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.scripts.doc_eval import (  # noqa: E402
    ETCSL_PATH,
    _load_space,
    _slot_documents,
    doc_centroid,
    sif_weights,
)

SEED = 42
N_NULL_DRAWS = 1000
MAX_PERMS = 1000
SLOTS = ("sumerian", "hittite", "greek")
THEMES = ("cosmogonic", "hymnic", "wisdom", "royal_control", "magical")
CONCEPTS = ("water", "chaos", "serpent", "name", "fate", "bind", "create",
            "mountain", "flood", "sky")

ROSTER_PATH = _ROOT / "shared" / "results" / "myth_study_roster.json"
RESULTS_PATH = _ROOT / "shared" / "results" / "myth_study.json"

# --- pinned document selections (verified present in the corpora) ---

SUMERIAN_COSMOGONIC = ("c141", "c174", "c111", "c112", "c113")

HITTITE_MERGES = {
    # CTH 344 Song of Kumarbi: main tablet under the join KBo 52.10+ plus fragment
    "kumarbi": ("KBo 52.10+", "KUB 47.56"),
    # CTH 345 Song of Ullikummi
    "ullikummi": ("KBo 26.58", "KBo 26.61"),
    # CTH 321 Illuyanka
    "illuyanka": ("KBo 3.7", "KUB 17.5"),
}
# arkuwar marker fails (syllabic hyphenated transliteration: 0 matches in TLHdig
# lines); these are the standard prayer tablets (CTH 373-377), all verified
# present; KUB 24.3+ content-checked (addresses the Sun-goddess of Arinna).
HITTITE_PRAYERS = ("KUB 24.3+", "KUB 17.21+", "KUB 24.1+", "KUB 30.10", "KUB 24.2")
HITTITE_RITUAL_MARKER = "SISKUR"

GREEK_COSMOGONIC = ("Hesiod (0020) - Theogony (001)",
                    "Hesiod (0020) - Works and Days (002)")
GREEK_THEOGONY = GREEK_COSMOGONIC[0]
GREEK_HYMN_TAG = "(0013)"  # TLG 0013 = Hymni Homerici
GREEK_ROYAL_CONTROL = ("Homer (0012) - Iliad (001)",
                       "Homer (0012) - Odyssey (002)",
                       "Herodotus (0016) - Histories (001)",
                       "Thucydides (0003) - History (001)",
                       "Xenophon (0032) - Anabasis (006)")


# ---------------------------------------------------------------------------
# Corpus loading + roster
# ---------------------------------------------------------------------------

def load_corpora():
    """Tokenized docs per slot plus the raw material roster rules need."""
    docs = _slot_documents()
    with open(ETCSL_PATH) as f:
        etcsl_lines = Counter(r["line_id"].split(".")[0]
                              for r in json.load(f) if r.get("line_id"))
    hittite_path = _ROOT / "languages" / "hittite" / "data" / "raw" / "hittite_texts.json"
    with open(hittite_path) as f:
        hittite_lines = {r["p_number"]: r["lines"] for r in json.load(f)}
    return {"docs": docs, "sumerian_line_counts": dict(etcsl_lines),
            "hittite_lines": hittite_lines}


def _entry(doc_id, tokens, n_lines=None, components=None):
    e = {"doc_id": doc_id, "n_tokens": len(tokens)}
    if n_lines is not None:
        e["n_lines"] = n_lines
    if components is not None:
        e["components"] = list(components)
    return e


def _longest_by_class(prefix, line_counts, docs, n):
    cands = [(c, ln) for c, ln in line_counts.items()
             if c.startswith(prefix) and c in docs]
    cands.sort(key=lambda x: (-x[1], x[0]))
    return cands[:n]


def build_roster(corpora):
    """Deterministic roster: roster[slot][theme] -> list of doc entries.

    Returns (roster, notes, roster_tokens); roster_tokens[slot][doc_id] is the
    token list actually used for centroids (merged docs = concatenation).
    """
    docs = corpora["docs"]
    roster = {s: {t: [] for t in THEMES} for s in SLOTS}
    notes = {s: {t: {"rule": None, "reason_empty": None} for t in THEMES}
             for s in SLOTS}
    roster_tokens = {s: {} for s in SLOTS}

    # --- sumerian ---
    slc = corpora["sumerian_line_counts"]
    sdocs = docs["sumerian"]
    for comp in SUMERIAN_COSMOGONIC:
        if comp not in sdocs:
            raise KeyError(f"pinned sumerian composition missing: {comp}")
        roster["sumerian"]["cosmogonic"].append(
            _entry(comp, sdocs[comp], slc.get(comp)))
        roster_tokens["sumerian"][comp] = sdocs[comp]
    notes["sumerian"]["cosmogonic"]["rule"] = (
        "pinned: c141 Inanna's Descent, c174 Flood/Ziusudra, c111 Enki&Ninhursaga, "
        "c112 Enki&Ninmah, c113 Enki&World Order")
    for theme, prefix, n, rule in (
            ("hymnic", "c4", 5, "5 longest ETCSL class-4 (hymns) by line count"),
            ("wisdom", "c6", 3, "3 longest ETCSL class-6 (proverbs) by line count"),
            ("royal_control", "c2", 5, "5 longest ETCSL class-2 (royal) by line count")):
        for comp, ln in _longest_by_class(prefix, slc, sdocs, n):
            roster["sumerian"][theme].append(_entry(comp, sdocs[comp], ln))
            roster_tokens["sumerian"][comp] = sdocs[comp]
        notes["sumerian"][theme]["rule"] = rule
    notes["sumerian"]["magical"]["reason_empty"] = (
        "inspected ETCSL class 0 and class 3: class 0 compositions are "
        "catalogues/incipit lists, class 3 are letters (u3-na-a-dug4 formula); "
        "no incantation content verified by inspection, so the slot is left empty")

    # --- hittite ---
    hdocs = docs["hittite"]
    hlines = corpora["hittite_lines"]
    selected = set()
    for merged_id, components in HITTITE_MERGES.items():
        toks = []
        for c in components:
            if c not in hdocs:
                raise KeyError(f"pinned hittite tablet missing: {c}")
            toks.extend(hdocs[c])
        n_lines = sum(len(hlines[c]) for c in components)
        roster["hittite"]["cosmogonic"].append(
            _entry(merged_id, toks, n_lines, components))
        roster_tokens["hittite"][merged_id] = toks
        selected.update(components)
    notes["hittite"]["cosmogonic"]["rule"] = (
        "merged joins: kumarbi = KBo 52.10+ + KUB 47.56 (CTH 344), "
        "ullikummi = KBo 26.58 + KBo 26.61 (CTH 345), "
        "illuyanka = KBo 3.7 + KUB 17.5 (CTH 321)")

    for p in HITTITE_PRAYERS:
        if p not in hdocs:
            raise KeyError(f"pinned hittite prayer missing: {p}")
        roster["hittite"]["hymnic"].append(_entry(p, hdocs[p], len(hlines[p])))
        roster_tokens["hittite"][p] = hdocs[p]
        selected.add(p)
    notes["hittite"]["hymnic"]["rule"] = (
        "arkuwar marker unusable (syllabic hyphenated transliteration, 0 matches); "
        "pinned known prayer tablets CTH 373-377, all verified present: "
        + ", ".join(HITTITE_PRAYERS))

    is_ritual = {p: any(HITTITE_RITUAL_MARKER in ln for ln in ls)
                 for p, ls in hlines.items()}
    ritual_cands = [(p, len(hlines[p])) for p in hdocs
                    if is_ritual.get(p) and p not in selected]
    ritual_cands.sort(key=lambda x: (-x[1], x[0]))
    for p, ln in ritual_cands[:5]:
        roster["hittite"]["magical"].append(_entry(p, hdocs[p], ln))
        roster_tokens["hittite"][p] = hdocs[p]
        selected.add(p)
    notes["hittite"]["magical"]["rule"] = (
        f"5 longest texts (by line count) containing the Sumerogram ritual marker "
        f"'{HITTITE_RITUAL_MARKER}' ({sum(is_ritual.values())} texts match), "
        f"excluding docs already selected")

    royal_cands = [(p, len(hlines[p])) for p in hdocs
                   if p not in selected and not is_ritual.get(p)]
    royal_cands.sort(key=lambda x: (-x[1], x[0]))
    for p, ln in royal_cands[:5]:
        roster["hittite"]["royal_control"].append(_entry(p, hdocs[p], ln))
        roster_tokens["hittite"][p] = hdocs[p]
        selected.add(p)
    notes["hittite"]["royal_control"]["rule"] = (
        "5 longest non-myth texts (by line count), excluding SISKUR-marked ritual "
        "texts and docs already selected (brief's 'simply 5 longest non-myth' option)")
    notes["hittite"]["wisdom"]["reason_empty"] = (
        "no wisdom selection defined for hittite in this phase")

    # --- greek ---
    gdocs = docs["greek"]
    for g in GREEK_COSMOGONIC:
        if g not in gdocs:
            raise KeyError(f"pinned greek text missing: {g}")
        roster["greek"]["cosmogonic"].append(_entry(g, gdocs[g]))
        roster_tokens["greek"][g] = gdocs[g]
    notes["greek"]["cosmogonic"]["rule"] = "pinned: Hesiod Theogony + Works and Days"

    hymns = sorted(((g, len(t)) for g, t in gdocs.items() if GREEK_HYMN_TAG in g),
                   key=lambda x: (-x[1], x[0]))
    for g, _n in hymns[:5]:
        roster["greek"]["hymnic"].append(_entry(g, gdocs[g]))
        roster_tokens["greek"][g] = gdocs[g]
    notes["greek"]["hymnic"]["rule"] = (
        "5 longest Homeric Hymns entries (TLG author 0013) by token count")

    for g in GREEK_ROYAL_CONTROL:
        if g not in gdocs:
            raise KeyError(f"pinned greek text missing: {g}")
        roster["greek"]["royal_control"].append(_entry(g, gdocs[g]))
        roster_tokens["greek"][g] = gdocs[g]
    notes["greek"]["royal_control"]["rule"] = (
        "pinned substantial epic/prose narrative controls: Iliad, Odyssey, "
        "Herodotus Histories, Thucydides History, Xenophon Anabasis")
    notes["greek"]["wisdom"]["reason_empty"] = (
        "Theognis and Hesiod fragments absent from the Diorisis extract")
    notes["greek"]["magical"]["reason_empty"] = (
        "Greek Magical Papyri (PGM) are not in the Diorisis literary corpus; the "
        "Greek magical slot is EMPTY and every magical-theme comparison excludes Greek")

    return roster, notes, roster_tokens


# ---------------------------------------------------------------------------
# Plane B primitives
# ---------------------------------------------------------------------------

def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _cos(a, b):
    return float(_unit(np.asarray(a, dtype=np.float64))
                 @ _unit(np.asarray(b, dtype=np.float64)))


def doc_profile(doc_id, doc_vecs, members_by_theme, ladder):
    """Cosine profile of a doc against leave-one-out theme centroids."""
    out = []
    for theme in ladder:
        rows = [doc_vecs[d] for d in members_by_theme[theme]
                if d != doc_id and d in doc_vecs]
        if not rows:
            raise ValueError(
                f"theme '{theme}' has no members after excluding {doc_id!r}")
        out.append(_cos(doc_vecs[doc_id], np.mean(rows, axis=0)))
    return np.array(out)


def theme_sim_matrix(theme_cents, ladder):
    """K x K cosine matrix over theme centroids in ladder order."""
    M = np.array([_unit(np.asarray(theme_cents[t], dtype=np.float64))
                  for t in ladder])
    return M @ M.T


def upper_tri(M):
    i, j = np.triu_indices(len(M), 1)
    return np.asarray(M)[i, j]


def rsa_permutation(A, B, max_perms=MAX_PERMS, rng=None):
    """Spearman over upper triangles + theme-label permutation p.

    Exhaustive over all K! label permutations of B when K! <= max_perms
    (identity included, so min p = 1/K!); otherwise max_perms random draws.
    """
    rho = float(spearmanr(upper_tri(A), upper_tri(B)).statistic)
    k = len(A)
    n_all = math.factorial(k)
    B = np.asarray(B)
    if n_all <= max_perms:
        perms = list(itertools.permutations(range(k)))
        exhaustive = True
    else:
        rng = rng or np.random.default_rng(SEED)
        perms = [tuple(rng.permutation(k)) for _ in range(max_perms)]
        exhaustive = False
    ge = 0
    for p in perms:
        idx = np.asarray(p)
        r = spearmanr(upper_tri(A), upper_tri(B[np.ix_(idx, idx)])).statistic
        if r >= rho - 1e-12:
            ge += 1
    return {"rho": round(rho, 4), "p": round(ge / len(perms), 4),
            "n_perms": len(perms), "exhaustive": exhaustive}


def percentile_in_null(obs, null):
    """Midrank percentile of obs within null (100 = above all null draws)."""
    null = np.asarray(null, dtype=np.float64)
    below = float((null < obs).sum())
    ties = float((null == obs).sum())
    return float((below + 0.5 * ties) / len(null) * 100)


def rank_delta_report(M_native, M_aligned, ids, top=5):
    """Spearman between the two upper triangles + largest per-pair rank deltas."""
    un, ua = upper_tri(M_native), upper_tri(M_aligned)
    rho = float(spearmanr(un, ua).statistic)
    rn, ra = rankdata(un), rankdata(ua)
    delta = rn - ra
    i, j = np.triu_indices(len(ids), 1)
    order = np.argsort(-np.abs(delta), kind="stable")[:top]
    entries = [{"pair": (ids[i[k]], ids[j[k]]),
                "rank_native": float(rn[k]), "rank_aligned": float(ra[k]),
                "delta": float(delta[k])} for k in order]
    return rho, entries


# ---------------------------------------------------------------------------
# Study orchestration
# ---------------------------------------------------------------------------

def _slot_doc_ids(roster, slot):
    """Roster doc ids of a slot in deterministic THEMES-then-entry order."""
    return [e["doc_id"] for t in THEMES for e in roster[slot][t]]


def _members_by_theme(roster, slot):
    return {t: [e["doc_id"] for e in roster[slot][t]] for t in THEMES
            if roster[slot][t]}


def _centroids_for_slot(roster_tokens, slot, npz_path, weights):
    vocab, vectors = _load_space(npz_path)
    cents, dropped = {}, []
    for did, toks in roster_tokens[slot].items():
        v = doc_centroid(toks, vocab, vectors, weights)
        if v is None:
            dropped.append(did)
        else:
            cents[did] = v
    del vectors
    return cents, dropped


def _concept_vectors(slot):
    path = _ROOT / "languages" / slot / "models" / "english_gemma_whitened_768d.npz"
    if not path.exists():
        return None, f"MISSING cache: {path}"
    data = np.load(str(path), allow_pickle=True)
    lookup = {str(w): i for i, w in enumerate(data["vocab"])}
    missing = [c for c in CONCEPTS if c not in lookup]
    if missing:
        return None, f"concepts missing from cache vocab: {missing}"
    vecs = {c: data["vectors"][lookup[c]].astype(np.float64) for c in CONCEPTS}
    return vecs, None


def _shared_ladder(roster, slot_a, slot_b):
    return [t for t in THEMES if roster[slot_a][t] and roster[slot_b][t]]


def run():
    rng = np.random.default_rng(SEED)
    print("Loading corpora ...")
    corpora = load_corpora()
    roster, notes, roster_tokens = build_roster(corpora)

    ROSTER_PATH.parent.mkdir(exist_ok=True)
    with open(ROSTER_PATH, "w") as f:
        json.dump({"seed": SEED, "slots": roster, "notes": notes}, f, indent=2,
                  ensure_ascii=False)
    print(f"Roster written: {ROSTER_PATH}")
    for slot in SLOTS:
        cells = ", ".join(f"{t}={len(roster[slot][t])}" for t in THEMES)
        print(f"  {slot:<9} {cells}")

    # SIF weights from slot-wide token counts (full corpus of each slot)
    weights = {s: sif_weights(Counter(t for d in corpora["docs"][s].values()
                                      for t in d)) for s in SLOTS}

    native_cents, aligned_cents, fingerprints = {}, {}, {}
    dropped_docs, fingerprint_status = {}, {}
    for slot in SLOTS:
        print(f"Computing centroids for {slot} ...")
        native_path = (_ROOT / "languages" / slot / "models"
                       / "fused_embeddings_1536d.npz")
        aligned_path = (_ROOT / "languages" / slot / "final_output"
                        / f"{slot}_aligned_gemma_vectors.npz")
        native_cents[slot], drop_n = _centroids_for_slot(
            roster_tokens, slot, native_path, weights[slot])
        aligned_cents[slot], drop_a = _centroids_for_slot(
            roster_tokens, slot, aligned_path, weights[slot])
        dropped_docs[slot] = sorted(set(drop_n) | set(drop_a))
        cvecs, err = _concept_vectors(slot)
        fingerprint_status[slot] = err or "ok"
        if cvecs is not None:
            fingerprints[slot] = {
                did: [round(_cos(v, cvecs[c]), 4) for c in CONCEPTS]
                for did, v in aligned_cents[slot].items()}

    results = {"seed": SEED, "n_null_draws": N_NULL_DRAWS,
               "roster_ref": str(ROSTER_PATH.relative_to(_ROOT)),
               "dropped_docs": dropped_docs}

    # ---- Plane B (iii): slot-pair structural RSA on theme ladders ----
    pair_rsa = {}
    slot_pairs = [("hittite", "greek"), ("hittite", "sumerian"),
                  ("greek", "sumerian")]
    theme_cents = {}
    for slot in SLOTS:
        members = _members_by_theme(roster, slot)
        theme_cents[slot] = {
            t: np.mean([native_cents[slot][d] for d in ms
                        if d in native_cents[slot]], axis=0)
            for t, ms in members.items()}
    for a, b in slot_pairs:
        ladder = _shared_ladder(roster, a, b)
        dropped = [t for t in THEMES if t not in ladder
                   and (roster[a][t] or roster[b][t])]
        A = theme_sim_matrix(theme_cents[a], ladder)
        B = theme_sim_matrix(theme_cents[b], ladder)
        out = rsa_permutation(A, B, rng=rng)
        out.update({"ladder": ladder, "themes_dropped_from_ladder": dropped})
        pair_rsa[f"{a}-{b}"] = out
    results["slot_pair_rsa"] = pair_rsa
    rho_hg = pair_rsa["hittite-greek"]["rho"]
    rho_hs = pair_rsa["hittite-sumerian"]["rho"]
    rho_gs = pair_rsa["greek-sumerian"]["rho"]
    results["ie_gradient"] = {
        "rho_hittite_greek": rho_hg, "rho_hittite_sumerian": rho_hs,
        "rho_greek_sumerian": rho_gs,
        "hittite_greek_highest": bool(rho_hg > rho_hs and rho_hg > rho_gs)}

    # ---- Plane B (ii): doc-level positive control (kumarbi et al. vs Theogony) ----
    ladder_hg = _shared_ladder(roster, "hittite", "greek")
    members_h = _members_by_theme(roster, "hittite")
    members_g = _members_by_theme(roster, "greek")
    prof_h = {d: doc_profile(d, native_cents["hittite"], members_h, ladder_hg)
              for d in _slot_doc_ids(roster, "hittite")
              if d in native_cents["hittite"]}
    prof_g = {d: doc_profile(d, native_cents["greek"], members_g, ladder_hg)
              for d in _slot_doc_ids(roster, "greek")
              if d in native_cents["greek"]}

    theog = prof_g[GREEK_THEOGONY]
    kumarbi = prof_h["kumarbi"]
    noncosmo_h = [d for t in THEMES if t != "cosmogonic"
                  for e in roster["hittite"][t]
                  if (d := e["doc_id"]) in prof_h]
    noncosmo_g = [d for t in THEMES if t != "cosmogonic"
                  for e in roster["greek"][t]
                  if (d := e["doc_id"]) in prof_g]
    null = []
    for _ in range(N_NULL_DRAWS):
        d = noncosmo_h[rng.integers(len(noncosmo_h))]
        null.append(spearmanr(prof_h[d], theog).statistic)
    for _ in range(N_NULL_DRAWS):
        d = noncosmo_g[rng.integers(len(noncosmo_g))]
        null.append(spearmanr(kumarbi, prof_g[d]).statistic)
    null = [r for r in null if not np.isnan(r)]

    positive_control = {"ladder": ladder_hg, "n_null": len(null),
                        "null_mean": round(float(np.mean(null)), 4),
                        "pairs": {}}
    for hdoc in ("kumarbi", "ullikummi", "illuyanka"):
        rho = float(spearmanr(prof_h[hdoc], theog).statistic)
        positive_control["pairs"][f"{hdoc}-theogony"] = {
            "rho": round(rho, 4),
            "percentile_in_null": round(percentile_in_null(rho, null), 2)}
    results["positive_control"] = positive_control

    # ---- Translation delta (within-language) ----
    tdelta = {}
    for slot in SLOTS:
        ids = [d for d in _slot_doc_ids(roster, slot)
               if d in native_cents[slot] and d in aligned_cents[slot]]
        Nn = np.array([_unit(np.asarray(native_cents[slot][d], np.float64))
                       for d in ids])
        Na = np.array([_unit(np.asarray(aligned_cents[slot][d], np.float64))
                       for d in ids])
        rho, deltas = rank_delta_report(Nn @ Nn.T, Na @ Na.T, ids, top=5)
        tdelta[slot] = {"n_docs": len(ids), "spearman": round(rho, 4),
                        "largest_rank_deltas": deltas}
    results["translation_delta"] = tdelta

    # ---- Concept fingerprints (second-order Plane A) ----
    fp_section = {"concepts": list(CONCEPTS), "status": fingerprint_status,
                  "theme_mean_fingerprints": {}, "cosmogonic_cross_slot": {}}
    theme_fp = {}
    for slot in fingerprints:
        theme_fp[slot] = {}
        for t in THEMES:
            fps = [fingerprints[slot][e["doc_id"]] for e in roster[slot][t]
                   if e["doc_id"] in fingerprints[slot]]
            if fps:
                theme_fp[slot][t] = [round(float(x), 4)
                                     for x in np.mean(fps, axis=0)]
        fp_section["theme_mean_fingerprints"][slot] = theme_fp[slot]
    for a, b in slot_pairs:
        if a in theme_fp and b in theme_fp and \
                "cosmogonic" in theme_fp[a] and "cosmogonic" in theme_fp[b]:
            r = float(spearmanr(theme_fp[a]["cosmogonic"],
                                theme_fp[b]["cosmogonic"]).statistic)
            fp_section["cosmogonic_cross_slot"][f"{a}-{b}"] = round(r, 4)
    results["concept_fingerprints"] = fp_section

    results["notes"] = {
        "plane_a": "direct cross-language cosine NO-GO (Gate 2 FAIL); concept "
                   "fingerprints are second-order and within-language only",
        "greek_magical": notes["greek"]["magical"]["reason_empty"],
        "sumerian_magical": notes["sumerian"]["magical"]["reason_empty"],
        "ladder_coarseness": "K=3 shared ladders give 3 upper-triangle values and "
                             "6 exhaustive label permutations (min p = 0.1667)"}

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- summary ----
    print("\n=== Positive control (Plane B doc-level, native space) ===")
    print(f"ladder: {ladder_hg}  null draws kept: {len(null)} "
          f"(mean rho {positive_control['null_mean']})")
    for k, v in positive_control["pairs"].items():
        print(f"  {k:<22} rho={v['rho']:+.3f}  percentile={v['percentile_in_null']:.1f}")
    print("\n=== Slot-pair structural RSA (theme ladders, native space) ===")
    for k, v in pair_rsa.items():
        print(f"  {k:<18} rho={v['rho']:+.3f}  p={v['p']:.3f} "
              f"(n_perms={v['n_perms']}, exhaustive={v['exhaustive']}) "
              f"ladder={v['ladder']}")
    print(f"  IE gradient (hittite-greek highest): "
          f"{results['ie_gradient']['hittite_greek_highest']}")
    print("\n=== Translation delta (native vs aligned, within-language) ===")
    for slot, v in tdelta.items():
        print(f"  {slot:<9} n={v['n_docs']:>2}  spearman={v['spearman']:+.3f}")
    print("\n=== Cosmogonic concept-fingerprint cross-slot Spearman ===")
    for k, v in fp_section["cosmogonic_cross_slot"].items():
        print(f"  {k:<18} rho={v:+.3f}")
    print(f"\nNOTE: {notes['greek']['magical']['reason_empty']}")
    print(f"Saved to: {RESULTS_PATH}")
    return results


def main():
    import argparse

    p = argparse.ArgumentParser(description="Myth study (Plane B primary).")
    p.add_argument("command", choices=("run",))
    args = p.parse_args()
    if args.command == "run":
        run()


if __name__ == "__main__":
    main()
