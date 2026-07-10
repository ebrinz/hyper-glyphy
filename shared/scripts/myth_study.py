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
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
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


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _cos(a, b):
    return float(_unit(np.asarray(a, dtype=np.float64))
                 @ _unit(np.asarray(b, dtype=np.float64)))


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


