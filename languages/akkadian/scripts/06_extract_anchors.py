"""
Anchor Extraction: Build Akkadian-English word pairs from ORACC lemma streams.

Source: ORACC project glosses (concatenated from ob_literary_lemmas.json,
ob_letters_lemmas.json, dcclt_lemmas.json).

Mirrors languages/sumerian/scripts/06_extract_anchors.py, with the source
normalizer swapped to akkadian_normalize.

Notes for v2: eBL integration was deferred — the eBL bulk endpoint returns only
IDs (20k strings), and per-word queries against /api/words/{id} are not viable
at scale for v1. ORACC-only anchors are the proven pattern (Sumerian achieves
52% top-1 with this approach).
"""
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token  # noqa: E402

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
DATA_DICTS = Path(__file__).parent.parent / "data" / "dictionaries"

JUNK_ENGLISH = {
    "x", "xx", "0", "00", "1", "n", "c", "e", "i", "u", "s", "unmng", "cf",
}


def _filter_gloss(gw: str) -> bool:
    if not gw or gw in JUNK_ENGLISH:
        return False
    if len(gw) <= 2:
        return False
    if gw.isdigit():
        return False
    if gw.startswith("~"):
        return False
    return True


def extract_oracc_anchors(lemmas: list[dict], min_occurrences: int = 5) -> list[dict]:
    """Extract Akkadian-English pairs from ORACC lemmatization data.

    Counts (citation_form, gloss) and (surface_form, gloss) pairs and emits
    one anchor per pair that meets the occurrence threshold.

    Mirrors Sumerian's extract_epsd2_anchors.
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    for lemma in lemmas:
        gw = (lemma.get("gw") or "").strip().lower()
        if not _filter_gloss(gw):
            continue
        cf = normalize_akkadian_token((lemma.get("cf") or "").strip())
        form = normalize_akkadian_token((lemma.get("form") or "").strip())
        if cf:
            pair_counts[(cf, gw)] += 1
        if form and form != cf:
            pair_counts[(form, gw)] += 1

    anchors: list[dict] = []
    for (form_norm, gw), count in pair_counts.items():
        if count < min_occurrences:
            continue
        confidence = min(0.95, 0.5 + (count / 100))
        anchors.append({
            "akkadian": form_norm,
            "english": gw,
            "confidence": round(confidence, 4),
            "frequency": count,
            "source": "ORACC",
        })
    return sorted(anchors, key=lambda a: a["confidence"], reverse=True)


def _load_oracc_lemmas() -> list[dict]:
    """Concatenate Akkadian lemmas across all three ORACC dumps."""
    out = []
    for name in ("ob_literary_lemmas.json", "ob_letters_lemmas.json", "dcclt_lemmas.json"):
        path = DATA_RAW / name
        if path.exists():
            with open(path) as f:
                out.extend(json.load(f))
    return out


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_DICTS.mkdir(parents=True, exist_ok=True)

    oracc_lemmas = _load_oracc_lemmas()
    print(f"Loaded {len(oracc_lemmas)} ORACC lemma records")

    # Cache the merged ORACC lemma stream for downstream tools (coverage diagnostic etc.)
    with open(DATA_DICTS / "oracc_lemmas.json", "w", encoding="utf-8") as f:
        json.dump(oracc_lemmas, f, ensure_ascii=False, indent=2)

    anchors = extract_oracc_anchors(oracc_lemmas, min_occurrences=5)

    output_path = DATA_PROCESSED / "english_anchors.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(anchors, f, ensure_ascii=False, indent=2)

    print(f"Total anchors: {len(anchors)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
