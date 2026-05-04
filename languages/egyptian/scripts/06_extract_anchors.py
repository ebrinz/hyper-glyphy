"""
Anchor Extraction: Normalize heiroglyphy anchor pairs to hyper-glyphy format.

Reads the existing english_anchors.json (8,541 pairs from TLA/Ramses/BBAW)
and normalizes field names, applies quality filters, and deduplicates.

Input format:  {"hieroglyphic": str, "english": str, "german": str, "confidence": float, "frequency": int}
Output format: {"egyptian": str, "english": str, "confidence": float, "frequency": int, "source": str}
"""
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from languages.egyptian.scripts.egyptian_normalize import normalize_egyptian_token  # noqa: E402

_LANG_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = _LANG_ROOT / "data" / "processed"

_JUNK_ENGLISH = {"x", "xx", "0", "00", "1", "n", "c", "e", "i", "u", "s"}


def normalize_anchors(
    raw_anchors: list[dict],
    min_frequency: int = 5,
) -> list[dict]:
    """Normalize heiroglyphy anchors to hyper-glyphy format with quality filters."""
    best = {}

    for anchor in raw_anchors:
        egyptian = normalize_egyptian_token(anchor.get("hieroglyphic", ""))
        english = anchor.get("english", "").strip().lower()
        confidence = anchor.get("confidence", 0.0)
        frequency = anchor.get("frequency", 0)

        if not egyptian or not english:
            continue
        if frequency < min_frequency:
            continue
        if len(english) <= 1:
            continue
        if english in _JUNK_ENGLISH:
            continue
        if english.isdigit():
            continue

        raw_key = anchor.get("hieroglyphic", "").strip()

        if raw_key not in best or confidence > best[raw_key]["confidence"]:
            best[raw_key] = {
                "egyptian": egyptian,
                "egyptian_raw": raw_key,
                "english": english,
                "confidence": round(confidence, 4),
                "frequency": frequency,
                "source": "TLA/Ramses",
            }

    return sorted(best.values(), key=lambda a: a["confidence"], reverse=True)


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    input_path = DATA_PROCESSED / "english_anchors.json"
    print(f"Loading raw anchors from {input_path}")
    with open(input_path) as f:
        raw = json.load(f)
    print(f"Raw anchors: {len(raw)}")

    filtered = normalize_anchors(raw, min_frequency=5)

    output_path = DATA_PROCESSED / "english_anchors_normalized.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Normalized anchors: {len(filtered)}")
    print(f"Filtered: {len(raw) - len(filtered)} removed")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
