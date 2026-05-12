"""
Corpus Deduplication: Merge ETCSL, CDLI, and ORACC texts.

Deduplicates by CDLI P-number. When duplicates exist, keeps the version
with the most transliteration lines. Texts without P-numbers (some ETCSL)
are kept as-is.
"""
import json
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def deduplicate(texts: list[dict]) -> list[dict]:
    """Deduplicate texts by P-number, keeping the version with more lines."""
    result, _ = deduplicate_with_stats(texts)
    return result


def deduplicate_with_stats(texts: list[dict]) -> tuple[list[dict], dict]:
    """Deduplicate and return stats."""
    total_input = len(texts)

    no_p = [t for t in texts if not t.get("p_number")]
    with_p = [t for t in texts if t.get("p_number")]

    best = {}
    for t in with_p:
        p = t["p_number"]
        if p not in best or len(t.get("lines", [])) > len(best[p].get("lines", [])):
            best[p] = t

    result = list(best.values()) + no_p

    stats = {
        "total_input": total_input,
        "total_output": len(result),
        "duplicates_removed": total_input - len(result),
        "texts_without_p_number": len(no_p),
        "unique_p_numbers": len(best),
    }

    return result, stats


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    all_texts = []

    for src_name, src_label in [
        ("greek_texts.json", "Diorisis"),
    ]:
        path = DATA_RAW / src_name
        if path.exists():
            with open(path) as f:
                texts = json.load(f)
            all_texts.extend(texts)
            print(f"{src_label}: {len(texts)} texts")
        else:
            print(f"{src_label}: missing ({path})")

    result, stats = deduplicate_with_stats(all_texts)

    output_path = DATA_PROCESSED / "merged_corpus.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nDeduplication stats: {json.dumps(stats, indent=2)}")
    total_lines = sum(len(t.get("lines", [])) for t in result)
    print(f"Total texts: {len(result)}")
    print(f"Total lines: {total_lines}")


if __name__ == "__main__":
    main()
