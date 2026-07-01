"""
Anchor Extraction: Build Sumerian-English word pairs from two sources.

Source 1 (ePSD2): ORACC lemma data provides citation forms with English guide words.
Source 2 (ETCSL co-occurrence): Parallel Sumerian/English text analysis.

Output: english_anchors.json with fields: sumerian, english, confidence, source, frequency
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from languages.sumerian.scripts.sumerian_normalize import normalize_sumerian_token  # noqa: E402

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
DATA_DICTS = Path(__file__).parent.parent / "data" / "dictionaries"


def extract_epsd2_anchors(lemmas: list[dict], min_occurrences: int = 5) -> list[dict]:
    """
    Extract Sumerian-English pairs from ORACC lemmatization data.

    Deduplicates by (cf, gw) and filters by occurrence count.
    """
    pair_counts = Counter()
    pair_lemmas: dict[tuple[str, str], set[str]] = {}
    for lemma in lemmas:
        gw = lemma.get("gw", "").strip().lower()
        if not gw:
            continue
        # Count both citation form and surface form as potential anchors
        cf = normalize_sumerian_token(lemma.get("cf", "").strip())
        form = normalize_sumerian_token(lemma.get("form", "").strip())
        lemma_key = cf or form
        if cf:
            pair_counts[(cf, gw)] += 1
            pair_lemmas.setdefault((cf, gw), set()).add(lemma_key)
        if form and form != cf:
            pair_counts[(form, gw)] += 1
            pair_lemmas.setdefault((form, gw), set()).add(lemma_key)

    # Junk English values to filter out
    junk_english = {
        "x", "xx", "0", "00", "1", "n", "c", "e", "i", "u", "s",
        "unmng", "cf",
    }

    anchors = []
    for (cf, gw), count in pair_counts.items():
        if count >= min_occurrences:
            # Filter: skip junk English, short words, purely numeric
            if gw in junk_english:
                continue
            if len(gw) <= 2:
                continue
            if gw.isdigit():
                continue
            if gw.startswith("~"):
                continue

            confidence = min(0.95, 0.5 + (count / 100))
            anchors.append({
                "sumerian": cf,
                "english": gw,
                "confidence": round(confidence, 4),
                "frequency": count,
                "source": "ePSD2",
                "lemmas": sorted(pair_lemmas[(cf, gw)]),
            })

    return anchors


def extract_cooccurrence_anchors(
    parallel_lines: list[dict],
    min_cooccurrences: int = 3,
    min_confidence: float = 0.3,
) -> list[dict]:
    """
    Extract anchors from parallel Sumerian-English lines via co-occurrence.

    Calculates P(english|sumerian) = co-occurrence_count / sumerian_word_total_count
    """
    stop_words = {"the", "of", "and", "in", "to", "a", "is", "was", "for", "on",
                  "with", "his", "her", "its", "their", "he", "she", "it", "they",
                  "that", "this", "by", "from", "at", "an", "be", "has", "had",
                  "not", "but", "who", "which", "as", "or", "if", "my", "your"}

    cooc = defaultdict(Counter)
    sum_counts = Counter()

    for line in parallel_lines:
        trans = line.get("transliteration", "")
        transl = line.get("translation", "")
        if not trans or not transl:
            continue

        sum_words = set(re.findall(r"[a-zA-Z\u0161\u0160\u1E2B\u1E2A\u1E6D\u1E6C\u1E63\u1E62\u011D\u011C]+\d*", trans.lower()))
        eng_words = set(re.findall(r"[a-z]+", transl.lower())) - stop_words

        for sw in sorted(sum_words):
            sum_counts[sw] += 1
            for ew in sorted(eng_words):
                cooc[sw][ew] += 1

    anchors = []
    for sw, eng_counts in cooc.items():
        total = sum_counts[sw]
        if total < min_cooccurrences:
            continue

        best_ew, best_count = eng_counts.most_common(1)[0]
        confidence = best_count / total

        if confidence >= min_confidence:
            anchors.append({
                "sumerian": sw,
                "english": best_ew,
                "confidence": round(confidence, 4),
                "frequency": total,
                "source": "ETCSL",
                "lemmas": [sw],
            })

    return anchors


def merge_anchors(dict_anchors: list[dict], cooc_anchors: list[dict]) -> list[dict]:
    """Merge dictionary and co-occurrence anchors, keeping higher confidence."""
    best = {}

    for anchor in dict_anchors + cooc_anchors:
        key = anchor["sumerian"]
        if key not in best or anchor["confidence"] > best[key]["confidence"]:
            best[key] = anchor

    return sorted(best.values(), key=lambda a: a["confidence"], reverse=True)


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_DICTS.mkdir(parents=True, exist_ok=True)

    lemma_path = DATA_RAW / "oracc_lemmas.json"
    if lemma_path.exists():
        with open(lemma_path) as f:
            lemmas = json.load(f)
        dict_anchors = extract_epsd2_anchors(lemmas, min_occurrences=5)
        print(f"ePSD2 anchors: {len(dict_anchors)}")

        with open(DATA_DICTS / "epsd2_entries.json", "w", encoding="utf-8") as f:
            json.dump(dict_anchors, f, ensure_ascii=False, indent=2)
    else:
        print("No ORACC lemma data found, skipping ePSD2 anchors")
        dict_anchors = []

    etcsl_path = DATA_RAW / "etcsl_texts.json"
    if etcsl_path.exists():
        with open(etcsl_path) as f:
            etcsl_lines = json.load(f)
        parallel = [l for l in etcsl_lines if l.get("translation")]
        cooc_anchors = extract_cooccurrence_anchors(parallel, min_cooccurrences=3, min_confidence=0.3)
        print(f"ETCSL co-occurrence anchors: {len(cooc_anchors)}")
    else:
        print("No ETCSL data found, skipping co-occurrence anchors")
        cooc_anchors = []

    merged = merge_anchors(dict_anchors, cooc_anchors)

    output_path = DATA_PROCESSED / "english_anchors.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nMerged anchors: {len(merged)}")
    print(f"From ePSD2: {sum(1 for a in merged if a['source'] == 'ePSD2')}")
    print(f"From ETCSL: {sum(1 for a in merged if a['source'] == 'ETCSL')}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
