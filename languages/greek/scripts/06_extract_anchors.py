"""
Greek Anchor Extraction: Diorisis lemmas + LSJ glosses.

Pipeline:
1. Load Diorisis lemmas (10M token-lemma records from 820 texts).
2. Load LSJ glosses (90k Greek→English entries from Perseus LSJ XML).
3. Join: for each Diorisis lemma's `cf`, normalize and look up in LSJ
   keyed by `lemma_norm`. Use the first LSJ gloss as the anchor's English.
4. From the LSJ gloss (e.g., "not to be injured, inviolable"), extract the
   first English content word that exists in the english_gemma_768d cache
   vocab — that becomes the anchor's `english` field.

No multilingual-Gemma translation step needed (unlike Hittite) — LSJ
glosses are native English.

Mirrors Akkadian/Hittite anchor schema: {greek, english, confidence,
frequency, source}.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from languages.greek.scripts.greek_normalize import normalize_greek_token  # noqa: E402

GRC_DIR = Path(__file__).parent.parent
DATA_RAW = GRC_DIR / "data" / "raw"
DATA_PROCESSED = GRC_DIR / "data" / "processed"
DATA_DICTS = GRC_DIR / "data" / "dictionaries"

SHARED_MODELS = _ROOT / "shared" / "models"
ENG_GEMMA_PATH = SHARED_MODELS / "english_gemma_768d.npz"

MIN_OCCURRENCES = 5
STOP_WORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with",
    "not", "no", "be", "is", "are", "was", "were", "as", "or", "and",
    "but", "if", "so", "do", "did", "have", "has", "had", "from", "into",
    "out", "up", "down", "over", "under", "between", "during", "before",
    "after", "above", "below", "any", "some", "all", "each", "every",
    "one", "two", "three", "four", "five",
}

_WORD_RE = re.compile(r"[a-z][a-z'\-]*")


def _load_gloss_first_english(eng_vocab_set: set[str], gloss: str) -> str | None:
    """Return the first English content word in `gloss` that exists in eng_vocab.

    Handles multi-word glosses like "not to be injured" -> "injured".
    Strips hyphens to also try compound matches.
    """
    if not gloss:
        return None
    lowered = gloss.lower()
    for word in _WORD_RE.findall(lowered):
        if word in STOP_WORDS:
            continue
        if word in eng_vocab_set:
            return word
        # Try hyphen-removed form (e.g., "ear-ring" -> "earring")
        if "-" in word:
            joined = word.replace("-", "")
            if joined in eng_vocab_set:
                return joined
    return None


def _load_english_gemma_vocab() -> set[str]:
    data = np.load(str(ENG_GEMMA_PATH))
    return {str(w) for w in data["vocab"]}


def build_lsj_index(lsj_entries: list[dict]) -> dict[str, dict]:
    """Index LSJ entries by lemma_norm for fast lookup."""
    return {e["lemma_norm"]: e for e in lsj_entries}


def extract_anchors(
    lemmas: list[dict],
    lsj_index: dict[str, dict],
    eng_vocab_set: set[str],
    min_occurrences: int = MIN_OCCURRENCES,
) -> list[dict]:
    """Build (greek, english) anchors via Diorisis-LSJ join."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    lsj_hits = 0
    lsj_misses = 0
    gloss_no_eng = 0

    for lemma in lemmas:
        cf = (lemma.get("cf") or "").strip()
        form = (lemma.get("form") or "").strip()
        cf_norm = normalize_greek_token(cf)
        form_norm = normalize_greek_token(form)
        if not cf_norm:
            continue

        lsj = lsj_index.get(cf_norm)
        if not lsj:
            lsj_misses += 1
            continue
        lsj_hits += 1

        english = _load_gloss_first_english(eng_vocab_set, lsj.get("gloss_first", ""))
        if not english:
            # Try subsequent glosses if first didn't match vocab
            for g in lsj.get("glosses", [])[1:5]:
                english = _load_gloss_first_english(eng_vocab_set, g)
                if english:
                    break
        if not english:
            gloss_no_eng += 1
            continue

        # Register both citation and surface forms
        surfaces: set[str] = set()
        if cf_norm:
            surfaces.add(cf_norm)
        if form_norm and form_norm != cf_norm:
            surfaces.add(form_norm)
        for surface in surfaces:
            pair_counts[(surface, english)] += 1

    print(f"  LSJ join: {lsj_hits} hits, {lsj_misses} misses (cf not in LSJ)")
    print(f"  Glosses with no in-vocab English: {gloss_no_eng}")

    anchors: list[dict] = []
    for (greek_form, eng), count in pair_counts.items():
        if count < min_occurrences:
            continue
        confidence = min(0.95, 0.5 + (count / 100))
        anchors.append({
            "greek": greek_form,
            "english": eng,
            "confidence": round(confidence, 4),
            "frequency": count,
            "source": "Diorisis+LSJ",
        })
    return sorted(anchors, key=lambda a: a["confidence"], reverse=True)


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    with open(DATA_RAW / "greek_lemmas.json") as f:
        lemmas = json.load(f)
    print(f"Loaded {len(lemmas)} Diorisis lemma records")

    with open(DATA_DICTS / "lsj_glosses.json") as f:
        lsj_entries = json.load(f)
    print(f"Loaded {len(lsj_entries)} LSJ entries")

    lsj_index = build_lsj_index(lsj_entries)

    eng_vocab_set = _load_english_gemma_vocab()
    print(f"English Gemma vocab: {len(eng_vocab_set)} entries")

    anchors = extract_anchors(lemmas, lsj_index, eng_vocab_set)
    output_path = DATA_PROCESSED / "english_anchors.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(anchors, f, ensure_ascii=False, indent=2)

    print(f"\nTotal anchors: {len(anchors)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
