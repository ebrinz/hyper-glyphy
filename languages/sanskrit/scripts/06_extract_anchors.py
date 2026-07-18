"""
Sanskrit Anchor Extraction: DCS lemmas + MW glosses.

Pipeline:
1. Load DCS lemmas (token-lemma records from the conllu dump).
2. Load MW glosses (Cologne CDSL Monier-Williams entries, 02_parse_mw.py).
3. Join: for each DCS lemma's `cf`, normalize and look up in MW keyed by
   `lemma_norm`. Use the first usable MW gloss as the anchor's English.
4. From the gloss, extract the first English content word that exists in the
   english_gemma_768d cache vocab — that becomes the anchor's `english`.

Canonical filters (shared `gloss_filters`, suite v2): negation rule
(reject glosses with "not", "no", "without", "never" before first in-vocab
content), cross-reference starters ("see", "cf", "vid"), scaffold words
("having", "relating", etc.), and single-letter words. The caller falls
through to the entry's next gloss on rejection.

Gate (PGM lesson): the DCS-lemma -> MW token-level join hit rate is computed
and persisted to anchor_stats.json BEFORE any FastText compute; below
MIN_HIT_RATE (40%) is a hard SystemExit, never a silent continue.

Mirrors Akkadian/Hittite anchor schema: {sanskrit, english, confidence,
frequency, source, lemmas}.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from languages.sanskrit.scripts.sanskrit_normalize import normalize_sanskrit_token  # noqa: E402
from shared.scripts.gloss_filters import (  # noqa: E402
    check_hit_rate_gate,
    first_english,
    hit_rate_stats,
)

GRC_DIR = Path(__file__).parent.parent
DATA_RAW = GRC_DIR / "data" / "raw"
DATA_PROCESSED = GRC_DIR / "data" / "processed"
DATA_DICTS = GRC_DIR / "data" / "dictionaries"

SHARED_MODELS = _ROOT / "shared" / "models"
ENG_GEMMA_PATH = SHARED_MODELS / "english_gemma_768d.npz"

MIN_OCCURRENCES = 5


def _load_english_gemma_vocab() -> set[str]:
    data = np.load(str(ENG_GEMMA_PATH))
    return {str(w) for w in data["vocab"]}


def build_mw_index(mw_entries: list[dict]) -> dict[str, dict]:
    """Index MW entries by lemma_norm for fast lookup."""
    return {e["lemma_norm"]: e for e in mw_entries}


def extract_anchors(
    lemmas: list[dict],
    mw_index: dict[str, dict],
    eng_vocab_set: set[str],
    min_occurrences: int = MIN_OCCURRENCES,
) -> list[dict]:
    """Build (sanskrit, english) anchors via DCS-MW join."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_lemmas: dict[tuple[str, str], set[str]] = {}
    mw_hits = 0
    mw_misses = 0
    gloss_no_eng = 0

    for lemma in lemmas:
        cf = (lemma.get("cf") or "").strip()
        form = (lemma.get("form") or "").strip()
        cf_norm = normalize_sanskrit_token(cf)
        form_norm = normalize_sanskrit_token(form)
        if not cf_norm:
            continue

        mw = mw_index.get(cf_norm)
        if not mw:
            mw_misses += 1
            continue
        mw_hits += 1

        english = first_english(mw.get("gloss_first", ""), eng_vocab_set)
        if not english:
            # Try subsequent glosses if first didn't match vocab
            for g in mw.get("glosses", [])[1:5]:
                english = first_english(g, eng_vocab_set)
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
            pair_lemmas.setdefault((surface, english), set()).add(cf_norm)

    print(f"  MW join: {mw_hits} hits, {mw_misses} misses (cf not in MW)")
    print(f"  Glosses with no in-vocab English: {gloss_no_eng}")

    anchors: list[dict] = []
    for (sanskrit_form, eng), count in pair_counts.items():
        if count < min_occurrences:
            continue
        confidence = min(0.95, 0.5 + (count / 100))
        anchors.append({
            "sanskrit": sanskrit_form,
            "english": eng,
            "confidence": round(confidence, 4),
            "frequency": count,
            "source": "DCS+MW",
            "lemmas": sorted(pair_lemmas[(sanskrit_form, eng)]),
        })
    anchors = sorted(anchors, key=lambda a: a["confidence"], reverse=True)

    stats = hit_rate_stats(hits=mw_hits, misses=mw_misses,
                           gloss_no_eng=gloss_no_eng, anchors=len(anchors))
    return anchors, stats


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    with open(DATA_RAW / "sanskrit_lemmas.json") as f:
        lemmas = json.load(f)
    print(f"Loaded {len(lemmas)} DCS lemma records")

    with open(DATA_DICTS / "mw_glosses.json") as f:
        mw_entries = json.load(f)
    print(f"Loaded {len(mw_entries)} MW entries")

    mw_index = build_mw_index(mw_entries)

    eng_vocab_set = _load_english_gemma_vocab()
    print(f"English Gemma vocab: {len(eng_vocab_set)} entries")

    anchors, stats = extract_anchors(lemmas, mw_index, eng_vocab_set)
    output_path = DATA_PROCESSED / "english_anchors.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(anchors, f, ensure_ascii=False, indent=2)
    stats_path = DATA_PROCESSED / "anchor_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTotal anchors: {len(anchors)}")
    print(f"Token-level MW join hit rate: {stats['token_hit_rate']:.1%}")
    print(f"Saved to: {output_path} (+ {stats_path.name})")

    check_hit_rate_gate(stats, "MW")


if __name__ == "__main__":
    main()
