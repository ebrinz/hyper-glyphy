"""
DCCLT Bridge Anchors (L6b): bootstrap Akkadian-English anchors via Sumerian's
alignment as a Rosetta stone.

For each Sumerian-Akkadian word pair from DCCLT, look up the Sumerian word in
Sumerian's aligned-Gemma space, find the top-1 English nearest neighbor, and
emit (akkadian_cf, english_top1) as a new anchor IF Sumerian's confidence
(cosine similarity) exceeds COSINE_THRESHOLD.

Output: merges into data/processed/english_anchors.json (preserving ORACC anchors).

Prerequisite: Sumerian vocab JSON. The Sumerian slot's production vocab is pickled
(its project convention); we cache a JSON copy at /tmp on first run to avoid the
pickle deserialization surface.

Run AFTER 06_extract_anchors.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from languages.akkadian.scripts.akkadian_normalize import normalize_akkadian_token  # noqa: E402

AKK_DIR = Path(__file__).parent.parent
PAIRS_PATH = AKK_DIR / "data" / "processed" / "sumerian_akkadian_pairs.jsonl"
ANCHORS_PATH = AKK_DIR / "data" / "processed" / "english_anchors.json"

SUM_DIR = ROOT / "languages" / "sumerian" / "final_output"
SUM_ALIGNED_PATH = SUM_DIR / "sumerian_aligned_gemma_vectors.npz"
SUM_VOCAB_JSON = Path("/tmp/sumerian_vocab_for_bridge.json")
SUM_VOCAB_PKL = SUM_DIR / "sumerian_aligned_vocab.pkl"

ENG_GEMMA_PATH = ROOT / "shared" / "models" / "english_gemma_whitened_768d.npz"

COSINE_THRESHOLD = 0.5


def ensure_sumerian_vocab_json() -> list[str]:
    """Load Sumerian vocab from JSON cache; populate it from the pickle once if absent.

    Done out-of-process via subprocess so the importer of this module never touches
    pickle. The cache file is a flat list of strings (safe to load).
    """
    if not SUM_VOCAB_JSON.exists():
        script = (
            "import pickle, json, sys\n"
            f"with open({str(SUM_VOCAB_PKL)!r}, 'rb') as f:\n"
            "    vocab = pickle.load(f)\n"
            f"with open({str(SUM_VOCAB_JSON)!r}, 'w', encoding='utf-8') as f:\n"
            "    json.dump([str(w) for w in vocab], f, ensure_ascii=False)\n"
        )
        subprocess.run([sys.executable, "-c", script], check=True)
    with open(SUM_VOCAB_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_sumerian_aligned():
    data = np.load(str(SUM_ALIGNED_PATH))
    vectors = data["vectors"].astype(np.float32)
    vocab = ensure_sumerian_vocab_json()
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return vectors, vocab, word_to_idx


def load_english_gemma():
    data = np.load(str(ENG_GEMMA_PATH))
    vectors = data["vectors"].astype(np.float32)
    vocab = [str(w) for w in data["vocab"]]
    return vectors, vocab


def main():
    print(f"Loading Sumerian aligned vectors from {SUM_ALIGNED_PATH}")
    sum_vectors, sum_vocab, sum_w2i = load_sumerian_aligned()
    print(f"Sumerian vocab: {len(sum_vocab)}, dim {sum_vectors.shape[1]}")

    print(f"Loading English Gemma whitened from {ENG_GEMMA_PATH}")
    eng_vectors, eng_vocab = load_english_gemma()
    print(f"English vocab: {len(eng_vocab)}, dim {eng_vectors.shape[1]}")

    eng_norms = np.linalg.norm(eng_vectors, axis=1, keepdims=True) + 1e-12
    eng_normed = eng_vectors / eng_norms

    print(f"Loading DCCLT pairs from {PAIRS_PATH}")
    pairs = []
    with open(PAIRS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} DCCLT pairs")

    bootstrap_anchors = []
    sum_misses = 0
    cosine_misses = 0
    seen: set = set()

    for pair in pairs:
        sum_cf = (pair.get("sumerian_cf") or "").strip()
        akk_cf = (pair.get("akkadian_cf") or "").strip()
        if not sum_cf or not akk_cf:
            continue
        if sum_cf not in sum_w2i:
            sum_misses += 1
            continue
        sum_vec = sum_vectors[sum_w2i[sum_cf]]
        qn = sum_vec / (np.linalg.norm(sum_vec) + 1e-12)
        sims = eng_normed @ qn
        idx = int(np.argmax(sims))
        cosine = float(sims[idx])
        if cosine < COSINE_THRESHOLD:
            cosine_misses += 1
            continue
        eng_word = eng_vocab[idx].lower()
        akk_norm = normalize_akkadian_token(akk_cf)
        if not akk_norm:
            continue
        key = (akk_norm, eng_word)
        if key in seen:
            continue
        seen.add(key)
        bootstrap_anchors.append({
            "akkadian": akk_norm,
            "english": eng_word,
            "confidence": round(min(0.85, 0.4 + cosine / 2), 4),
            "frequency": 1,
            "source": "DCCLT_bridge",
            "bridge_cosine": round(cosine, 4),
            "bridge_sumerian": sum_cf,
        })

    print(f"\nDCCLT bridge anchors emitted: {len(bootstrap_anchors)}")
    print(f"Sumerian-vocab misses: {sum_misses}")
    print(f"Cosine-threshold misses (< {COSINE_THRESHOLD}): {cosine_misses}")

    with open(ANCHORS_PATH) as f:
        existing = json.load(f)
    print(f"Existing anchors: {len(existing)}")

    best: dict = {}
    for a in existing + bootstrap_anchors:
        key = (a["akkadian"], a["english"])
        if key not in best or a["confidence"] > best[key]["confidence"]:
            best[key] = a
    merged = sorted(best.values(), key=lambda a: a["confidence"], reverse=True)

    with open(ANCHORS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Merged anchor pool: {len(merged)}")
    print(f"  ORACC: {sum(1 for a in merged if a.get('source') == 'ORACC')}")
    print(f"  DCCLT_bridge: {sum(1 for a in merged if a.get('source') == 'DCCLT_bridge')}")


if __name__ == "__main__":
    main()
