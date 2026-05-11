"""
Hittite Anchor Extraction: German-glossed TLHdig lemmas + heterogram-bridge.

Two anchor sources:

1. **Primary (German via Gemma)**: TLHdig glosses are German. Multilingual
   EmbeddingGemma encodes both German and English into the same 768d space.
   For each unique German gloss, encode with Gemma and find the nearest
   English word in the existing english_gemma_768d cache. Use that English
   word as the anchor's `english` field — the rest of the Ridge pipeline
   then works unchanged.

2. **Bridge (heterograms via Sumerian/Akkadian)**: Hittite tablets contain
   embedded Sumerograms (`<sGr>`) and Akkadograms (`<aGr>`). For each
   heterogram type that occurs in our corpus, look it up in Sumerian's (or
   Akkadian's) aligned-Gemma space and use the top-1 English neighbor as
   an anchor target. Filtered by cosine threshold.

Both sources merge into english_anchors.json with the same schema as the
Akkadian/Sumerian slots: {hittite, english, confidence, frequency, source}.

Prerequisite: Sumerian vocab cached as JSON at /tmp/sumerian_vocab_for_bridge.json
(created during Akkadian L6b; if missing, run the bridge fetcher there first
to regenerate).
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

from languages.hittite.scripts.hittite_normalize import normalize_hittite_token  # noqa: E402

HIT_DIR = Path(__file__).parent.parent
DATA_RAW = HIT_DIR / "data" / "raw"
DATA_PROCESSED = HIT_DIR / "data" / "processed"
DATA_DICTS = HIT_DIR / "data" / "dictionaries"

SHARED_MODELS = _ROOT / "shared" / "models"
ENG_GEMMA_PATH = SHARED_MODELS / "english_gemma_768d.npz"  # raw, gloss-aware

GEMMA_MODEL = "google/embeddinggemma-300m"
TRANSLATION_CACHE = DATA_DICTS / "german_to_english.json"

# Thresholds
MIN_GLOSS_OCCURRENCES = 5
JUNK_GLOSSES = {"", "?", "x", "X", "...", "??"}

# Heterogram bridge config
SUM_ALIGNED_PATH = _ROOT / "languages" / "sumerian" / "final_output" / "sumerian_aligned_gemma_vectors.npz"
SUM_VOCAB_JSON = Path("/tmp/sumerian_vocab_for_bridge.json")
AKK_ALIGNED_PATH = _ROOT / "languages" / "akkadian" / "final_output" / "akkadian_aligned_gemma_vectors.npz"
AKK_VOCAB_JSON = _ROOT / "languages" / "akkadian" / "final_output" / "akkadian_aligned_vocab.json"

HETERO_COSINE_THRESHOLD = 0.5


def _load_english_gemma():
    data = np.load(str(ENG_GEMMA_PATH))
    vectors = data["vectors"].astype(np.float32)
    vocab = [str(w) for w in data["vocab"]]
    return vectors, vocab


def _load_english_wordset() -> set[str]:
    """Load NLTK words corpus into a lowercase set for English filtering.

    The english_gemma cache was built from GloVe 400k vocab which contains many
    non-English tokens (German, Dutch, etc.). For German -> English translation
    we restrict nearest-neighbor lookup to entries in NLTK's English wordlist —
    cleaner than BSD's /usr/share/dict/words (correctly excludes 'gott',
    'tempel', 'könig', etc.).
    """
    try:
        import nltk
        nltk.download("words", quiet=True)
        from nltk.corpus import words as nltk_words
        return {w.lower() for w in nltk_words.words()}
    except Exception as e:
        print(f"  WARNING: NLTK words unavailable ({e}); translation will be promiscuous")
        return set()


def _build_english_mask(eng_vocab: list[str], english_set: set[str]) -> np.ndarray:
    """Return a 1D bool array marking which english_gemma entries are real English."""
    if not english_set:
        return np.ones(len(eng_vocab), dtype=bool)
    return np.array([w.lower() in english_set for w in eng_vocab], dtype=bool)


def translate_german_glosses(
    german_glosses: list[str],
    eng_vectors: np.ndarray,
    eng_vocab: list[str],
    batch_size: int = 64,
) -> dict[str, str]:
    """Encode each German gloss with Gemma; return its nearest English word.

    Cached to disk on first run.
    """
    if TRANSLATION_CACHE.exists():
        with open(TRANSLATION_CACHE, encoding="utf-8") as f:
            cached = json.load(f)
        if set(german_glosses) <= set(cached.keys()):
            print(f"  Using cached translations from {TRANSLATION_CACHE}")
            return {g: cached[g] for g in german_glosses}

    from sentence_transformers import SentenceTransformer

    print(f"  Encoding {len(german_glosses)} German glosses with {GEMMA_MODEL}...")
    model = SentenceTransformer(GEMMA_MODEL)
    eng_norms = np.linalg.norm(eng_vectors, axis=1, keepdims=True) + 1e-12
    eng_normed = eng_vectors / eng_norms

    # Restrict candidate English words to entries in a real English wordlist
    english_set = _load_english_wordset()
    mask = _build_english_mask(eng_vocab, english_set)
    print(f"  English wordlist filter: {mask.sum()} / {len(eng_vocab)} vocab entries match")

    translations: dict[str, str] = {}
    for i in range(0, len(german_glosses), batch_size):
        batch = german_glosses[i : i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        emb_normed = embeddings / emb_norms
        sims = emb_normed @ eng_normed.T
        # Force non-English candidates to -inf so argmax skips them
        sims[:, ~mask] = -np.inf
        top_idx = np.argmax(sims, axis=1)
        for g, idx in zip(batch, top_idx):
            translations[g] = eng_vocab[idx]

    DATA_DICTS.mkdir(parents=True, exist_ok=True)
    with open(TRANSLATION_CACHE, "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)
    print(f"  Cached {len(translations)} translations to {TRANSLATION_CACHE}")
    return translations


def _filter_gloss(gw: str) -> bool:
    """Reject junk, numeric, and parenthesized-only glosses."""
    if not gw or gw in JUNK_GLOSSES:
        return False
    if len(gw) <= 2:
        return False
    if gw[0].isdigit() or gw.isdigit():
        return False
    # Reject glosses that are just parentheses noise: "(...)", "(?)" etc.
    if gw.startswith("(") and gw.endswith(")"):
        return False
    # Reject Hittite proper-name-looking glosses (titlecase + non-ASCII initial
    # marks like Š/Ḫ — these are typically scribal names mistakenly captured
    # as German glosses). Heuristic: starts with uppercase non-ASCII letter.
    if gw[0] in "ŠḪṢṬĀĒĪŪŌ":
        return False
    return True


def extract_german_anchors(
    lemmas: list[dict],
    eng_vectors: np.ndarray,
    eng_vocab: list[str],
    min_occurrences: int = MIN_GLOSS_OCCURRENCES,
) -> list[dict]:
    """Extract Hittite-English anchors via German glosses + Gemma translation."""
    gloss_counts: Counter[str] = Counter()
    for lemma in lemmas:
        gw = (lemma.get("gw") or "").strip()
        if not _filter_gloss(gw):
            continue
        gloss_counts[gw] += 1
    surviving_glosses = sorted(g for g, c in gloss_counts.items() if c >= min_occurrences)
    print(f"  Glosses meeting >= {min_occurrences} threshold: {len(surviving_glosses)}")

    translations = translate_german_glosses(surviving_glosses, eng_vectors, eng_vocab)

    pair_counts: Counter[tuple[str, str]] = Counter()
    for lemma in lemmas:
        gw = (lemma.get("gw") or "").strip()
        if gw not in translations:
            continue
        english = translations[gw].lower()
        if not english or len(english) < 2:
            continue
        cf = normalize_hittite_token((lemma.get("cf") or "").strip())
        form = normalize_hittite_token((lemma.get("form") or "").strip())
        surfaces: set[str] = set()
        if cf:
            surfaces.add(cf)
        if form and form != cf:
            surfaces.add(form)
        for surface in surfaces:
            pair_counts[(surface, english)] += 1

    anchors: list[dict] = []
    for (form_norm, eng), count in pair_counts.items():
        if count < min_occurrences:
            continue
        confidence = min(0.95, 0.5 + (count / 100))
        anchors.append({
            "hittite": form_norm,
            "english": eng,
            "confidence": round(confidence, 4),
            "frequency": count,
            "source": "TLHdig_de",
        })
    return sorted(anchors, key=lambda a: a["confidence"], reverse=True)


def _topk_english(query_vec, eng_normed, eng_vocab):
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    sims = eng_normed @ qn
    idx = int(np.argmax(sims))
    return eng_vocab[idx], float(sims[idx])


def extract_heterogram_anchors(
    heterograms: dict,
    eng_vectors: np.ndarray,
    eng_vocab: list[str],
    cosine_threshold: float = HETERO_COSINE_THRESHOLD,
    min_occurrences: int = MIN_GLOSS_OCCURRENCES,
) -> list[dict]:
    """Bridge Sumerograms/Akkadograms in Hittite text via existing aligned spaces."""
    anchors: list[dict] = []

    eng_norms = np.linalg.norm(eng_vectors, axis=1, keepdims=True) + 1e-12
    eng_normed = eng_vectors / eng_norms

    if SUM_ALIGNED_PATH.exists() and SUM_VOCAB_JSON.exists():
        sum_vectors = np.load(str(SUM_ALIGNED_PATH))["vectors"].astype(np.float32)
        with open(SUM_VOCAB_JSON, encoding="utf-8") as f:
            sum_vocab = json.load(f)
        sum_w2i = {w: i for i, w in enumerate(sum_vocab)}
        for sumerogram, count in heterograms.get("sumerograms", {}).items():
            if count < min_occurrences:
                continue
            for key in (sumerogram, sumerogram.lower()):
                if key in sum_w2i:
                    eng, cosine = _topk_english(sum_vectors[sum_w2i[key]], eng_normed, eng_vocab)
                    if cosine >= cosine_threshold:
                        anchors.append({
                            "hittite": normalize_hittite_token(sumerogram),
                            "english": eng.lower(),
                            "confidence": round(min(0.85, 0.4 + cosine / 2), 4),
                            "frequency": count,
                            "source": "heterogram_sux",
                            "bridge_cosine": round(cosine, 4),
                            "bridge_source_word": key,
                        })
                    break

    if AKK_ALIGNED_PATH.exists() and AKK_VOCAB_JSON.exists():
        akk_vectors = np.load(str(AKK_ALIGNED_PATH))["vectors"].astype(np.float32)
        with open(AKK_VOCAB_JSON, encoding="utf-8") as f:
            akk_vocab = json.load(f)
        akk_w2i = {w: i for i, w in enumerate(akk_vocab)}
        for akkadogram, count in heterograms.get("akkadograms", {}).items():
            if count < min_occurrences:
                continue
            akk_norm = normalize_hittite_token(akkadogram)
            if akk_norm in akk_w2i:
                eng, cosine = _topk_english(akk_vectors[akk_w2i[akk_norm]], eng_normed, eng_vocab)
                if cosine >= cosine_threshold:
                    anchors.append({
                        "hittite": akk_norm,
                        "english": eng.lower(),
                        "confidence": round(min(0.85, 0.4 + cosine / 2), 4),
                        "frequency": count,
                        "source": "heterogram_akk",
                        "bridge_cosine": round(cosine, 4),
                        "bridge_source_word": akkadogram,
                    })

    return anchors


def merge_anchors(primary: list[dict], heterogram: list[dict]) -> list[dict]:
    best: dict[tuple[str, str], dict] = {}
    for a in primary + heterogram:
        key = (a["hittite"], a["english"])
        if key not in best or a["confidence"] > best[key]["confidence"]:
            best[key] = a
    return sorted(best.values(), key=lambda a: a["confidence"], reverse=True)


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_DICTS.mkdir(parents=True, exist_ok=True)

    with open(DATA_RAW / "hittite_lemmas.json") as f:
        lemmas = json.load(f)
    print(f"Loaded {len(lemmas)} TLHdig lemma records")

    with open(DATA_RAW / "hittite_heterograms.json") as f:
        heterograms = json.load(f)
    print(
        f"Heterograms: {len(heterograms.get('sumerograms', {}))} Sumerograms, "
        f"{len(heterograms.get('akkadograms', {}))} Akkadograms"
    )

    eng_vectors, eng_vocab = _load_english_gemma()
    print(f"English Gemma cache: {len(eng_vocab)} words, {eng_vectors.shape[1]}d")

    print("\n[Primary] German-gloss anchors via multilingual Gemma...")
    primary = extract_german_anchors(lemmas, eng_vectors, eng_vocab)
    print(f"  Primary anchors: {len(primary)}")

    print("\n[Bridge] Heterogram anchors via existing aligned spaces...")
    bridge = extract_heterogram_anchors(heterograms, eng_vectors, eng_vocab)
    bridge_sux = sum(1 for a in bridge if a["source"] == "heterogram_sux")
    bridge_akk = sum(1 for a in bridge if a["source"] == "heterogram_akk")
    print(f"  Bridge anchors: {len(bridge)} (Sumerogram: {bridge_sux}, Akkadogram: {bridge_akk})")

    merged = merge_anchors(primary, bridge)
    output_path = DATA_PROCESSED / "english_anchors.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nTotal merged anchors: {len(merged)}")
    print(f"  TLHdig_de: {sum(1 for a in merged if a['source'] == 'TLHdig_de')}")
    print(f"  heterogram_sux: {sum(1 for a in merged if a['source'] == 'heterogram_sux')}")
    print(f"  heterogram_akk: {sum(1 for a in merged if a['source'] == 'heterogram_akk')}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
