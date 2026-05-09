"""
Phase A: Ridge alignment of Sumerian FastText into EmbeddingGemma 768d.

Mirrors 09_align_and_evaluate.py but targets EmbeddingGemma-encoded
English vectors instead of GloVe. Reuses helpers from align_09 to
keep the comparison apples-to-apples.

See: docs/superpowers/specs/2026-04-16-gemma-embed-alignment-design.md
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is importable when invoked directly (pytest.ini only affects pytest).
_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.model_selection import train_test_split

from languages.sumerian.scripts.align_09 import (
    build_training_data,
    train_ridge,
    evaluate_alignment,
)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

FUSED_PATH = MODELS_DIR / "fused_embeddings_1536d.npz"
ENGLISH_GEMMA_PATHS = {
    "gloss": MODELS_DIR / "english_gemma_768d.npz",
    "bare": MODELS_DIR / "english_gemma_bare_768d.npz",
    "whitened": MODELS_DIR / "english_gemma_whitened_768d.npz",
    "bare_whitened": MODELS_DIR / "english_gemma_bare_whitened_768d.npz",
}
RESULTS_SUFFIXES = {
    "gloss": "",
    "bare": "_bare",
    "whitened": "_whitened",
    "bare_whitened": "_bare_whitened",
}
ANCHOR_PATH = DATA_PROCESSED / "english_anchors.json"
GLOVE_BASELINE_PATH = RESULTS_DIR / "alignment_results.json"

RIDGE_ALPHA = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42
EXPECTED_TARGET_DIM = 768


def main():
    parser = argparse.ArgumentParser(description="Ridge alignment: Sumerian FastText -> EmbeddingGemma 768d.")
    parser.add_argument(
        "--mode",
        choices=list(ENGLISH_GEMMA_PATHS.keys()),
        default="gloss",
        help="Which English Gemma cache to use as the target.",
    )
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Shortcut for --mode bare (kept for backwards compatibility).",
    )
    args = parser.parse_args()
    mode_label = "bare" if args.bare else args.mode
    english_gemma_path = ENGLISH_GEMMA_PATHS[mode_label]
    suffix = RESULTS_SUFFIXES[mode_label]
    ridge_out_path = MODELS_DIR / f"ridge_weights_gemma{suffix}.npz"
    results_out_path = RESULTS_DIR / f"alignment_results_gemma{suffix}.json"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not english_gemma_path.exists():
        print(f"ERROR: English Gemma {mode_label} cache not found at {english_gemma_path}", file=sys.stderr)
        if mode_label in ("gloss", "bare"):
            hint = " --bare" if mode_label == "bare" else ""
            print(f"Run: python scripts/embed_english_gemma.py{hint}", file=sys.stderr)
        elif mode_label == "whitened":
            print("Run: python scripts/whiten_gemma.py", file=sys.stderr)
        elif mode_label == "bare_whitened":
            print("Run: python scripts/whiten_gemma.py --source bare", file=sys.stderr)
        sys.exit(1)

    print(f"Mode: {mode_label}")
    print(f"Loading fused Sumerian vectors from {FUSED_PATH}")
    fused = np.load(str(FUSED_PATH))
    sum_vectors = fused["vectors"]
    sum_vocab_list = [str(w) for w in fused["vocab"]]
    sum_vocab = {w: i for i, w in enumerate(sum_vocab_list)}
    print(f"Sumerian vocab: {len(sum_vocab)} words, {sum_vectors.shape[1]}d")

    print(f"Loading Gemma English vectors from {english_gemma_path}")
    gemma = np.load(str(english_gemma_path))
    eng_vectors = gemma["vectors"]
    eng_vocab_list = [str(w) for w in gemma["vocab"]]
    eng_vocab = {w: i for i, w in enumerate(eng_vocab_list)}
    gloss_hit_rate = float(gemma["gloss_hit_rate"]) if "gloss_hit_rate" in gemma.files else None
    gemma_model = str(gemma["gemma_model"]) if "gemma_model" in gemma.files else None
    print(f"English vocab: {len(eng_vocab)} words, {eng_vectors.shape[1]}d")

    assert eng_vectors.shape[1] == EXPECTED_TARGET_DIM, (
        f"English target dim is {eng_vectors.shape[1]}, expected {EXPECTED_TARGET_DIM}. "
        "Regenerate the Gemma cache with scripts/embed_english_gemma.py."
    )

    with open(ANCHOR_PATH) as f:
        anchors = json.load(f)
    print(f"Loaded {len(anchors)} anchors")

    X, Y, valid_anchors = build_training_data(
        anchors, sum_vocab, sum_vectors, eng_vocab, eng_vectors
    )
    print(
        f"Valid anchors: {len(valid_anchors)} / {len(anchors)} "
        f"({len(valid_anchors)/len(anchors)*100:.1f}%)"
    )

    X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
        X, Y, valid_anchors, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print(f"Training Ridge (alpha={RIDGE_ALPHA})...")
    model = train_ridge(X_train, Y_train, alpha=RIDGE_ALPHA)

    Y_pred = model.predict(X_test)
    test_english = [a["english"] for a in anchors_test]
    results = evaluate_alignment(Y_pred, test_english, eng_vocab_list, eng_vectors)

    baseline = None
    if GLOVE_BASELINE_PATH.exists():
        with open(GLOVE_BASELINE_PATH) as f:
            baseline = json.load(f).get("accuracy", {})

    print(f"\n=== RESULTS (Gemma target) ===")
    for k_str in ("top1", "top5", "top10"):
        gemma_val = results[k_str]
        if baseline and k_str in baseline:
            delta = gemma_val - baseline[k_str]
            print(
                f"{k_str.upper():<6} Gemma {gemma_val:6.2f}%  "
                f"GloVe {baseline[k_str]:6.2f}%  "
                f"delta {delta:+.2f}pp"
            )
        else:
            print(f"{k_str.upper():<6} Gemma {gemma_val:6.2f}%")

    full_results = {
        "accuracy": results,
        "baseline_glove": baseline,
        "deltas_vs_glove": (
            {k: results[k] - baseline[k] for k in results if k in baseline}
            if baseline
            else None
        ),
        "config": {
            "alignment": "Ridge",
            "alpha": RIDGE_ALPHA,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_size": len(X_train),
            "test_size_count": len(X_test),
            "valid_anchors": len(valid_anchors),
            "total_anchors": len(anchors),
            "sumerian_vocab": len(sum_vocab),
            "english_vocab": len(eng_vocab),
            "fused_dim": int(sum_vectors.shape[1]),
            "target_dim": int(eng_vectors.shape[1]),
            "gemma_model": gemma_model,
            "gloss_hit_rate": gloss_hit_rate,
            "mode": mode_label,
        },
    }

    with open(results_out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {results_out_path}")

    np.savez_compressed(
        str(ridge_out_path),
        coef=model.coef_,
        intercept=model.intercept_,
    )
    print(f"Ridge weights saved to: {ridge_out_path}")


if __name__ == "__main__":
    main()
