"""
Ridge alpha sweep on the whitened-gloss Gemma target.

One-shot diagnostic: does adjusting ridge regularization get us the
remaining 0.5pp to clear the +3pp phase-A gate, or is 19.85% at
alpha=100 already near the ceiling for this target space?

Reuses helpers from align_09 to keep the comparison identical to the
main 09b run in every way except the alpha value.

See: docs/EXPERIMENT_JOURNAL.md
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.model_selection import train_test_split

from languages.greek.scripts.align_09 import (
    build_training_data,
    train_ridge,
    evaluate_alignment,
)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

FUSED_PATH = MODELS_DIR / "fused_embeddings_1536d.npz"
ENGLISH_GEMMA_PATH = MODELS_DIR / "english_gemma_whitened_768d.npz"
ANCHOR_PATH = DATA_PROCESSED / "english_anchors.json"
RESULTS_OUT_PATH = RESULTS_DIR / "ridge_alpha_sweep.json"
GLOVE_BASELINE_PATH = RESULTS_DIR / "alignment_results.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]


def main():
    if not ENGLISH_GEMMA_PATH.exists():
        print(f"ERROR: whitened Gemma cache not found at {ENGLISH_GEMMA_PATH}", file=sys.stderr)
        print("Run: python scripts/whiten_gemma.py", file=sys.stderr)
        sys.exit(1)

    print("Loading inputs...")
    fused = np.load(str(FUSED_PATH))
    sum_vectors = fused["vectors"]
    sum_vocab_list = [str(w) for w in fused["vocab"]]
    sum_vocab = {w: i for i, w in enumerate(sum_vocab_list)}

    gemma = np.load(str(ENGLISH_GEMMA_PATH))
    eng_vectors = gemma["vectors"]
    eng_vocab_list = [str(w) for w in gemma["vocab"]]
    eng_vocab = {w: i for i, w in enumerate(eng_vocab_list)}

    with open(ANCHOR_PATH) as f:
        anchors = json.load(f)

    # Match production: load FastText model for OOV subword inference (L5)
    from gensim.models import FastText
    ft_path = MODELS_DIR / "fasttext_sumerian.model"
    print(f"Loading FastText model for OOV subword inference: {ft_path}")
    ft_model = FastText.load(str(ft_path))

    X, Y, valid_anchors = build_training_data(
        anchors, sum_vocab, sum_vectors, eng_vocab, eng_vectors, fasttext_model=ft_model
    )

    # Match production L5-refined partition: OOV anchors training-only,
    # test drawn from in-vocab anchors only.
    in_vocab_mask = np.array([not a.get("subword_inferred") for a in valid_anchors])
    X_in = X[in_vocab_mask]
    Y_in = Y[in_vocab_mask]
    anchors_in = [a for a, m in zip(valid_anchors, in_vocab_mask) if m]
    X_oov = X[~in_vocab_mask]
    Y_oov = Y[~in_vocab_mask]
    anchors_oov = [a for a, m in zip(valid_anchors, in_vocab_mask) if not m]

    X_in_train, X_test, Y_in_train, Y_test, anchors_in_train, anchors_test = train_test_split(
        X_in, Y_in, anchors_in, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train = np.concatenate([X_in_train, X_oov], axis=0) if len(X_oov) else X_in_train
    Y_train = np.concatenate([Y_in_train, Y_oov], axis=0) if len(Y_oov) else Y_in_train
    test_english = [a["english"] for a in anchors_test]
    print(
        f"In-vocab: {len(anchors_in)} ({len(anchors_in_train)} train + {len(anchors_test)} test). "
        f"OOV: {len(anchors_oov)} (train-only)."
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    baseline = None
    if GLOVE_BASELINE_PATH.exists():
        with open(GLOVE_BASELINE_PATH) as f:
            baseline = json.load(f).get("accuracy", {})
    baseline_top1 = baseline["top1"] if baseline else None

    print(f"\n=== Ridge alpha sweep (whitened-gloss target) ===")
    print(f"{'alpha':>10s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'delta top1':>11s}")
    print("-" * 56)

    sweep_results = []
    for alpha in ALPHAS:
        model = train_ridge(X_train, Y_train, alpha=alpha)
        Y_pred = model.predict(X_test)
        results = evaluate_alignment(Y_pred, test_english, eng_vocab_list, eng_vectors)
        delta_str = (
            f"{results['top1'] - baseline_top1:+6.2f}pp"
            if baseline_top1 is not None else "      —"
        )
        print(
            f"{alpha:>10.3g}  {results['top1']:6.2f}%  {results['top5']:6.2f}%  "
            f"{results['top10']:6.2f}%  {delta_str:>11s}"
        )
        sweep_results.append({
            "alpha": alpha,
            "accuracy": results,
            "delta_top1_vs_glove": (
                results["top1"] - baseline_top1 if baseline_top1 is not None else None
            ),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT_PATH, "w") as f:
        json.dump({
            "target": "english_gemma_whitened_768d",
            "baseline_glove": baseline,
            "sweep": sweep_results,
        }, f, indent=2)
    print(f"\nSweep saved to: {RESULTS_OUT_PATH}")


if __name__ == "__main__":
    main()
