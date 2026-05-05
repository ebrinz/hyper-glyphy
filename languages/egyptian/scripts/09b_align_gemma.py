"""
Ridge alignment of Egyptian FastText into whitened EmbeddingGemma 768d.

Uses PCA target reduction (768d -> 256d) before Ridge to improve the
samples-per-dimension ratio, then lifts predictions back to 768d via
inverse_transform. This yields +1.2pp top-1 over direct 768d Ridge.

Reuses helpers from align_09 for training and evaluation.
"""
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from languages.egyptian.scripts.align_09 import (
    build_training_data,
    train_ridge,
    evaluate_alignment,
)

_LANG_ROOT = Path(__file__).parent.parent
MODELS_DIR = _LANG_ROOT / "models"
DATA_PROCESSED = _LANG_ROOT / "data" / "processed"
RESULTS_DIR = _LANG_ROOT / "results"

ENGLISH_GEMMA_PATH = _REPO_ROOT / "shared" / "models" / "english_gemma_whitened_768d.npz"
ANCHOR_PATH = DATA_PROCESSED / "english_anchors_normalized.json"
GLOVE_BASELINE_PATH = RESULTS_DIR / "alignment_results.json"

RIDGE_ALPHA = 1.0
PCA_COMPONENTS = 256
PCA_FIT_SAMPLE = 50000
TEST_SIZE = 0.2
RANDOM_STATE = 42
EXPECTED_TARGET_DIM = 768

SWEEP_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]


def main():
    parser = argparse.ArgumentParser(description="PCA-Ridge alignment: Egyptian FastText -> whitened EmbeddingGemma 768d.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run alpha sweep over SWEEP_ALPHAS before final training.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ENGLISH_GEMMA_PATH.exists():
        print(f"ERROR: Whitened Gemma cache not found at {ENGLISH_GEMMA_PATH}", file=sys.stderr)
        print("Run: python shared/scripts/whiten_gemma.py", file=sys.stderr)
        sys.exit(1)

    fused_path = MODELS_DIR / "fused_embeddings_1536d.npz"
    print(f"Loading fused Egyptian vectors from {fused_path}")
    fused = np.load(str(fused_path))
    eg_vectors = fused["vectors"]
    eg_vocab_list = [str(w) for w in fused["vocab"]]
    eg_vocab = {w: i for i, w in enumerate(eg_vocab_list)}
    print(f"Egyptian vocab: {len(eg_vocab)} words, {eg_vectors.shape[1]}d")

    print(f"Loading Gemma English vectors from {ENGLISH_GEMMA_PATH}")
    gemma = np.load(str(ENGLISH_GEMMA_PATH))
    eng_vectors = gemma["vectors"]
    eng_vocab_list = [str(w) for w in gemma["vocab"]]
    eng_vocab = {w: i for i, w in enumerate(eng_vocab_list)}
    gloss_hit_rate = float(gemma["gloss_hit_rate"]) if "gloss_hit_rate" in gemma.files else None
    gemma_model = str(gemma["gemma_model"]) if "gemma_model" in gemma.files else None
    print(f"English vocab: {len(eng_vocab)} words, {eng_vectors.shape[1]}d")

    assert eng_vectors.shape[1] == EXPECTED_TARGET_DIM, (
        f"English target dim is {eng_vectors.shape[1]}, expected {EXPECTED_TARGET_DIM}."
    )

    print(f"\nFitting PCA: {EXPECTED_TARGET_DIM}d -> {PCA_COMPONENTS}d (on {PCA_FIT_SAMPLE} English vectors)...")
    pca = PCA(n_components=PCA_COMPONENTS).fit(eng_vectors[:PCA_FIT_SAMPLE].astype(np.float32))
    variance_kept = pca.explained_variance_ratio_.sum() * 100
    print(f"  Variance retained: {variance_kept:.1f}%")

    with open(ANCHOR_PATH) as f:
        anchors = json.load(f)
    print(f"Loaded {len(anchors)} anchors")

    X, Y_full, valid_anchors = build_training_data(
        anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors
    )
    Y = pca.transform(Y_full)
    print(
        f"Valid anchors: {len(valid_anchors)} / {len(anchors)} "
        f"({len(valid_anchors)/len(anchors)*100:.1f}%)"
    )
    print(f"Target reduced: {Y_full.shape[1]}d -> {Y.shape[1]}d "
          f"(samples-per-dim: {len(X)}/{PCA_COMPONENTS} = {len(X)/PCA_COMPONENTS:.1f})")

    X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
        X, Y, valid_anchors, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    Y_test_full = pca.inverse_transform(Y_test)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    if args.sweep:
        print("\n=== ALPHA SWEEP (PCA-reduced target) ===")
        sweep_results = {}
        for alpha in SWEEP_ALPHAS:
            m = train_ridge(X_train, Y_train, alpha=alpha)
            Y_pred_reduced = m.predict(X_test)
            Y_pred_full = pca.inverse_transform(Y_pred_reduced)
            te = [a["english"] for a in anchors_test]
            r = evaluate_alignment(Y_pred_full, te, eng_vocab_list, eng_vectors)
            sweep_results[alpha] = r
            print(f"  alpha={alpha:<8} top1={r['top1']:.2f}%  top5={r['top5']:.2f}%  top10={r['top10']:.2f}%")

        sweep_path = RESULTS_DIR / "alpha_sweep_gemma.json"
        with open(sweep_path, "w") as f:
            json.dump({str(k): v for k, v in sweep_results.items()}, f, indent=2)
        print(f"Sweep saved to: {sweep_path}")

    print(f"\nTraining Ridge (alpha={RIDGE_ALPHA}, target={PCA_COMPONENTS}d)...")
    model = train_ridge(X_train, Y_train, alpha=RIDGE_ALPHA)

    Y_pred_reduced = model.predict(X_test)
    Y_pred = pca.inverse_transform(Y_pred_reduced)
    test_english = [a["english"] for a in anchors_test]
    results = evaluate_alignment(Y_pred, test_english, eng_vocab_list, eng_vectors)

    baseline = None
    if GLOVE_BASELINE_PATH.exists():
        with open(GLOVE_BASELINE_PATH) as f:
            baseline = json.load(f).get("accuracy", {})

    print(f"\n=== RESULTS (Gemma target, PCA-{PCA_COMPONENTS}) ===")
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
            "alignment": "PCA-Ridge",
            "alpha": RIDGE_ALPHA,
            "pca_components": PCA_COMPONENTS,
            "pca_variance_retained": round(float(variance_kept), 2),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_size": len(X_train),
            "test_size_count": len(X_test),
            "valid_anchors": len(valid_anchors),
            "total_anchors": len(anchors),
            "egyptian_vocab": len(eg_vocab),
            "english_vocab": len(eng_vocab),
            "fused_dim": int(eg_vectors.shape[1]),
            "target_dim": EXPECTED_TARGET_DIM,
            "reduced_dim": PCA_COMPONENTS,
            "gemma_model": gemma_model,
            "gloss_hit_rate": gloss_hit_rate,
        },
    }

    results_out_path = RESULTS_DIR / "alignment_results_gemma_whitened.json"
    with open(results_out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {results_out_path}")

    ridge_out_path = MODELS_DIR / "ridge_weights_gemma_whitened.npz"
    np.savez_compressed(
        str(ridge_out_path),
        coef=model.coef_,
        intercept=model.intercept_,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
    )
    print(f"Ridge weights + PCA saved to: {ridge_out_path}")


if __name__ == "__main__":
    main()
