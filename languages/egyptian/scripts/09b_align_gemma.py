"""
Ridge alignment of Egyptian FastText into whitened EmbeddingGemma 768d.

Uses PCA target reduction (768d -> 256d) before Ridge to improve the
samples-per-dimension ratio, then lifts predictions back to 768d via
inverse_transform. This yields +1.2pp top-1 over direct 768d Ridge.

Reuses helpers from align_09 for training and evaluation.
"""
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.decomposition import PCA

from languages.egyptian.scripts.align_09 import (
    build_training_data,
    train_ridge,
    evaluate_alignment,
    select_alpha,
)

from shared.scripts.anchor_split import group_split, SEED, TEST_SIZE, VAL_SIZE

_LANG_ROOT = Path(__file__).parent.parent
MODELS_DIR = _LANG_ROOT / "models"
DATA_PROCESSED = _LANG_ROOT / "data" / "processed"
RESULTS_DIR = _LANG_ROOT / "results"

ENGLISH_GEMMA_PATH = _REPO_ROOT / "shared" / "models" / "english_gemma_whitened_768d.npz"
ANCHOR_PATH = DATA_PROCESSED / "english_anchors_normalized.json"
GLOVE_BASELINE_PATH = RESULTS_DIR / "alignment_results.json"

PCA_COMPONENTS = 256
PCA_FIT_SAMPLE = 50000
EXPECTED_TARGET_DIM = 768

SURFACE_KEY = "egyptian_raw"
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]


def main():
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

    train_anchors, val_anchors, test_anchors = group_split(
        anchors, surface_key=SURFACE_KEY
    )
    print(
        f"Group split (seed={SEED}): {len(train_anchors)} train / "
        f"{len(val_anchors)} val / {len(test_anchors)} test raw anchors"
    )

    X_train, Yf_train, train_valid = build_training_data(
        train_anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors
    )
    X_val, Yf_val, val_valid = build_training_data(
        val_anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors
    )
    X_test, Yf_test, test_valid = build_training_data(
        test_anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors
    )
    n_valid = len(train_valid) + len(val_valid) + len(test_valid)
    print(
        f"Valid anchors: {n_valid} / {len(anchors)} — "
        f"{len(train_valid)} train / {len(val_valid)} val / {len(test_valid)} test"
    )
    print(f"Target reduced: {Yf_train.shape[1]}d -> {PCA_COMPONENTS}d")

    print("Selecting alpha on validation (predictions lifted to 768d)...")
    val_english = [a["english"] for a in val_valid]
    best_alpha, sweep = select_alpha(
        X_train, pca.transform(Yf_train), X_val, val_english,
        eng_vocab_list, eng_vectors, ALPHAS,
        predict_transform=pca.inverse_transform,
    )
    print(f"Selected alpha={best_alpha}")

    X_fit = np.concatenate([X_train, X_val], axis=0)
    Y_fit = pca.transform(np.concatenate([Yf_train, Yf_val], axis=0))
    print(f"Retraining on train+val ({len(X_fit)}) at alpha={best_alpha}...")
    model = train_ridge(X_fit, Y_fit, alpha=best_alpha)

    Y_pred = pca.inverse_transform(model.predict(X_test))
    test_english = [a["english"] for a in test_valid]
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
            "alpha": best_alpha,
            "alpha_sweep_val": sweep,
            "pca_components": PCA_COMPONENTS,
            "pca_variance_retained": round(float(variance_kept), 2),
            "train_size": len(X_fit),
            "test_size_count": len(X_test),
            "valid_anchors": n_valid,
            "total_anchors": len(anchors),
            "egyptian_vocab": len(eg_vocab),
            "english_vocab": len(eng_vocab),
            "fused_dim": int(eg_vectors.shape[1]),
            "target_dim": EXPECTED_TARGET_DIM,
            "reduced_dim": PCA_COMPONENTS,
            "gemma_model": gemma_model,
            "gloss_hit_rate": gloss_hit_rate,
            "split": {
                "method": "gloss-group",
                "seed": SEED,
                "val_size": VAL_SIZE,
                "test_size": TEST_SIZE,
                "raw": {"train": len(train_anchors), "val": len(val_anchors),
                        "test": len(test_anchors)},
                "valid": {"train": len(train_valid), "val": len(val_valid),
                          "test": len(test_valid)},
            },
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
