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
from tqdm import tqdm  # noqa: F401 — canonical template; select_alpha uses it via align_09

# Ensure repo root is importable when invoked directly (pytest.ini only affects pytest).
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.decomposition import PCA

from shared.scripts.anchor_split import group_split, SEED, TEST_SIZE, VAL_SIZE
from shared.scripts.eval_suite import (
    CAND_SIZE,
    save_artifacts,
    score_suite,
    stratify,
    val_top1_csls,  # noqa: F401 — used inside imported select_alpha
)
from languages.egyptian.scripts.align_09 import (
    build_training_data,
    train_ridge,
    evaluate_alignment,
    select_alpha,
    filter_stopword_glosses,
)

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

    anchors, n_stopword_dropped = filter_stopword_glosses(anchors)
    print(f"Stopword-gloss filter: dropped {n_stopword_dropped}, kept {len(anchors)}")

    train_anchors, val_anchors, test_anchors = group_split(
        anchors, surface_key=SURFACE_KEY, fallback="surface_casefold"
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

    # Artifact bundle: predictions lifted to full 768d via pca.inverse_transform.
    rng = np.random.RandomState(SEED)
    non_oov_train = list(enumerate(train_valid))
    sample_idx = rng.choice(
        len(non_oov_train), size=min(1000, len(non_oov_train)), replace=False
    )
    train_sample = [non_oov_train[i] for i in sample_idx]
    Q_train = pca.inverse_transform(model.predict(X_train[[i for i, _ in train_sample]]))
    Q_val = pca.inverse_transform(model.predict(X_val))
    Q_test = pca.inverse_transform(model.predict(X_test))

    trainval_golds = {a["english"] for a in train_valid} | {
        a["english"] for a in val_valid
    }
    test_strata = stratify([a["english"] for a in test_valid], trainval_golds)

    config = {
        "target": "gemma",
        "target_cache": str(ENGLISH_GEMMA_PATH),
        "alpha": best_alpha,
        "alpha_selection": "val_top5_csls_v2",
        "alpha_sweep_val": sweep,
        "seed": SEED,
        "candidate_vocab_size": CAND_SIZE,
        "split": {
            "method": "surface-casefold-group",
            "seed": SEED,
            "val_size": VAL_SIZE,
            "test_size": TEST_SIZE,
            "near_surface_edges": True,
            "raw": {"train": len(train_anchors), "val": len(val_anchors),
                    "test": len(test_anchors)},
            "valid": {"train": len(train_valid), "val": len(val_valid),
                      "test": len(test_valid)},
        },
        "stopword_glosses_dropped": n_stopword_dropped,
    }
    prefix = str(RESULTS_DIR / "eval_artifacts_gemma")
    save_artifacts(
        prefix,
        coef=model.coef_, intercept=model.intercept_,
        Q_train=Q_train, Q_val=Q_val, Q_test=Q_test,
        train_sample=[{"surface": a[SURFACE_KEY], "gold": a["english"]}
                      for _, a in train_sample],
        val=[{"surface": a[SURFACE_KEY], "gold": a["english"]} for a in val_valid],
        test=[{"surface": a[SURFACE_KEY], "gold": a["english"]} for a in test_valid],
        test_strata=test_strata,
        config=config,
    )
    print(f"Artifacts saved to {prefix}.npz/.json")

    cand_vectors = eng_vectors[:CAND_SIZE]
    cand_vocab = eng_vocab_list[:CAND_SIZE]
    from shared.scripts.eval_suite import load_artifacts

    suite = score_suite(load_artifacts(prefix), cand_vectors, cand_vocab)

    print("\n=== METRIC SUITE (CSLS, 50k candidates, exact/syn) ===")
    for regime in ("dictionary_in_sample", "interpolation", "zero_shot", "test_combined"):
        r = suite[regime]
        print(f"{regime:<22} n={r['n']:>6} "
              + "  ".join(f"top{k}={r[f'top{k}']['exact']:.2f}/{r[f'top{k}']['syn']:.2f}"
                          for k in (1, 5, 10)))

    combined = suite["test_combined"]
    full_results = {
        # Legacy key: combined-strata test CSLS/restricted EXACT top-k, so
        # 10_export_production.py keeps reading the same shape.
        "accuracy": {f"top{k}": combined[f"top{k}"]["exact"] for k in (1, 5, 10)},
        "metric_suite": suite,
        "config": config | {
            "train_size": len(X_fit),
            "test_size_count": len(X_test),
            "valid_anchors": n_valid,
            "total_anchors": len(anchors),
            "egyptian_vocab": len(eg_vocab),
            "english_vocab": len(eng_vocab),
            "fused_dim": int(eg_vectors.shape[1]),
            "target_dim": EXPECTED_TARGET_DIM,
            "reduced_dim": PCA_COMPONENTS,
            "pca_components": PCA_COMPONENTS,
            "pca_variance_retained": round(float(variance_kept), 2),
            "gemma_model": gemma_model,
            "gloss_hit_rate": gloss_hit_rate,
        },
    }

    baseline = None
    if GLOVE_BASELINE_PATH.exists():
        with open(GLOVE_BASELINE_PATH) as f:
            baseline = json.load(f).get("accuracy", {})

    full_results["baseline_glove"] = baseline
    full_results["deltas_vs_glove"] = (
        {k: full_results["accuracy"][k] - baseline[k]
         for k in full_results["accuracy"] if k in baseline}
        if baseline
        else None
    )

    print(f"\n=== RESULTS (Gemma target, PCA-{PCA_COMPONENTS}) ===")
    for k_str in ("top1", "top5", "top10"):
        gemma_val = full_results["accuracy"][k_str]
        if baseline and k_str in baseline:
            delta = gemma_val - baseline[k_str]
            print(
                f"{k_str.upper():<6} Gemma {gemma_val:6.2f}%  "
                f"GloVe {baseline[k_str]:6.2f}%  "
                f"delta {delta:+.2f}pp"
            )
        else:
            print(f"{k_str.upper():<6} Gemma {gemma_val:6.2f}%")

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
