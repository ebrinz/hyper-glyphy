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
from tqdm import tqdm  # noqa: F401 — canonical template; select_alpha uses it via align_09

# Ensure repo root is importable when invoked directly (pytest.ini only affects pytest).
_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from shared.scripts.anchor_split import group_split, SEED, TEST_SIZE, VAL_SIZE
from shared.scripts.eval_suite import (
    CAND_SIZE,
    save_artifacts,
    score_suite,
    stratify,
    val_top1_csls,  # noqa: F401 — used inside imported select_alpha
)
from languages.sanskrit.scripts.align_09 import (
    build_training_data,
    train_ridge,
    evaluate_alignment,
    select_alpha,
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

SURFACE_KEY = "sanskrit"
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
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

    # L5: load FastText model for OOV subword inference
    from gensim.models import FastText
    ft_path = MODELS_DIR / "fasttext_sanskrit.model"
    print(f"Loading FastText model from {ft_path} (for OOV subword inference)")
    ft_model = FastText.load(str(ft_path))

    train_anchors, val_anchors, test_anchors = group_split(
        anchors, surface_key=SURFACE_KEY
    )
    print(
        f"Group split (seed={SEED}): {len(train_anchors)} train / "
        f"{len(val_anchors)} val / {len(test_anchors)} test raw anchors"
    )

    # OOV subword inference is training-only: val/test are built WITHOUT the
    # FastText fallback so they stay in-vocab.
    X_train, Y_train, train_valid = build_training_data(
        train_anchors, sum_vocab, sum_vectors, eng_vocab, eng_vectors,
        fasttext_model=ft_model,
    )
    X_val, Y_val, val_valid = build_training_data(
        val_anchors, sum_vocab, sum_vectors, eng_vocab, eng_vectors
    )
    X_test, Y_test, test_valid = build_training_data(
        test_anchors, sum_vocab, sum_vectors, eng_vocab, eng_vectors
    )
    n_oov_train = sum(1 for a in train_valid if a.get("subword_inferred"))
    n_valid = len(train_valid) + len(val_valid) + len(test_valid)
    print(
        f"Valid anchors: {n_valid} / {len(anchors)} — "
        f"{len(train_valid)} train ({n_oov_train} OOV-inferred) / "
        f"{len(val_valid)} val / {len(test_valid)} test"
    )

    print("Selecting alpha on validation...")
    val_english = [a["english"] for a in val_valid]
    best_alpha, sweep = select_alpha(
        X_train, Y_train, X_val, val_english, eng_vocab_list, eng_vectors, ALPHAS
    )
    print(f"Selected alpha={best_alpha}")

    X_fit = np.concatenate([X_train, X_val], axis=0)
    Y_fit = np.concatenate([Y_train, Y_val], axis=0)
    print(f"Retraining on train+val ({len(X_fit)}) at alpha={best_alpha}...")
    model = train_ridge(X_fit, Y_fit, alpha=best_alpha)

    # Artifact bundle: predictions in full target space + strata metadata.
    rng = np.random.RandomState(SEED)
    non_oov_train = [
        (i, a) for i, a in enumerate(train_valid) if not a.get("subword_inferred")
    ]
    sample_idx = rng.choice(
        len(non_oov_train), size=min(1000, len(non_oov_train)), replace=False
    )
    train_sample = [non_oov_train[i] for i in sample_idx]
    Q_train = model.predict(X_train[[i for i, _ in train_sample]])
    Q_val = model.predict(X_val)
    Q_test = model.predict(X_test)

    trainval_golds = {a["english"] for a in train_valid} | {
        a["english"] for a in val_valid
    }
    test_strata = stratify([a["english"] for a in test_valid], trainval_golds)

    config = {
        "target": "gemma",
        "target_cache": str(english_gemma_path),
        "alpha": best_alpha,
        "alpha_selection": "val_top5_csls_v2",
        "alpha_sweep_val": sweep,
        "seed": SEED,
        "candidate_vocab_size": CAND_SIZE,
        "split": {
            "method": "lemma-group",
            "seed": SEED,
            "val_size": VAL_SIZE,
            "test_size": TEST_SIZE,
            "near_surface_edges": True,
            "raw": {"train": len(train_anchors), "val": len(val_anchors),
                    "test": len(test_anchors)},
            "valid": {"train": len(train_valid), "val": len(val_valid),
                      "test": len(test_valid)},
            "oov_train_only": n_oov_train,
        },
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
            "test_size": len(X_test),
            "valid_anchors": n_valid,
            "total_anchors": len(anchors),
            "sumerian_vocab": len(sum_vocab),
            "fused_dim": int(sum_vectors.shape[1]),
            "target_dim": int(eng_vectors.shape[1]),
            "gemma_model": gemma_model,
            "gloss_hit_rate": gloss_hit_rate,
            "mode": mode_label,
            "english_vocab": len(eng_vocab),
        },
    }

    results = full_results["accuracy"]

    baseline = None
    if GLOVE_BASELINE_PATH.exists():
        with open(GLOVE_BASELINE_PATH) as f:
            baseline = json.load(f).get("accuracy", {})

    full_results["baseline_glove"] = baseline
    full_results["deltas_vs_glove"] = (
        {k: results[k] - baseline[k] for k in results if k in baseline}
        if baseline
        else None
    )

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
