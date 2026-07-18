"""
Ridge Alignment & Evaluation: Map Egyptian embeddings to GloVe English space.

Pipeline:
  1. Load fused 1536d Egyptian vectors
  2. Load GloVe 300d English vectors
  3. Load anchor pairs
  4. Surface-casefold-group 64/16/20 train/val/test split
  5. Select Ridge alpha by top-1 CSLS on the validation set
  6. Retrain at the chosen alpha on train+val
  7. Evaluate Top-1/5/10 accuracy on the held-out test set
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cdist
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_LANG_ROOT = Path(__file__).parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.scripts.anchor_split import group_split, SEED, TEST_SIZE, VAL_SIZE
from shared.scripts.eval_suite import (
    CAND_SIZE,
    save_artifacts,
    score_suite,
    stratify,
    val_topk_csls,
)

MODELS_DIR = _LANG_ROOT / "models"
DATA_PROCESSED = _LANG_ROOT / "data" / "processed"
RESULTS_DIR = _LANG_ROOT / "results"
GLOVE_PATH = _REPO_ROOT / "languages" / "sumerian" / "data" / "processed" / "glove.6B.300d.txt"

SURFACE_KEY = "egyptian_raw"
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

STOPWORD_GLOSSES = {
    "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with",
    "not", "no", "be", "is", "are", "was", "were", "as", "or", "and",
    "but", "if", "so", "do", "did", "have", "has", "had", "from", "into",
    "out", "up", "down", "over", "under", "between", "during", "before",
    "after", "above", "below", "any", "some", "all", "each", "every",
    "one", "two", "three", "four", "five", "des", "de",
}


def filter_stopword_glosses(anchors):
    """Drop anchors whose gloss is a pure function word. Returns (kept, n_dropped)."""
    kept = [a for a in anchors if a["english"] not in STOPWORD_GLOSSES]
    return kept, len(anchors) - len(kept)


def build_training_data(
    anchors: list[dict],
    eg_vocab: dict[str, int],
    eg_vectors: np.ndarray,
    eng_vocab: dict[str, int],
    eng_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build aligned X (Egyptian) and Y (English) matrices from anchor pairs."""
    X_list = []
    Y_list = []
    valid = []

    eg_vocab_cf = {}
    for w, i in eg_vocab.items():
        eg_vocab_cf.setdefault(w.casefold(), i)

    for anchor in anchors:
        e_word = anchor.get("egyptian_raw", anchor.get("egyptian", ""))
        eng_word = anchor["english"]

        idx = eg_vocab.get(e_word)
        if idx is None:
            idx = eg_vocab_cf.get(e_word.casefold())
        if idx is not None and eng_word in eng_vocab:
            X_list.append(eg_vectors[idx])
            Y_list.append(eng_vectors[eng_vocab[eng_word]])
            valid.append(anchor)

    if not X_list:
        return np.array([]), np.array([]), []

    return np.array(X_list), np.array(Y_list), valid


def train_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 0.001) -> Ridge:
    """Train Ridge regression to map X -> Y."""
    model = Ridge(alpha=alpha)
    model.fit(X, Y)
    return model


def evaluate_alignment(
    Y_pred: np.ndarray,
    test_english: list[str],
    eng_vocab_list: list[str],
    eng_vectors: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Evaluate alignment accuracy using Top-K nearest neighbor retrieval."""
    norms = np.linalg.norm(Y_pred, axis=1, keepdims=True)
    norms[norms == 0] = 1
    Y_pred_norm = Y_pred / norms

    g_norms = np.linalg.norm(eng_vectors, axis=1, keepdims=True)
    g_norms[g_norms == 0] = 1
    eng_norm = eng_vectors / g_norms

    distances = cdist(Y_pred_norm, eng_norm, metric="cosine")

    results = {}
    for k in ks:
        correct = 0
        for i, eng_word in enumerate(test_english):
            nn_indices = np.argsort(distances[i])[:k]
            nn_words = [eng_vocab_list[j] for j in nn_indices]
            if eng_word in nn_words:
                correct += 1
        total = len(test_english)
        results[f"top{k}"] = (correct / total * 100) if total > 0 else 0.0

    return results


def select_alpha(
    X_train, Y_train, X_val, val_english, eng_vocab_list, eng_vectors,
    alphas, predict_transform=None,
):
    """Pick the Ridge alpha by val top-5 CSLS; plateau ties -> lowest alpha.

    Suite v2 rule: scores within one anchor's worth (100/n_val percentage
    points) of the max form a plateau, and the LOWEST alpha on it wins —
    less regularization preserves the dictionary stratum. Fixes the v1
    failure mode where a flat-noise sweep picked alpha=1e4 by a one-anchor
    margin (journal 2026-07-09, Akkadian-Gemma).

    predict_transform: optional callable applied to raw predictions before
    evaluation (Egyptian's PCA path lifts 256d back to 768d with it).
    Returns (best_alpha, sweep_records).
    """
    sweep = []
    for alpha in tqdm(alphas, desc="alpha sweep", file=sys.stderr,
                      disable=not sys.stderr.isatty()):
        model = train_ridge(X_train, Y_train, alpha=alpha)
        Y_pred = model.predict(X_val)
        if predict_transform is not None:
            Y_pred = predict_transform(Y_pred)
        top1, top5 = val_topk_csls(
            Y_pred, val_english, eng_vectors[:CAND_SIZE], eng_vocab_list[:CAND_SIZE]
        )
        sweep.append({"alpha": alpha, "val_top1_csls_exact": top1,
                      "val_top5_csls_exact": top5})
        print(f"  alpha={alpha:<10g} val top5 (CSLS/50k)={top5:.2f}%  top1={top1:.2f}%")
    best_top5 = max(r["val_top5_csls_exact"] for r in sweep)
    plateau_eps = 100.0 / max(1, len(val_english))  # one anchor, in pp
    best_alpha = min(r["alpha"] for r in sweep
                     if r["val_top5_csls_exact"] >= best_top5 - plateau_eps)
    return best_alpha, sweep


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fused_path = MODELS_DIR / "fused_embeddings_1536d.npz"
    print(f"Loading fused vectors from {fused_path}")
    fused_data = np.load(str(fused_path), allow_pickle=True)
    eg_vectors = fused_data["vectors"]
    eg_vocab_list = list(fused_data["vocab"])
    eg_vocab = {w: i for i, w in enumerate(eg_vocab_list)}
    print(f"Egyptian vocab: {len(eg_vocab)} words, {eg_vectors.shape[1]}d")

    print(f"Loading GloVe from {GLOVE_PATH}")
    glove_vocab = []
    glove_vectors_list = []
    with open(GLOVE_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            glove_vocab.append(word)
            glove_vectors_list.append(vec)
    glove_vectors = np.array(glove_vectors_list)
    eng_vocab = {w: i for i, w in enumerate(glove_vocab)}
    print(f"GloVe vocab: {len(glove_vocab)} words, {glove_vectors.shape[1]}d")

    anchor_path = DATA_PROCESSED / "english_anchors_normalized.json"
    with open(anchor_path) as f:
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

    X_train, Y_train, train_valid = build_training_data(
        train_anchors, eg_vocab, eg_vectors, eng_vocab, glove_vectors
    )
    X_val, Y_val, val_valid = build_training_data(
        val_anchors, eg_vocab, eg_vectors, eng_vocab, glove_vectors
    )
    X_test, Y_test, test_valid = build_training_data(
        test_anchors, eg_vocab, eg_vectors, eng_vocab, glove_vectors
    )
    n_valid = len(train_valid) + len(val_valid) + len(test_valid)
    print(
        f"Valid anchors: {n_valid} / {len(anchors)} — "
        f"{len(train_valid)} train / {len(val_valid)} val / {len(test_valid)} test"
    )

    print("Selecting alpha on validation...")
    val_english = [a["english"] for a in val_valid]
    best_alpha, sweep = select_alpha(
        X_train, Y_train, X_val, val_english, glove_vocab, glove_vectors, ALPHAS
    )
    print(f"Selected alpha={best_alpha}")

    X_fit = np.concatenate([X_train, X_val], axis=0)
    Y_fit = np.concatenate([Y_train, Y_val], axis=0)
    print(f"Retraining on train+val ({len(X_fit)}) at alpha={best_alpha}...")
    model = train_ridge(X_fit, Y_fit, alpha=best_alpha)

    # Artifact bundle: predictions in full target space + strata metadata.
    rng = np.random.RandomState(SEED)
    non_oov_train = list(enumerate(train_valid))
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
        "target": "glove",
        "target_cache": str(GLOVE_PATH),
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
    prefix = str(RESULTS_DIR / "eval_artifacts_glove")
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

    cand_vectors = glove_vectors[:CAND_SIZE]
    cand_vocab = glove_vocab[:CAND_SIZE]
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
            "egyptian_vocab": len(eg_vocab),
            "fused_dim": int(eg_vectors.shape[1]),
            "glove_dim": int(glove_vectors.shape[1]),
        },
    }

    results_path = RESULTS_DIR / "alignment_results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    np.savez_compressed(
        str(MODELS_DIR / "ridge_weights.npz"),
        coef=model.coef_,
        intercept=model.intercept_,
    )
    print("Ridge weights saved")


if __name__ == "__main__":
    main()
