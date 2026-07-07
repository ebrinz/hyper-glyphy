"""
Ridge Alignment & Evaluation: Map Sumerian embeddings to GloVe English space.

Pipeline:
  1. Load fused 1536d Greek vectors
  2. Load GloVe 300d English vectors
  3. Load anchor pairs
  4. Lemma-group 64/16/20 train/val/test split (no lemma or surface spans partitions)
  5. Select Ridge alpha by top-1 on the validation set
  6. Retrain at the chosen alpha on train+val
  7. Evaluate Top-1/5/10 accuracy on the held-out test set

OOV anchors (FastText subword inference) are training-only; validation and
test contain in-vocab anchors exclusively.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cdist
import sys
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.scripts.anchor_split import group_split, SEED, TEST_SIZE, VAL_SIZE
from shared.scripts.eval_suite import (
    CAND_SIZE,
    save_artifacts,
    score_suite,
    stratify,
    val_top1_csls,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"

SURFACE_KEY = "greek"
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]


def build_training_data(
    anchors: list[dict],
    sum_vocab: dict[str, int],
    sum_vectors: np.ndarray,
    eng_vocab: dict[str, int],
    eng_vectors: np.ndarray,
    fasttext_model=None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build aligned X (Greek) and Y (English) matrices from anchor pairs.

    L5: when fasttext_model is provided, OOV anchors fall back to FastText's
    subword inference. The inferred 768d vector is zero-padded to match the
    fused 1536d format.
    """
    X_list = []
    Y_list = []
    valid = []

    # Determine fused dimension and FastText dimension from sum_vectors
    fused_dim = sum_vectors.shape[1] if sum_vectors.size else 1536
    pad_dim = fused_dim - (fasttext_model.vector_size if fasttext_model else fused_dim // 2)
    if pad_dim < 0:
        pad_dim = 0

    for anchor in anchors:
        s_word = anchor.get("greek") or anchor.get("sumerian")
        e_word = anchor["english"]
        if e_word not in eng_vocab:
            continue

        if s_word in sum_vocab:
            X_list.append(sum_vectors[sum_vocab[s_word]])
            Y_list.append(eng_vectors[eng_vocab[e_word]])
            valid.append(anchor)
        elif fasttext_model is not None:
            # L5: subword inference for OOV s_word.
            try:
                ft_vec = fasttext_model.wv.get_vector(s_word).astype(np.float32)
            except Exception:
                continue
            padded = np.concatenate([ft_vec, np.zeros(pad_dim, dtype=np.float32)])
            if padded.shape[0] != fused_dim:
                continue  # dimension mismatch safety
            X_list.append(padded)
            Y_list.append(eng_vectors[eng_vocab[e_word]])
            valid.append({**anchor, "subword_inferred": True})

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
    glove_vocab: list[str],
    glove_vectors: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Evaluate alignment accuracy using Top-K nearest neighbor retrieval."""
    norms = np.linalg.norm(Y_pred, axis=1, keepdims=True)
    norms[norms == 0] = 1
    Y_pred_norm = Y_pred / norms

    g_norms = np.linalg.norm(glove_vectors, axis=1, keepdims=True)
    g_norms[g_norms == 0] = 1
    glove_norm = glove_vectors / g_norms

    distances = cdist(Y_pred_norm, glove_norm, metric="cosine")

    results = {}
    for k in ks:
        correct = 0
        for i, eng_word in enumerate(test_english):
            nn_indices = np.argsort(distances[i])[:k]
            nn_words = [glove_vocab[j] for j in nn_indices]
            if eng_word in nn_words:
                correct += 1
        total = len(test_english)
        results[f"top{k}"] = (correct / total * 100) if total > 0 else 0.0

    return results


def select_alpha(
    X_train, Y_train, X_val, val_english, eng_vocab_list, eng_vectors,
    alphas, predict_transform=None,
):
    """Pick the Ridge alpha with the best top-1 on the validation set.

    predict_transform: optional callable applied to raw predictions before
    evaluation (Egyptian's PCA path lifts 256d back to 768d with it).
    Returns (best_alpha, sweep_records).
    """
    sweep = []
    best_alpha, best_top1 = None, -1.0
    for alpha in tqdm(alphas, desc="alpha sweep", file=sys.stderr,
                      disable=not sys.stderr.isatty()):
        model = train_ridge(X_train, Y_train, alpha=alpha)
        Y_pred = model.predict(X_val)
        if predict_transform is not None:
            Y_pred = predict_transform(Y_pred)
        top1 = val_top1_csls(
            Y_pred, val_english, eng_vectors[:CAND_SIZE], eng_vocab_list[:CAND_SIZE]
        )
        sweep.append({"alpha": alpha, "val_top1_csls_exact": top1})
        print(f"  alpha={alpha:<10g} val top1 (CSLS/50k)={top1:.2f}%")
        if top1 > best_top1:
            best_alpha, best_top1 = alpha, top1
    return best_alpha, sweep


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fused_path = MODELS_DIR / "fused_embeddings_1536d.npz"
    print(f"Loading fused vectors from {fused_path}")
    fused_data = np.load(str(fused_path), allow_pickle=True)
    sum_vectors = fused_data["vectors"]
    sum_vocab_list = list(fused_data["vocab"])
    sum_vocab = {w: i for i, w in enumerate(sum_vocab_list)}
    print(f"Sumerian vocab: {len(sum_vocab)} words, {sum_vectors.shape[1]}d")

    glove_path = DATA_PROCESSED / "glove.6B.300d.txt"
    print(f"Loading GloVe from {glove_path}")
    glove_vocab = []
    glove_vectors_list = []
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            glove_vocab.append(word)
            glove_vectors_list.append(vec)
    glove_vectors = np.array(glove_vectors_list)
    eng_vocab = {w: i for i, w in enumerate(glove_vocab)}
    print(f"GloVe vocab: {len(glove_vocab)} words, {glove_vectors.shape[1]}d")

    anchor_path = DATA_PROCESSED / "english_anchors.json"
    with open(anchor_path) as f:
        anchors = json.load(f)
    print(f"Loaded {len(anchors)} anchors")

    # L5: load FastText model for OOV subword inference
    from gensim.models import FastText
    ft_path = MODELS_DIR / "fasttext_sumerian.model"
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
        train_anchors, sum_vocab, sum_vectors, eng_vocab, glove_vectors,
        fasttext_model=ft_model,
    )
    X_val, Y_val, val_valid = build_training_data(
        val_anchors, sum_vocab, sum_vectors, eng_vocab, glove_vectors
    )
    X_test, Y_test, test_valid = build_training_data(
        test_anchors, sum_vocab, sum_vectors, eng_vocab, glove_vectors
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
        X_train, Y_train, X_val, val_english, glove_vocab, glove_vectors, ALPHAS
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
        "target": "glove",
        "target_cache": str(glove_path),
        "alpha": best_alpha,
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
            "sumerian_vocab": len(sum_vocab),
            "fused_dim": int(sum_vectors.shape[1]),
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
