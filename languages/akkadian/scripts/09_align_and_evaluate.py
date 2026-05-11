"""
Ridge Alignment & Evaluation: Map Sumerian embeddings to GloVe English space.

Pipeline:
  1. Load fused 1536d Sumerian vectors
  2. Load GloVe 300d English vectors
  3. Load anchor pairs
  4. Build training data (only anchors present in both vocabs)
  5. 80/20 train/test split (random_state=42)
  6. Train Ridge regression (alpha=0.001)
  7. Evaluate Top-1/5/10 accuracy on test set
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def build_training_data(
    anchors: list[dict],
    sum_vocab: dict[str, int],
    sum_vectors: np.ndarray,
    eng_vocab: dict[str, int],
    eng_vectors: np.ndarray,
    fasttext_model=None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build aligned X (Akkadian) and Y (English) matrices from anchor pairs.

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
        s_word = anchor.get("akkadian") or anchor.get("sumerian")
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

    X, Y, valid_anchors = build_training_data(
        anchors, sum_vocab, sum_vectors, eng_vocab, glove_vectors, fasttext_model=ft_model
    )
    print(f"Valid anchors: {len(valid_anchors)} / {len(anchors)} ({len(valid_anchors)/len(anchors)*100:.1f}%)")

    # L5-refined: partition valid anchors by subword_inferred flag. Only in-vocab
    # anchors go into the test set; OOV-inferred anchors are training-only.
    in_vocab_mask = np.array([not a.get("subword_inferred") for a in valid_anchors])
    X_in = X[in_vocab_mask]
    Y_in = Y[in_vocab_mask]
    anchors_in = [a for a, m in zip(valid_anchors, in_vocab_mask) if m]
    X_oov = X[~in_vocab_mask]
    Y_oov = Y[~in_vocab_mask]
    anchors_oov = [a for a, m in zip(valid_anchors, in_vocab_mask) if not m]

    X_in_train, X_test, Y_in_train, Y_test, anchors_in_train, anchors_test = train_test_split(
        X_in, Y_in, anchors_in, test_size=0.2, random_state=42
    )

    # Combine in-vocab train + all OOV anchors as training data
    X_train = np.concatenate([X_in_train, X_oov], axis=0) if len(X_oov) else X_in_train
    Y_train = np.concatenate([Y_in_train, Y_oov], axis=0) if len(Y_oov) else Y_in_train
    anchors_train = anchors_in_train + anchors_oov

    print(
        f"In-vocab anchors: {len(anchors_in)} (split into "
        f"{len(anchors_in_train)} train + {len(anchors_test)} test). "
        f"OOV anchors (subword-inferred): {len(anchors_oov)} (added to train only)."
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("Training Ridge regression (alpha=0.001)...")
    model = train_ridge(X_train, Y_train, alpha=0.001)  # L7: matches Sumerian's GloVe optimum

    Y_pred = model.predict(X_test)

    test_english = [a["english"] for a in anchors_test]
    results = evaluate_alignment(Y_pred, test_english, glove_vocab, glove_vectors)

    print(f"\n=== RESULTS ===")
    print(f"Top-1 Accuracy:  {results['top1']:.2f}%")
    print(f"Top-5 Accuracy:  {results['top5']:.2f}%")
    print(f"Top-10 Accuracy: {results['top10']:.2f}%")

    full_results = {
        "accuracy": results,
        "config": {
            "alignment": "Ridge",
            "alpha": 0.001,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "valid_anchors": len(valid_anchors),
            "total_anchors": len(anchors),
            "sumerian_vocab": len(sum_vocab),
            "fused_dim": sum_vectors.shape[1],
            "glove_dim": glove_vectors.shape[1],
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
