"""
Embedding Fusion: Concatenate FastText 768d with 768d zero-padding.

Produces 1536d fused vectors. The zero-padding acts as implicit regularization
during Ridge regression alignment, as discovered in heiroglyphy V15.
"""
import numpy as np
from gensim.models import KeyedVectors
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def fuse_embeddings(
    vocab: list[str],
    text_vectors: np.ndarray,
    pad_dim: int = 768,
) -> tuple[np.ndarray, list[str]]:
    """
    Fuse text embeddings with zero-padding.

    Args:
        vocab: List of words
        text_vectors: (N, text_dim) array of text embeddings
        pad_dim: Dimension of zero-padding (default 768)

    Returns:
        fused: (N, text_dim + pad_dim) fused vectors
        vocab: Same word list (passthrough)
    """
    n, text_dim = text_vectors.shape
    padding = np.zeros((n, pad_dim), dtype=np.float32)
    fused = np.concatenate([text_vectors, padding], axis=1)
    return fused, vocab


def main():
    vec_path = MODELS_DIR / "fasttext_sumerian.vec"
    print(f"Loading FastText vectors from {vec_path}")
    kv = KeyedVectors.load_word2vec_format(str(vec_path))

    vocab = list(kv.index_to_key)
    text_vectors = np.array([kv[w] for w in vocab], dtype=np.float32)
    print(f"Loaded {len(vocab)} words, {text_vectors.shape[1]}d")

    fused, _ = fuse_embeddings(vocab, text_vectors)
    print(f"Fused shape: {fused.shape}")

    output_path = MODELS_DIR / "fused_embeddings_1536d.npz"
    np.savez_compressed(
        str(output_path),
        vectors=fused,
        vocab=np.array(vocab),
    )
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
