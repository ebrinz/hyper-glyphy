import numpy as np


def test_fuse_with_zero_padding():
    from languages.egyptian.scripts.fuse_08 import fuse_embeddings

    vocab = ["nTr", "Hr", "wsjr"]
    text_vectors = np.random.randn(3, 768).astype(np.float32)

    fused, fused_vocab = fuse_embeddings(vocab, text_vectors)

    assert fused.shape == (3, 1536)
    np.testing.assert_array_equal(fused[:, :768], text_vectors)
    np.testing.assert_array_equal(fused[:, 768:], np.zeros((3, 768)))
    assert fused_vocab == vocab


def test_fuse_preserves_dtype():
    from languages.egyptian.scripts.fuse_08 import fuse_embeddings

    vocab = ["nTr"]
    text_vectors = np.random.randn(1, 768).astype(np.float32)

    fused, _ = fuse_embeddings(vocab, text_vectors)
    assert fused.dtype == np.float32
