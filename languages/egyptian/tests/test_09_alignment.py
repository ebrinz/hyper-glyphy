import numpy as np


def test_build_training_data():
    from languages.egyptian.scripts.align_09 import build_training_data

    anchors = [
        {"egyptian": "nTr", "english": "god"},
        {"egyptian": "Hr", "english": "face"},
        {"egyptian": "unknown", "english": "missing"},
    ]

    eg_vocab = {"nTr": 0, "Hr": 1, "wsjr": 2}
    eg_vectors = np.random.randn(3, 1536).astype(np.float32)

    eng_vocab = {"god": 0, "face": 1, "water": 2}
    eng_vectors = np.random.randn(3, 300).astype(np.float32)

    X, Y, valid_anchors = build_training_data(
        anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors
    )

    assert X.shape == (2, 1536)
    assert Y.shape == (2, 300)
    assert len(valid_anchors) == 2


def test_evaluate_alignment():
    from languages.egyptian.scripts.align_09 import evaluate_alignment

    np.random.seed(42)
    n_test = 10
    dim = 300

    Y_test = np.random.randn(n_test, dim).astype(np.float32)
    Y_pred = Y_test + np.random.randn(n_test, dim).astype(np.float32) * 0.01

    eng_vocab = [f"word_{i}" for i in range(n_test + 50)]
    eng_vectors = np.vstack([
        Y_test,
        np.random.randn(50, dim).astype(np.float32),
    ])

    test_english = [f"word_{i}" for i in range(n_test)]

    results = evaluate_alignment(Y_pred, test_english, eng_vocab, eng_vectors)

    assert "top1" in results
    assert "top5" in results
    assert "top10" in results
    assert results["top1"] > 0.5


def test_train_ridge():
    from languages.egyptian.scripts.align_09 import train_ridge

    X = np.random.randn(100, 1536).astype(np.float32)
    Y = np.random.randn(100, 300).astype(np.float32)

    model = train_ridge(X, Y, alpha=0.001)

    Y_pred = model.predict(X[:5])
    assert Y_pred.shape == (5, 300)


def test_align_09b_shape_contract_at_768d():
    """09b must work when the English target dim is 768 (EmbeddingGemma)."""
    from languages.egyptian.scripts.align_09b import build_training_data, train_ridge, evaluate_alignment

    anchors = [
        {"egyptian": "nTr", "english": "god"},
        {"egyptian": "Hr", "english": "face"},
        {"egyptian": "wsjr", "english": "osiris"},
    ]

    eg_vocab = {"nTr": 0, "Hr": 1, "wsjr": 2}
    eg_vectors = np.random.randn(3, 1536).astype(np.float32)

    eng_vocab = {"god": 0, "face": 1, "osiris": 2}
    eng_vectors = np.random.randn(3, 768).astype(np.float32)

    X, Y, valid = build_training_data(
        anchors, eg_vocab, eg_vectors, eng_vocab, eng_vectors
    )
    assert X.shape == (3, 1536)
    assert Y.shape == (3, 768)

    model = train_ridge(X, Y, alpha=100)
    assert model.coef_.shape == (768, 1536)

    Y_pred = model.predict(X)
    results = evaluate_alignment(
        Y_pred,
        ["god", "face", "osiris"],
        ["god", "face", "osiris"],
        eng_vectors,
        ks=(1, 2, 3),
    )
    assert "top1" in results
    assert "top2" in results
    assert "top3" in results
