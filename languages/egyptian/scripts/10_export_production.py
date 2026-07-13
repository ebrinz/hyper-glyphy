"""
Production Export: Dual-view Egyptian alignment.

Projects the fused Egyptian vectors into BOTH whitened-EmbeddingGemma 768d
and GloVe 300d, saving each as a separate fp16 npz alongside a shared vocab
file and a consolidated v2 metadata file.

Uses the standard library serialization module for vocab (locally-generated
data, project convention matching Sumerian pipeline).
"""
import json
import importlib
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FINAL_OUTPUT = Path(__file__).parent.parent / "final_output"

SCHEMA_VERSION = 2


def project_all_vectors(
    eg_vectors: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    """Project all Egyptian vectors into a target space using learned Ridge weights."""
    projected = eg_vectors @ coef.T + intercept
    return projected.astype(np.float16)


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    FINAL_OUTPUT.mkdir(parents=True, exist_ok=True)

    fused_data = np.load(str(MODELS_DIR / "fused_embeddings_1536d.npz"), allow_pickle=True)
    eg_vectors = fused_data["vectors"]
    eg_vocab = list(fused_data["vocab"])
    print(f"Egyptian vectors: {eg_vectors.shape}, vocab: {len(eg_vocab)}")

    glove_ridge = np.load(str(MODELS_DIR / "ridge_weights.npz"))
    glove_coef = glove_ridge["coef"]
    glove_intercept = glove_ridge["intercept"]
    print(f"GloVe ridge coef: {glove_coef.shape}")
    aligned_glove = project_all_vectors(eg_vectors, glove_coef, glove_intercept)
    np.savez_compressed(
        str(FINAL_OUTPUT / "egyptian_aligned_vectors.npz"),
        vectors=aligned_glove,
    )
    print(f"GloVe aligned: {aligned_glove.shape} ({aligned_glove.dtype})")

    gemma_ridge = np.load(str(MODELS_DIR / "ridge_weights_gemma_whitened.npz"))
    gemma_coef = gemma_ridge["coef"]
    gemma_intercept = gemma_ridge["intercept"]
    print(f"Gemma whitened ridge coef: {gemma_coef.shape}")
    aligned_gemma_reduced = project_all_vectors(eg_vectors, gemma_coef, gemma_intercept)
    if "pca_components" in gemma_ridge.files:
        pca_components = gemma_ridge["pca_components"]
        pca_mean = gemma_ridge["pca_mean"]
        print(f"Lifting PCA-{pca_components.shape[0]}d -> {pca_components.shape[1]}d")
        aligned_gemma = (aligned_gemma_reduced.astype(np.float32) @ pca_components + pca_mean).astype(np.float16)
    else:
        aligned_gemma = aligned_gemma_reduced
    np.savez_compressed(
        str(FINAL_OUTPUT / "egyptian_aligned_gemma_vectors.npz"),
        vectors=aligned_gemma,
    )
    print(f"Gemma aligned: {aligned_gemma.shape} ({aligned_gemma.dtype})")

    _serial = importlib.import_module("pickle")
    with open(FINAL_OUTPUT / "egyptian_aligned_vocab.pkl", "wb") as f:
        _serial.dump(eg_vocab, f)

    glove_results = _load_json_if_exists(RESULTS_DIR / "alignment_results.json") or {}
    gemma_results = _load_json_if_exists(RESULTS_DIR / "alignment_results_gemma_whitened.json") or {}

    glove_cfg = glove_results.get("config", {})
    gemma_cfg = gemma_results.get("config", {})

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "methodology": (
            "Hyper-glyphy dual-view "
            "(Egyptian 1536d -> whitened-EmbeddingGemma 768d primary, GloVe 300d secondary)"
        ),
        "shared": {
            "vocab_size": len(eg_vocab),
            "egyptian_fused_dim": int(eg_vectors.shape[1]),
            "random_state": gemma_cfg.get("seed", glove_cfg.get("seed", 42)),
            "train_size": gemma_cfg.get("train_size", glove_cfg.get("train_size")),
            "test_size_count": gemma_cfg.get("test_size", glove_cfg.get("test_size")),
            "valid_anchors": gemma_cfg.get("valid_anchors", glove_cfg.get("valid_anchors")),
            "total_anchors": gemma_cfg.get("total_anchors", glove_cfg.get("total_anchors")),
        },
        "spaces": {
            "gemma": {
                "dim": int(aligned_gemma.shape[1]),
                "dtype": str(aligned_gemma.dtype),
                "ridge_alpha": gemma_cfg.get("alpha", 100),
                "ridge_source": "models/ridge_weights_gemma_whitened.npz",
                "target_source": "shared/models/english_gemma_whitened_768d.npz",
                "encoder_model": gemma_cfg.get("gemma_model") or "google/embeddinggemma-300m",
                "accuracy": gemma_results.get("accuracy"),
            },
            "glove": {
                "dim": int(aligned_glove.shape[1]),
                "dtype": str(aligned_glove.dtype),
                "ridge_alpha": glove_cfg.get("alpha", 0.001),
                "ridge_source": "models/ridge_weights.npz",
                "target_source": "languages/sumerian/data/processed/glove.6B.300d.txt",
                "accuracy": glove_results.get("accuracy"),
            },
        },
    }

    with open(FINAL_OUTPUT / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProduction files saved to {FINAL_OUTPUT}/")


if __name__ == "__main__":
    main()
