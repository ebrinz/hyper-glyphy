"""
Production Export: Dual-view Sanskrit alignment.

Projects the fused Sanskrit vectors into BOTH whitened-EmbeddingGemma 768d
and GloVe 300d, saving each as a separate fp16 npz alongside a shared vocab
JSON and a consolidated v2 metadata file.

Uses JSON for vocab serialization (safe for cross-platform use).

See: docs/superpowers/specs/2026-04-16-phase-b-gemma-downstream-design.md
"""
import json
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FINAL_OUTPUT = Path(__file__).parent.parent / "final_output"

SCHEMA_VERSION = 2


def project_all_vectors(
    akk_vectors: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    """Project all Sanskrit vectors into a target space using learned Ridge weights.

    Works for any (target_dim, fused_dim) ridge. Returns fp16 for on-disk compactness.
    """
    projected = akk_vectors @ coef.T + intercept
    return projected.astype(np.float16)


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    FINAL_OUTPUT.mkdir(parents=True, exist_ok=True)

    fused_data = np.load(str(MODELS_DIR / "fused_embeddings_1536d.npz"), allow_pickle=True)
    akk_vectors = fused_data["vectors"]
    akk_vocab = list(fused_data["vocab"])
    print(f"Sanskrit vectors: {akk_vectors.shape}, vocab: {len(akk_vocab)}")

    glove_ridge = np.load(str(MODELS_DIR / "ridge_weights.npz"))
    glove_coef = glove_ridge["coef"]
    glove_intercept = glove_ridge["intercept"]
    print(f"GloVe ridge coef: {glove_coef.shape}")
    aligned_glove = project_all_vectors(akk_vectors, glove_coef, glove_intercept)
    np.savez_compressed(
        str(FINAL_OUTPUT / "sanskrit_aligned_vectors.npz"),
        vectors=aligned_glove,
    )
    print(f"GloVe aligned: {aligned_glove.shape} ({aligned_glove.dtype})")

    gemma_ridge = np.load(str(MODELS_DIR / "ridge_weights_gemma_whitened.npz"))
    gemma_coef = gemma_ridge["coef"]
    gemma_intercept = gemma_ridge["intercept"]
    print(f"Gemma whitened ridge coef: {gemma_coef.shape}")
    aligned_gemma = project_all_vectors(akk_vectors, gemma_coef, gemma_intercept)
    np.savez_compressed(
        str(FINAL_OUTPUT / "sanskrit_aligned_gemma_vectors.npz"),
        vectors=aligned_gemma,
    )
    print(f"Gemma aligned: {aligned_gemma.shape} ({aligned_gemma.dtype})")

    with open(FINAL_OUTPUT / "sanskrit_aligned_vocab.json", "w", encoding="utf-8") as f:
        json.dump(akk_vocab, f, ensure_ascii=False)

    glove_results = _load_json_if_exists(RESULTS_DIR / "alignment_results.json") or {}
    gemma_results = _load_json_if_exists(RESULTS_DIR / "alignment_results_gemma_whitened.json") or {}

    glove_cfg = glove_results.get("config", {})
    gemma_cfg = gemma_results.get("config", {})

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "methodology": (
            "Cuneiformy-Sanskrit dual-view "
            "(Sanskrit 1536d -> whitened-EmbeddingGemma 768d primary, GloVe 300d secondary)"
        ),
        "shared": {
            "vocab_size": len(akk_vocab),
            "sanskrit_fused_dim": int(akk_vectors.shape[1]),
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
                "target_source": "models/english_gemma_whitened_768d.npz",
                "whitening_transform": "models/gemma_whitening_transform.npz",
                "encoder_model": gemma_cfg.get("gemma_model") or "google/embeddinggemma-300m",
                "encoder_prompt": "Retrieval-document",
                "gloss_source": "WordNet first synset",
                "gloss_hit_rate_pct": gemma_cfg.get("gloss_hit_rate"),
                "accuracy": gemma_results.get("accuracy"),
            },
            "glove": {
                "dim": int(aligned_glove.shape[1]),
                "dtype": str(aligned_glove.dtype),
                "ridge_alpha": glove_cfg.get("alpha", 100),
                "ridge_source": "models/ridge_weights.npz",
                "target_source": "data/processed/glove.6B.300d.txt",
                "accuracy": glove_results.get("accuracy"),
            },
        },
    }

    with open(FINAL_OUTPUT / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProduction files saved to {FINAL_OUTPUT}/")


if __name__ == "__main__":
    main()
