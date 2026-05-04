"""
Dual-view Egyptian Semantic Lookup.

Find Egyptian words by English meaning in either the whitened-EmbeddingGemma
768d manifold (space="gemma", default) or the GloVe 300d manifold
(space="glove"). Both spaces share the same Egyptian vocabulary and index
order; the vectors just land in different target geometries.

Mirrors SumerianLookup API exactly for cross-civilizational consistency.

Uses the standard library serialization module for the shared Egyptian vocab
file -- locally-generated data, not untrusted input, matching the existing
project convention.
"""
from __future__ import annotations

import importlib

import numpy as np

_VALID_SPACES = ("gemma", "glove")

_serial = importlib.import_module("pickle")


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows, mapping zero-norm rows to zero (not NaN)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


class EgyptianLookup:
    def __init__(
        self,
        gemma_vectors_path: str,
        glove_vectors_path: str,
        vocab_path: str,
        gemma_english_path: str,
        glove_english_vectors: np.ndarray,
        glove_english_vocab: list[str],
    ):
        with open(vocab_path, "rb") as f:
            self.vocab: list[str] = list(_serial.load(f))

        eg_gemma = np.load(gemma_vectors_path)["vectors"].astype(np.float32)
        eg_glove = np.load(glove_vectors_path)["vectors"].astype(np.float32)
        if eg_gemma.shape[0] != len(self.vocab):
            raise ValueError(
                f"Gemma-space Egyptian rows {eg_gemma.shape[0]} "
                f"!= vocab size {len(self.vocab)}"
            )
        if eg_glove.shape[0] != len(self.vocab):
            raise ValueError(
                f"GloVe-space Egyptian rows {eg_glove.shape[0]} "
                f"!= vocab size {len(self.vocab)}"
            )
        if eg_gemma.shape[1] != 768:
            raise ValueError(
                f"Gemma-space Egyptian dim {eg_gemma.shape[1]} != 768"
            )
        if eg_glove.shape[1] != 300:
            raise ValueError(
                f"GloVe-space Egyptian dim {eg_glove.shape[1]} != 300"
            )

        gemma_eng = np.load(gemma_english_path)
        eng_gemma_vocab = [str(w) for w in gemma_eng["vocab"]]
        eng_gemma_vec = gemma_eng["vectors"].astype(np.float32)
        if eng_gemma_vec.shape[1] != 768:
            raise ValueError(
                f"English Gemma cache dim {eng_gemma_vec.shape[1]} != 768"
            )
        if eng_gemma_vec.shape[0] != len(eng_gemma_vocab):
            raise ValueError(
                "English Gemma vocab/vectors row count mismatch"
            )

        glove_eng_vec = np.asarray(glove_english_vectors, dtype=np.float32)
        if glove_eng_vec.shape[1] != 300:
            raise ValueError(
                f"GloVe English dim {glove_eng_vec.shape[1]} != 300"
            )
        if glove_eng_vec.shape[0] != len(glove_english_vocab):
            raise ValueError(
                "GloVe English vocab/vectors row count mismatch"
            )

        self._spaces = {
            "gemma": {
                "eg_norm": _normalize_rows(eg_gemma),
                "eg_dim": eg_gemma.shape[1],
                "eng_vocab_map": {w.lower(): i for i, w in enumerate(eng_gemma_vocab)},
                "eng_norm": _normalize_rows(eng_gemma_vec),
            },
            "glove": {
                "eg_norm": _normalize_rows(eg_glove),
                "eg_dim": eg_glove.shape[1],
                "eng_vocab_map": {w.lower(): i for i, w in enumerate(glove_english_vocab)},
                "eng_norm": _normalize_rows(glove_eng_vec),
            },
        }

    def _validate_space(self, space: str) -> None:
        if space not in _VALID_SPACES:
            raise ValueError(
                f"space must be one of {_VALID_SPACES!r}, got {space!r}"
            )

    def _english_vector(self, word: str, space: str) -> np.ndarray | None:
        s = self._spaces[space]
        idx = s["eng_vocab_map"].get(word.lower())
        if idx is None:
            return None
        return s["eng_norm"][idx]

    def _top_k_from_query(self, query: np.ndarray, space: str, top_k: int) -> list[tuple[str, float]]:
        s = self._spaces[space]
        sims = s["eg_norm"] @ query
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.vocab[int(i)], float(sims[int(i)])) for i in top_indices]

    def find(self, english_word: str, top_k: int = 10, space: str = "gemma") -> list[tuple[str, float]]:
        self._validate_space(space)
        vec = self._english_vector(english_word, space)
        if vec is None:
            return []
        return self._top_k_from_query(vec, space, top_k)

    def find_both(self, english_word: str, top_k: int = 10) -> dict[str, list[tuple[str, float]]]:
        return {
            "gemma": self.find(english_word, top_k=top_k, space="gemma"),
            "glove": self.find(english_word, top_k=top_k, space="glove"),
        }

    def find_analogy(
        self,
        a: str,
        b: str,
        c: str,
        top_k: int = 10,
        space: str = "gemma",
    ) -> list[tuple[str, float]]:
        self._validate_space(space)
        va = self._english_vector(a, space)
        vb = self._english_vector(b, space)
        vc = self._english_vector(c, space)
        if any(v is None for v in (va, vb, vc)):
            return []
        target = vc - va + vb
        norm = np.linalg.norm(target)
        if norm == 0:
            return []
        target = target / norm
        return self._top_k_from_query(target, space, top_k)

    def find_blend(
        self,
        weights: dict[str, float],
        top_k: int = 10,
        space: str = "gemma",
    ) -> list[tuple[str, float]]:
        self._validate_space(space)
        if not weights:
            return []
        s = self._spaces[space]
        target = np.zeros(s["eg_dim"], dtype=np.float32)
        any_resolved = False
        for word, weight in weights.items():
            vec = self._english_vector(word, space)
            if vec is not None:
                target += float(weight) * vec
                any_resolved = True
        if not any_resolved:
            return []
        norm = np.linalg.norm(target)
        if norm == 0:
            return []
        target = target / norm
        return self._top_k_from_query(target, space, top_k)
