"""
HittiteLookup: query the production Hittite alignment artifacts.

Mirrors AkkadianLookup. Two views: 'gemma' (768d, primary, whitened) and
'glove' (300d, secondary). Vocab is serialized as JSON for safety.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

_VECTORS = {
    "gemma": _HERE / "hittite_aligned_gemma_vectors.npz",
    "glove": _HERE / "hittite_aligned_vectors.npz",
}
_VOCAB_PATH = _HERE / "hittite_aligned_vocab.json"


class HittiteLookup:
    """Look up English neighbors for Hittite words in the aligned space."""

    def __init__(self, space: str = "gemma"):
        if space not in _VECTORS:
            raise ValueError(f"space must be one of {list(_VECTORS)}, got {space!r}")
        self.space = space
        with open(_VOCAB_PATH, encoding="utf-8") as f:
            self.vocab: list[str] = json.load(f)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.vectors = np.load(str(_VECTORS[space]))["vectors"].astype(np.float32)

    def lookup(self, word: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k vocab words nearest to `word` by cosine similarity."""
        idx = self.word_to_idx.get(word)
        if idx is None:
            return []
        v = self.vectors[idx]
        norms = np.linalg.norm(self.vectors, axis=1)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return []
        cos = (self.vectors @ v) / (norms * v_norm + 1e-12)
        cos[idx] = -1.0
        top = np.argsort(-cos)[:k]
        return [(self.vocab[i], float(cos[i])) for i in top]
