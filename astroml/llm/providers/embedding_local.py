"""Local TF-IDF embedding provider (zero-dependency fallback).

Produces bag-of-words TF-IDF vectors using only NumPy, which is already
in requirements.txt.  This provider always succeeds and has sub-millisecond
latency, making it the last-resort fallback in the ``EmbeddingRouter``.

Trade-offs
----------
- **Quality**: Lower semantic fidelity than dense models; good enough for
  exact and near-exact string matching.  Fails on paraphrases that use
  completely different vocabulary (e.g., "car" vs "automobile").
- **Cost**: Free — no network calls, no API key.
- **Latency**: < 1 ms per call.
- **Dimensions**: Dynamic (grows with vocabulary); capped at ``max_vocab``
  (default 4096).  The ``EmbeddingRouter`` pads/truncates to a fixed target
  dimension when mixing with dense providers.
- **Requirement**: ``numpy`` (already in requirements.txt).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from .embedding_base import EmbeddingProvider, EmbeddingError


class LocalEmbeddingProvider(EmbeddingProvider):
    """Zero-dependency TF-IDF embedding provider for local/fallback use."""

    name = "local"
    # output_dim is dynamic; set to 0 to signal "variable" to the router.
    output_dim = 0

    def __init__(self, max_vocab: int = 4096) -> None:
        self.max_vocab = max_vocab
        self._vocab: Dict[str, int] = {}
        self._df: Dict[str, int] = {}
        self._n_docs: int = 0

    # ------------------------------------------------------------------
    # Tokenisation & vocabulary management
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _ensure_token(self, token: str) -> Optional[int]:
        if token in self._vocab:
            return self._vocab[token]
        if len(self._vocab) >= self.max_vocab:
            return None
        idx = len(self._vocab)
        self._vocab[token] = idx
        return idx

    # ------------------------------------------------------------------
    # Core encode
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> np.ndarray:
        tokens = self._tokenise(text)
        if not tokens:
            return np.zeros(max(len(self._vocab), 1), dtype=np.float32)

        self._n_docs += 1
        tf = Counter(tokens)
        for tok in set(tokens):
            self._df[tok] = self._df.get(tok, 0) + 1
            self._ensure_token(tok)

        dim = max(len(self._vocab), 1)
        vec = np.zeros(dim, dtype=np.float32)
        n_docs = max(self._n_docs, 1)
        for tok, count in tf.items():
            idx = self._vocab.get(tok)
            if idx is None or idx >= dim:
                continue
            tf_val = count / len(tokens)
            idf_val = math.log((n_docs + 1) / (self._df.get(tok, 0) + 1)) + 1.0
            vec[idx] = tf_val * idf_val

        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec /= norm
        return vec

    # ------------------------------------------------------------------
    # EmbeddingProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return True  # Always available — pure Python + NumPy

    def embed(self, text: str) -> List[float]:
        return self._encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._encode(t).tolist() for t in texts]

    @property
    def current_dim(self) -> int:
        """Return the current vocabulary dimension (grows over time)."""
        return max(len(self._vocab), 1)
