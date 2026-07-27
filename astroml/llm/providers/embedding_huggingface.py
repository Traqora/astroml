"""HuggingFace embedding provider.

Uses the HuggingFace Inference API with ``sentence-transformers/all-MiniLM-L6-v2``
(384-dim) by default.  Can also be pointed at any Feature Extraction endpoint.

Trade-offs
----------
- **Quality**: Good for general-purpose sentence similarity; specialised
  models (e.g., ``BAAI/bge-large-en``) can match OpenAI quality.
- **Cost**: Free tier available (rate-limited); self-hosted is free.
- **Latency**: ~150–400 ms via Inference API; <5 ms self-hosted.
- **Dimensions**: Depends on model; 384 for MiniLM-L6-v2, 768 for
  ``all-mpnet-base-v2``, 1024 for ``BAAI/bge-large-en``.
- **Requirement**: ``HUGGINGFACE_API_KEY`` env var (or empty for public
  models) + ``requests`` (stdlib-like, already available via aiohttp).
- **Note**: The HuggingFace Inference API returns embeddings as a nested
  list ``[[float, …]]``; we take ``response[0]`` for single inputs.
"""
from __future__ import annotations

import os
from typing import List

from .embedding_base import EmbeddingProvider, EmbeddingError

# Default endpoint base for the Inference API.
_HF_API_BASE = "https://api-inference.huggingface.co/pipeline/feature-extraction"


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the HuggingFace Inference API."""

    name = "huggingface"
    output_dim = 384  # all-MiniLM-L6-v2 default

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        timeout: float = 0.45,
    ) -> None:
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY", "")
        self.model = model
        self.timeout = timeout
        # Map well-known models to their output dimensions.
        _DIMS = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "BAAI/bge-large-en": 1024,
            "BAAI/bge-base-en": 768,
        }
        self.output_dim = _DIMS.get(model, 768)
        self._url = f"{_HF_API_BASE}/{model}"

    def is_available(self) -> bool:
        # The API works without a key for public models, but may be rate-limited.
        # We only report unavailable if we can't even import requests.
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def embed(self, text: str) -> List[float]:
        try:
            import requests
            payload = {"inputs": text, "options": {"wait_for_model": True}}
            resp = requests.post(
                self._url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # Feature-extraction endpoint returns [[float, ...]] for a single string.
            if isinstance(data, list) and data and isinstance(data[0], list):
                return data[0]
            if isinstance(data, list) and data and isinstance(data[0], float):
                return data
            raise EmbeddingError(f"Unexpected HuggingFace response shape: {type(data)}")
        except ImportError as exc:
            raise EmbeddingError("requests package not installed") from exc
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"HuggingFace embed failed: {exc}") from exc

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            import requests
            payload = {"inputs": texts, "options": {"wait_for_model": True}}
            resp = requests.post(
                self._url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # Returns [[float, ...], [float, ...], ...] for a list of strings.
            if isinstance(data, list) and data and isinstance(data[0], list):
                return data
            raise EmbeddingError(f"Unexpected HuggingFace batch response shape: {type(data)}")
        except ImportError as exc:
            raise EmbeddingError("requests package not installed") from exc
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"HuggingFace embed_batch failed: {exc}") from exc
