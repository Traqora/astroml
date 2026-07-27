"""OpenAI embedding provider.

Uses ``text-embedding-3-small`` (1536-dim) by default.  Falls back
gracefully when the ``openai`` package is not installed or the API key
is absent.

Trade-offs
----------
- **Quality**: Best-in-class semantic similarity for English text.
- **Cost**: ~$0.02 per 1M tokens (as of 2024).
- **Latency**: ~100–200 ms per single-text call; batch API reduces
  per-text cost.
- **Dimensions**: 1536 (``text-embedding-3-small``), 3072
  (``text-embedding-3-large``).
- **Requirement**: ``OPENAI_API_KEY`` env var + ``openai`` package.
"""
from __future__ import annotations

import os
from typing import List

from .embedding_base import EmbeddingProvider, EmbeddingError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the OpenAI Embeddings API."""

    name = "openai"
    output_dim = 1536  # text-embedding-3-small default

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        timeout: float = 0.45,  # stay well under 500 ms fallback budget
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.timeout = timeout
        # Dimension is model-dependent.
        _DIMS = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self.output_dim = _DIMS.get(model, 1536)

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
            return bool(self.api_key)
        except ImportError:
            return False

    def embed(self, text: str) -> List[float]:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, timeout=self.timeout)
            response = client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding
        except ImportError as exc:
            raise EmbeddingError("openai package not installed") from exc
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embed failed: {exc}") from exc

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, timeout=self.timeout)
            response = client.embeddings.create(input=texts, model=self.model)
            # API returns results in the same order as input.
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        except ImportError as exc:
            raise EmbeddingError("openai package not installed") from exc
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embed_batch failed: {exc}") from exc
