"""Cohere embedding provider.

Uses ``embed-english-v3.0`` (1024-dim) by default.

Trade-offs
----------
- **Quality**: Very competitive with OpenAI for retrieval tasks; slightly
  behind on general semantic similarity benchmarks.
- **Cost**: ~$0.10 per 1M tokens (as of 2024) — cheaper than OpenAI for
  large volumes.
- **Latency**: ~120–250 ms per single call; batch API available.
- **Dimensions**: 1024 (v3 models).
- **Requirement**: ``COHERE_API_KEY`` env var + ``cohere`` package.
- **Note**: Cohere requires an ``input_type`` parameter (e.g.,
  ``"search_document"`` vs ``"search_query"``); defaults to
  ``"search_document"`` here which works for most caching use-cases.
"""
from __future__ import annotations

import os
from typing import List

from .embedding_base import EmbeddingProvider, EmbeddingError


class CohereEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the Cohere Embed API."""

    name = "cohere"
    output_dim = 1024  # embed-english-v3.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
        timeout: float = 0.45,
    ) -> None:
        self.api_key = api_key or os.getenv("COHERE_API_KEY", "")
        self.model = model
        self.input_type = input_type
        self.timeout = timeout
        _DIMS = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
        }
        self.output_dim = _DIMS.get(model, 1024)

    def is_available(self) -> bool:
        try:
            import cohere  # noqa: F401
            return bool(self.api_key)
        except ImportError:
            return False

    def embed(self, text: str) -> List[float]:
        try:
            import cohere
            client = cohere.Client(api_key=self.api_key, timeout=self.timeout)
            response = client.embed(
                texts=[text],
                model=self.model,
                input_type=self.input_type,
            )
            return list(response.embeddings[0])
        except ImportError as exc:
            raise EmbeddingError("cohere package not installed") from exc
        except Exception as exc:
            raise EmbeddingError(f"Cohere embed failed: {exc}") from exc

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            import cohere
            client = cohere.Client(api_key=self.api_key, timeout=self.timeout)
            response = client.embed(
                texts=texts,
                model=self.model,
                input_type=self.input_type,
            )
            return [list(v) for v in response.embeddings]
        except ImportError as exc:
            raise EmbeddingError("cohere package not installed") from exc
        except Exception as exc:
            raise EmbeddingError(f"Cohere embed_batch failed: {exc}") from exc
