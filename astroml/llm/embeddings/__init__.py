"""Embeddings service for vector generation and storage."""

from .service import EmbeddingsService
from .models import EmbeddingModel, EmbeddingConfig

__all__ = ["EmbeddingsService", "EmbeddingModel", "EmbeddingConfig"]
