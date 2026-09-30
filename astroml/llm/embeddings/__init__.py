"""Embeddings service for vector generation and storage."""

from .models import EmbeddingConfig, EmbeddingModel
from .service import EmbeddingsService

__all__ = ["EmbeddingsService", "EmbeddingModel", "EmbeddingConfig"]
