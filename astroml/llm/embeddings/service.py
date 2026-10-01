"""Embeddings generation and retrieval service."""

import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any

from .models import EmbeddingConfig


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        pass


class EmbeddingsService:
    """Service for generating, storing, and retrieving embeddings."""

    def __init__(self, config -> Any: EmbeddingConfig, provider: EmbeddingProvider | None = None):
        """Initialize embeddings service.

        Args:
            config: Embedding configuration
            provider: Embedding provider implementation
        """
        self.config = config
        self.provider = provider
        self.embeddings_cache: dict[str, list[float]] = {}
        self.metadata_store: dict[str, dict[str, Any]] = {}
        self.chunk_count = 0

    def chunk_text(self, text: str) -> list[str]:
        """Chunk text based on configured strategy.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if self.config.chunking_strategy.value == "fixed_size":
            return self._chunk_fixed_size(text)
        elif self.config.chunking_strategy.value == "recursive":
            return self._chunk_recursive(text)
        else:
            return [text]

    def _chunk_fixed_size(self, text: str, chunk_size: int = 500) -> list[str]:
        """Chunk text into fixed-size overlapping chunks (by tokens approximation)."""
        words = text.split()
        chunks = []
        chunk = []
        current_size = 0

        for word in words:
            word_tokens = len(word) // 4 + 1
            if current_size + word_tokens > chunk_size and chunk:
                chunks.append(" ".join(chunk))
                overlap_size = min(len(chunk), self.config.chunk_overlap // 4)
                chunk = chunk[-overlap_size:] if overlap_size > 0 else []
                current_size = sum(len(w) // 4 + 1 for w in chunk)

            chunk.append(word)
            current_size += word_tokens

        if chunk:
            chunks.append(" ".join(chunk))

        return chunks

    def _chunk_recursive(self, text: str) -> list[str]:
        """Recursively chunk text by sentences then words."""
        sentences = text.split(". ")
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_tokens = len(sentence) // 4 + 1
            if current_size + sentence_tokens > self.config.chunk_size and current_chunk:
                chunks.append(". ".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_tokens

        if current_chunk:
            chunks.append(". ".join(current_chunk))

        return chunks

    def embed_texts_batch(
        self, texts: list[str], metadata: list[dict[str, Any]] | None = None
    ) -> tuple[list[list[float]], list[str]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            metadata: Optional metadata for each text

        Returns:
            Tuple of (embeddings, text_ids)
        """
        embeddings = []
        text_ids = []

        for i, text in enumerate(texts):
            text_id = self._get_text_id(text)
            text_ids.append(text_id)

            if self.config.cache_embeddings and text_id in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[text_id])
            else:
                if self.provider:
                    emb = self.provider.embed_text(text)
                else:
                    emb = self._get_dummy_embedding()
                embeddings.append(emb)

                if self.config.cache_embeddings:
                    self.embeddings_cache[text_id] = emb

            if metadata and i < len(metadata):
                self.metadata_store[text_id] = {
                    **metadata[i],
                    "text": text,
                    "embedded_at": time.time(),
                }
            else:
                self.metadata_store[text_id] = {"text": text, "embedded_at": time.time()}

        self.chunk_count += len(texts)
        return embeddings, text_ids

    def similarity_search(
        self, query: str, top_k: int = 10, metadata_filter: dict | None = None
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar documents.

        Args:
            query: Query text
            top_k: Number of results to return
            metadata_filter: Optional metadata filtering

        Returns:
            List of (text_id, similarity_score, metadata)
        """
        if self.provider:
            query_emb = self.provider.embed_text(query)
        else:
            query_emb = self._get_dummy_embedding()

        results = []
        for text_id, embedding in self.embeddings_cache.items():
            if metadata_filter:
                meta = self.metadata_store.get(text_id, {})
                if not all(meta.get(k) == v for k, v in metadata_filter.items()):
                    continue

            similarity = self._cosine_similarity(query_emb, embedding)
            results.append((text_id, similarity, self.metadata_store.get(text_id, {})))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "cached_embeddings": len(self.embeddings_cache),
            "total_chunks": self.chunk_count,
            "cache_size_mb": len(self.embeddings_cache)
            * self.config.embedding_dim
            * 4
            / (1024 * 1024),
            "model": self.config.model.value,
            "embedding_dim": self.config.embedding_dim,
        }

    @staticmethod
    def _get_text_id(text: str) -> str:
        """Generate stable ID for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between vectors."""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _get_dummy_embedding(self) -> list[float]:
        """Get placeholder embedding (for testing without API)."""
        return [0.0] * self.config.embedding_dim
