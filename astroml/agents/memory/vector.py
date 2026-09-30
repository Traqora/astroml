"""Vector-based memory with semantic search capabilities."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .base import MemoryStore, Message


class VectorMemory(MemoryStore):
    """Memory store with vector-based semantic search."""

    def __init__(self, embedding_dim: int = 768) -> None:
        """Initialize vector memory.

        Args:
            embedding_dim: Dimension of embedding vectors
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self._embeddings: List[List[float]] = []

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add message with optional embedding.

        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
            embedding: Optional pre-computed embedding
        """
        super().add_message(role, content, metadata)

        # Use provided embedding or compute simple hash-based embedding
        if embedding is None:
            embedding = self._compute_simple_embedding(content)
        self._embeddings.append(embedding)

    def _compute_simple_embedding(self, text: str) -> List[float]:
        """Compute a simple hash-based embedding (placeholder for real embeddings).

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        # Simple hash-based embedding for demonstration
        # In production, use actual embedding models (sentence-transformers, OpenAI, etc.)
        embedding = [0.0] * self.embedding_dim
        for i, char in enumerate(text):
            idx = (ord(char) * (i + 1)) % self.embedding_dim
            embedding[idx] += 0.1

        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[tuple[Message, float]]:
        """Search messages by semantic similarity.

        Args:
            query: Search query
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (message, similarity_score) tuples
        """
        if not self._messages:
            return []

        query_embedding = self._compute_simple_embedding(query)
        similarities = []

        for i, msg_embedding in enumerate(self._embeddings):
            similarity = self._cosine_similarity(query_embedding, msg_embedding)
            if similarity >= threshold:
                similarities.append((self._messages[i], similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_context_by_similarity(
        self,
        query: str,
        max_tokens: int = 2000,
        avg_tokens_per_char: float = 0.25,
    ) -> List[Message]:
        """Get relevant context based on semantic similarity.

        Args:
            query: Query to find relevant context for
            max_tokens: Maximum tokens in context
            avg_tokens_per_char: Average tokens per character

        Returns:
            List of relevant messages
        """
        results = self.similarity_search(query, top_k=20)
        context = []
        total_chars = 0
        max_chars = int(max_tokens / avg_tokens_per_char)

        for msg, _ in results:
            msg_chars = len(msg.content)
            if total_chars + msg_chars > max_chars:
                break
            context.append(msg)
            total_chars += msg_chars

        return context
