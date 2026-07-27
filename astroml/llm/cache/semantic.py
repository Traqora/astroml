"""Semantic similarity cache using embeddings."""
import logging
import numpy as np
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class SemanticCache:
    """Semantic cache that matches similar prompts using embeddings."""

    def __init__(
        self,
        store: "CacheStore",
        embedding_provider: "EmbeddingProvider" = None,
        similarity_threshold: float = 0.95,
    ):
        """Initialize semantic cache.

        Args:
            store: Backend storage implementation
            embedding_provider: Provider for generating embeddings
            similarity_threshold: Minimum cosine similarity for cache hit
        """
        self.store = store
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self._embedding_cache = {}  # In-memory cache for embeddings

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, with local caching."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        if self.embedding_provider is None:
            # Fallback: use simple bag-of-words representation
            from collections import Counter
            words = text.lower().split()
            vocab_size = 1000
            vec = np.zeros(vocab_size)
            for word in words:
                idx = hash(word) % vocab_size
                vec[idx] += 1
            if vec.sum() > 0:
                vec = vec / vec.sum()
            embedding = vec
        else:
            embedding = self.embedding_provider.embed(text)

        self._embedding_cache[text] = embedding
        return embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get(self, prompt: str, **kwargs) -> Optional[str]:
        """Retrieve cached response for similar prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (currently unused)

        Returns:
            Cached response if similar prompt found, else None
        """
        prompt_embedding = self._get_embedding(prompt)

        # Get all semantic cache entries
        entries = self.store.scan_prefix("semantic:")
        if not entries:
            return None

        best_match = None
        best_similarity = 0.0

        for cache_key, cached_response in entries:
            # Extract stored embedding from metadata
            metadata = self.store.get_metadata(cache_key)
            if not metadata or "embedding" not in metadata:
                continue

            cached_embedding = np.array(metadata["embedding"])
            similarity = self._cosine_similarity(prompt_embedding, cached_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_response

        if best_similarity >= self.similarity_threshold:
            logger.info(f"Semantic cache hit (similarity={best_similarity:.3f})")
            return best_match

        logger.debug(f"No semantic match found (best={best_similarity:.3f})")
        return None

    def set(
        self,
        prompt: str,
        response: str,
        ttl: int = 3600,
        **kwargs,
    ) -> None:
        """Store response with embedding for semantic matching.

        Args:
            prompt: Input prompt
            response: LLM response
            ttl: Time to live in seconds
            **kwargs: Additional parameters
        """
        prompt_embedding = self._get_embedding(prompt)

        # Use hash as key (same as exact match)
        import hashlib
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_key = f"semantic:{prompt_hash}"

        # Store response with embedding metadata
        metadata = {
            "embedding": prompt_embedding.tolist(),
            "prompt_preview": prompt[:100],
        }

        self.store.set(cache_key, response, ttl=ttl, metadata=metadata)
        logger.debug(f"Cached semantic entry: {cache_key[:16]}...")
from typing import Optional, Dict, Tuple, List
from astroml.search.embedders import get_embedder

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        # Stores: query_text -> (response_text, embedding_vector)
        self.cache: Dict[str, Tuple[str, List[float]]] = {}

    def get(self, query: str) -> Optional[str]:
        if not self.cache:
            return None
            
        embedder = get_embedder()
        query_vec = embedder.generate_embedding(query)
        
        best_query = None
        best_score = -1.0
        
        for cached_query, (response, cached_vec) in self.cache.items():
            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_vec, cached_vec))
            if dot > best_score:
                best_score = dot
                best_query = cached_query
                
        if best_score >= self.threshold and best_query:
            return self.cache[best_query][0]
        return None

    def set(self, query: str, response: str):
        embedder = get_embedder()
        vec = embedder.generate_embedding(query)
        self.cache[query] = (response, vec)
