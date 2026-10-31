"""Semantic similarity cache using embeddings."""

import hashlib
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    """Semantic cache that matches similar prompts using embeddings."""

    def __init__(
        self,
        store: Any | None = None,
        embedding_provider: Any | None = None,
        similarity_threshold: float = 0.95,
        ttl: int = 86400,
    ):
        if store is None:
            from astroml.llm.cache.store import DiskStore

            store = DiskStore()
        self.store = store
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        if self.embedding_provider is None:
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
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _prune_expired(self) -> None:
        entries = self.store.scan_prefix("semantic:")
        now = time.time()
        for cache_key, _ in entries:
            metadata = self.store.get_metadata(cache_key)
            expires_at = metadata.get("expires_at") if metadata else None
            if expires_at is not None and expires_at <= now:
                self.store.delete(cache_key)

    def get(self, prompt: str, **kwargs) -> str | None:
        self._prune_expired()
        prompt_embedding = self._get_embedding(prompt)
        entries = self.store.scan_prefix("semantic:")
        if not entries:
            return None

        best_match = None
        best_similarity = 0.0

        for cache_key, cached_response in entries:
            metadata = self.store.get_metadata(cache_key)
            if not metadata or "embedding" not in metadata:
                continue
            expires_at = metadata.get("expires_at")
            if expires_at is not None and expires_at <= time.time():
                self.store.delete(cache_key)
                continue
            cached_embedding = np.array(metadata["embedding"])
            similarity = self._cosine_similarity(prompt_embedding, cached_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_response

        if best_similarity >= self.similarity_threshold:
            logger.info("Semantic cache hit (similarity=%.3f)", best_similarity)
            return best_match

        logger.debug("No semantic match found (best=%.3f)", best_similarity)
        return None

    def set(self, prompt: str, response: str, ttl: int | None = None, **kwargs) -> None:
        prompt_embedding = self._get_embedding(prompt)
        cache_key = f"semantic:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}"
        metadata = {
            "embedding": prompt_embedding.tolist(),
            "prompt_preview": prompt[:100],
            "expires_at": time.time() + (ttl if ttl is not None else self.ttl),
        }
        self.store.set(cache_key, response, ttl=ttl if ttl is not None else self.ttl, metadata=metadata)
        logger.debug("Cached semantic entry: %s...", cache_key[:16])


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.85, ttl: int = 86400, **kwargs) -> None:
        self.threshold = similarity_threshold
        self.ttl = ttl
        self.cache: dict[str, tuple[str, list[float], float]] = {}

    def get(self, query: str) -> str | None:
        from astroml.search.embedders import get_embedder

        now = time.time()
        for cached_query, (response, cached_vec, expires_at) in list(self.cache.items()):
            if expires_at <= now:
                del self.cache[cached_query]
        if not self.cache:
            return None

        embedder = get_embedder()
        query_vec = embedder.generate_embedding(query)

        best_query = None
        best_score = -1.0
        for cached_query, (response, cached_vec, expires_at) in self.cache.items():
            if expires_at <= time.time():
                del self.cache[cached_query]
                continue
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
        self.cache[query] = (response, vec, time.time() + self.ttl)


SemanticCache = SemanticCache
