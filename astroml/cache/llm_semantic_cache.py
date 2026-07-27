"""Semantic similarity cache for LLM responses.

This module implements a similarity-based lookup layer:
- Compute an embedding for the incoming prompt
- Find cached prompts with cosine similarity >= threshold
- Return cached response (cache hit)
- Otherwise caller is expected to compute the LLM response and store it

Design goals (for acceptance):
- Redis-backed storage for cached responses + prompt embeddings
- Lookup fast: keep the number of candidates bounded

Implementation notes:
- To avoid heavy vector dependencies, we store embeddings as floats and do an
  in-Python scan over a limited candidate set.
- Candidate selection strategy: time-ordered buckets via Redis sorted sets.
  We keep a ZSET per model+namespace and fetch the most recent K items.

If you already have a vector index setup (RedisVector/pgvector), this module
can be swapped without changing the wrapper interface.
"""

from __future__ import annotations

import os
import time
import math
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import redis

from astroml.cache.redis_cache import RedisCache

logger = logging.getLogger(__name__)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    # Defensive: handle mismatched dimensions.
    if not a or not b or len(a) != len(b):
        return -1.0

    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return -1.0
    return dot / math.sqrt(na * nb)


@dataclass(frozen=True)
class SemanticCacheConfig:
    """Configuration for semantic caching."""

    namespace: str = "llm:semantic"
    similarity_threshold: float = float(os.environ.get("LLM_CACHE_SIMILARITY_THRESHOLD", "0.88"))
    ttl_seconds: int = int(os.environ.get("LLM_CACHE_TTL_SECONDS", "600"))
    candidate_top_k: int = int(os.environ.get("LLM_CACHE_LOOKBACK_K", "200"))


class LLMEmbeddingProvider:
    """Embedding provider interface.

    Provide an implementation that returns a dense embedding vector for the input text.
    """

    def embed(self, *, text: str, model: str) -> List[float]:  # pragma: no cover
        raise NotImplementedError


class DefaultNoopEmbeddingProvider(LLMEmbeddingProvider):
    """Fallback provider.

    This is a placeholder so the semantic cache module is importable even when no
    embedding provider is configured. The wrapper should inject a real provider.
    """

    def embed(self, *, text: str, model: str) -> List[float]:
        raise RuntimeError(
            "No embedding provider configured. "
            "Provide an embedding provider to LLMSemanticCacheWrapper."
        )


@dataclass(frozen=True)
class SemanticCacheHit:
    response: Any
    similarity: float
    cache_key: str
    cached_at: Optional[float] = None


class LLMSemanticCache:
    """Redis-backed semantic cache.

    Storage layout (keys):
    - response payload: {namespace}:resp:{model}:{cache_id}
    - embedding vector: {namespace}:emb:{model}:{cache_id}
    - metadata: {namespace}:meta:{model}:{cache_id}
    - index sorted set: {namespace}:idx:{model}:{bucket}

    For speed, we keep one index bucket (current time slice) and fetch top K recent.
    """

    def __init__(
        self,
        *,
        redis_cache: Optional[RedisCache] = None,
        config: Optional[SemanticCacheConfig] = None,
        embedding_provider: Optional[LLMEmbeddingProvider] = None,
    ):
        self._redis_cache = redis_cache or RedisCache()
        self._config = config or SemanticCacheConfig()
        self._embedding_provider = embedding_provider or DefaultNoopEmbeddingProvider()

        # Use the underlying Redis client from the existing RedisCache.
        # RedisCache.client is typed as Redis.
        self._redis: redis.Redis = self._redis_cache.client

    @property
    def config(self) -> SemanticCacheConfig:
        return self._config

    def _idx_key(self, *, model: str) -> str:
        # Single bucket. If needed, extend to hour/day buckets.
        return f"{self._config.namespace}:idx:{model}:all"

    def _resp_key(self, *, model: str, cache_id: str) -> str:
        return f"{self._config.namespace}:resp:{model}:{cache_id}"

    def _emb_key(self, *, model: str, cache_id: str) -> str:
        return f"{self._config.namespace}:emb:{model}:{cache_id}"

    def _meta_key(self, *, model: str, cache_id: str) -> str:
        return f"{self._config.namespace}:meta:{model}:{cache_id}"

    def _now_bucket_score(self) -> float:
        # Higher score = more recent.
        return time.time()

    def lookup(
        self,
        *,
        prompt: str,
        model: str,
        embedding_model: str,
        ttl_seconds: Optional[int] = None,
    ) -> Tuple[Optional[SemanticCacheHit], float]:
        """Lookup semantic cache.

        Returns:
            (hit, lookup_ms)
        """
        start = time.perf_counter()
        ttl_seconds = ttl_seconds or self._config.ttl_seconds

        # Compute embedding.
        query_emb = self._embedding_provider.embed(text=prompt, model=embedding_model)

        idx_key = self._idx_key(model=model)

        # Fetch bounded recent candidates.
        # ZREVRANGE gives highest scores first.
        # Candidate cache_id list is expected to be strings.
        try:
            candidate_ids = self._redis.zrevrange(idx_key, 0, self._config.candidate_top_k - 1)
        except Exception as e:  # pragma: no cover
            logger.warning("Semantic cache ZREVRANGE failed: %s", e)
            candidate_ids = []

        best: Optional[Tuple[float, str]] = None

        # Pipeline get embeddings for candidates.
        pipe = self._redis.pipeline()
        for cid_b in candidate_ids:
            cid = cid_b.decode("utf-8") if isinstance(cid_b, (bytes, bytearray)) else str(cid_b)
            pipe.get(self._emb_key(model=model, cache_id=cid))
        emb_blobs = []
        try:
            emb_blobs = pipe.execute()
        except Exception:  # pragma: no cover
            emb_blobs = []

        for cid_b, emb_blob in zip(candidate_ids, emb_blobs):
            cid = cid_b.decode("utf-8") if isinstance(cid_b, (bytes, bytearray)) else str(cid_b)
            if not emb_blob:
                continue
            try:
                emb_vec = json.loads(emb_blob)
            except Exception:
                continue
            if not isinstance(emb_vec, list):
                continue

            sim = _cosine_similarity(query_emb, emb_vec)
            if best is None or sim > best[0]:
                best = (sim, cid)

        if best is None:
            return None, (time.perf_counter() - start) * 1000.0

        best_sim, best_cid = best
        if best_sim < self._config.similarity_threshold:
            return None, (time.perf_counter() - start) * 1000.0

        # Fetch response + metadata.
        resp_key = self._resp_key(model=model, cache_id=best_cid)
        meta_key = self._meta_key(model=model, cache_id=best_cid)
        pipe = self._redis.pipeline()
        pipe.get(resp_key)
        pipe.get(meta_key)
        resp_blob, meta_blob = pipe.execute()

        if resp_blob is None:
            return None, (time.perf_counter() - start) * 1000.0

        # resp_cache uses pickle in RedisCache.get/set; but here we directly store
        # the same pickled payload so we can unpickle via RedisCache.get.
        # However we don't have the full key-format contract here; simplest is to
        # use RedisCache.get on resp_key.
        try:
            response_obj = self._redis_cache.get(resp_key)
        except Exception:
            response_obj = None

        cached_at: Optional[float] = None
        if meta_blob:
            try:
                meta = json.loads(meta_blob)
                cached_at = meta.get("cached_at")
            except Exception:
                cached_at = None

        hit = SemanticCacheHit(
            response=response_obj,
            similarity=float(best_sim),
            cache_key=f"{model}:{best_cid}",
            cached_at=cached_at,
        )
        return hit, (time.perf_counter() - start) * 1000.0

    def store(
        self,
        *,
        prompt: str,
        response: Any,
        model: str,
        embedding_model: str,
        cache_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Store response in semantic cache."""
        ttl_seconds = ttl_seconds or self._config.ttl_seconds
        cache_id = cache_id or str(int(time.time() * 1000))

        emb_vec = self._embedding_provider.embed(text=prompt, model=embedding_model)

        resp_key = self._resp_key(model=model, cache_id=cache_id)
        emb_key = self._emb_key(model=model, cache_id=cache_id)
        meta_key = self._meta_key(model=model, cache_id=cache_id)
        idx_key = self._idx_key(model=model)

        # Store embedding + metadata as JSON strings (fast to read).
        # Store response using existing RedisCache.set (pickle).
        try:
            self._redis_cache.set(resp_key, response, ttl_seconds=ttl_seconds)
        except Exception as e:  # pragma: no cover
            logger.warning("Semantic cache response set failed: %s", e)

        emb_json = json.dumps(list(map(float, emb_vec)))
        meta_json = json.dumps({"cached_at": time.time()})

        # Store embedding/metadata with TTL too.
        try:
            self._redis.setex(emb_key, ttl_seconds, emb_json)
            self._redis.setex(meta_key, ttl_seconds, meta_json)
            # Index: add candidate id; keep index roughly bounded by trimming.
            self._redis.zadd(idx_key, {cache_id: self._now_bucket_score()})
            # Soft cap: keep only recent 10x candidate_top_k
            cap = max(self._config.candidate_top_k * 10, 100)
            self._redis.zremrangebyrank(idx_key, 0, -(cap + 1))
        except Exception as e:  # pragma: no cover
            logger.warning("Semantic cache embedding/index set failed: %s", e)

        return cache_id


class SimpleDeterministicEmbeddingProvider(LLMEmbeddingProvider):
    """Developer-friendly embedding provider (NOT semantic-quality).

    Intended for unit tests / local usage when no embedding model exists.
    Produces a fixed-length vector derived from hash of text.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, *, text: str, model: str) -> List[float]:
        import hashlib
        h = hashlib.md5((model + ":" + text).encode("utf-8")).digest()
        # Expand digest to dim floats.
        out = []
        for i in range(self.dim):
            b = h[i % len(h)]
            out.append((b / 255.0) * 2.0 - 1.0)
        return out

