"""Cached LLM client wrapper.

This wrapper is provider-agnostic. It expects an injected underlying client
that can execute the actual LLM call.

The wrapper performs:
1) Semantic similarity cache lookup
2) If hit: return cached response
3) If miss: call underlying LLM, then store semantic cache

It also tracks basic cache metrics in Redis (separate counters) so that the
API can expose hit rate and average lookup latency.
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Protocol

from astroml.cache.redis_cache import RedisCache
from astroml.cache.llm_semantic_cache import (
    LLMSemanticCache,
    LLMEmbeddingProvider,
    SemanticCacheConfig,
    SemanticCacheHit,
)

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    def complete(self, *, model: str, prompt: str, **kwargs: Any) -> Any:  # pragma: no cover
        ...


@dataclass(frozen=True)
class LLMCachedClientConfig:
    redis_url: Optional[str] = os.environ.get("LLM_CACHE_REDIS_URL")
    model: Optional[str] = None
    embedding_model: str = os.environ.get("LLM_CACHE_EMBEDDING_MODEL", "text-embedding-placeholder")

    similarity_threshold: float = float(os.environ.get("LLM_CACHE_SIMILARITY_THRESHOLD", "0.88"))
    ttl_seconds: int = int(os.environ.get("LLM_CACHE_TTL_SECONDS", "600"))
    candidate_top_k: int = int(os.environ.get("LLM_CACHE_LOOKBACK_K", "200"))

    metrics_prefix: str = "llm:semantic:metrics"


class LLMCachedClient:
    """Semantic caching wrapper for an LLM provider."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        embedding_provider: LLMEmbeddingProvider,
        config: Optional[LLMCachedClientConfig] = None,
        redis_cache: Optional[RedisCache] = None,
    ):
        self._provider = provider
        self._redis_cache = redis_cache or RedisCache()
        self._config = config or LLMCachedClientConfig()

        self._semantic_cache = LLMSemanticCache(
            redis_cache=self._redis_cache,
            config=SemanticCacheConfig(
                similarity_threshold=self._config.similarity_threshold,
                ttl_seconds=self._config.ttl_seconds,
                candidate_top_k=self._config.candidate_top_k,
            ),
            embedding_provider=embedding_provider,
        )

        self._redis = self._redis_cache.client

    def _metric_key(self, suffix: str) -> str:
        return f"{self._config.metrics_prefix}:{suffix}"

    def _incr(self, key: str, amount: int = 1) -> None:
        try:
            self._redis.incrby(key, amount)
        except Exception:
            pass

    def _observe_ms(self, key: str, value_ms: float) -> None:
        # Keep sum + count for avg.
        try:
            pipe = self._redis.pipeline()
            pipe.incrbyfloat(self._metric_key(key), value_ms)
            pipe.incrby(self._metric_key(key) + ":n", 1)
            pipe.execute()
        except Exception:
            pass

    def complete(self, *, model: str, prompt: str, request_id: Optional[str] = None, **kwargs: Any) -> Any:
        request_id = request_id or str(uuid.uuid4())

        # Lookup
        hit, lookup_ms = self._semantic_cache.lookup(
            prompt=prompt,
            model=model,
            embedding_model=self._config.embedding_model,
        )

        self._observe_ms("lookup_ms_sum", lookup_ms)
        self._incr(self._metric_key("lookup_ms_n"), 1)

        if hit is not None and hit.response is not None:
            self._incr(self._metric_key("hits"), 1)
            return hit.response

        self._incr(self._metric_key("misses"), 1)

        # Miss -> call provider
        start = time.perf_counter()
        response = self._provider.complete(model=model, prompt=prompt, request_id=request_id, **kwargs)
        _ = time.perf_counter() - start

        # Store
        try:
            self._semantic_cache.store(
                prompt=prompt,
                response=response,
                model=model,
                embedding_model=self._config.embedding_model,
                ttl_seconds=self._config.ttl_seconds,
            )
        except Exception as e:
            logger.warning("Semantic cache store failed: %s", e)

        return response


def get_semantic_cache_metrics(*, redis_cache: Optional[RedisCache] = None, metrics_prefix: str = "llm:semantic:metrics") -> Dict[str, Any]:
    rc = redis_cache or RedisCache()
    redis_client = rc.client

    def _get_int(k: str) -> int:
        try:
            v = redis_client.get(k)
            if v is None:
                return 0
            return int(v)
        except Exception:
            return 0

    hits = _get_int(f"{metrics_prefix}:hits")
    misses = _get_int(f"{metrics_prefix}:misses")

    # Lookup ms: sum+count
    try:
        sum_ms = float(redis_client.get(f"{metrics_prefix}:lookup_ms_sum") or 0.0)
        n = _get_int(f"{metrics_prefix}:lookup_ms_n")
    except Exception:
        sum_ms, n = 0.0, 0

    total = hits + misses
    hit_rate = (hits / total) if total else 0.0
    avg_lookup_ms = (sum_ms / n) if n else 0.0

    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
        "avg_lookup_ms": avg_lookup_ms,
        "total_lookups": total,
    }

