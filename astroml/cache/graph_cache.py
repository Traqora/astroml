"""Graph computation cache for repeated graph outputs — issue #767.

Caches intermediate graph outputs (adjacency lists, edge features, node
features) per data version and window to avoid recomputation across
experiments.  Supports both in-memory (default) and Redis backends.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class GraphCacheBackend(Enum):
    """Backend for graph computation cache."""

    MEMORY = "memory"
    REDIS = "redis"


@dataclass
class GraphCacheConfig:
    """Configuration for graph computation cache."""

    backend: GraphCacheBackend = GraphCacheBackend.MEMORY
    max_size: int = 512
    default_ttl_seconds: int = 3600  # 1 hour
    redis_url: str = "redis://localhost:6379"
    # Per-prefix TTL overrides (seconds)
    adjacency_ttl: int = 3600
    edge_feature_ttl: int = 1800
    node_feature_ttl: int = 1800
    snapshot_ttl: int = 3600


@dataclass
class GraphCacheStats:
    """Graph cache hit/miss statistics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }


class _MemoryGraphStore:
    """Thread-safe in-memory LRU cache for graph computations."""

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._data: dict[str, tuple[Any, float | None]] = {}  # key -> (value, expires_at)
        self._access_order: list[str] = []
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        import time

        with self._lock:
            if key not in self._data:
                return None
            value, expires_at = self._data[key]
            if expires_at is not None and time.time() > expires_at:
                del self._data[key]
                self._access_order.remove(key)
                return None
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        import time

        with self._lock:
            if key in self._data:
                self._access_order.remove(key)
            elif len(self._data) >= self._max_size:
                # Evict LRU
                oldest = self._access_order.pop(0)
                del self._data[oldest]

            expires_at = time.time() + ttl_seconds if ttl_seconds else None
            self._data[key] = (value, expires_at)
            self._access_order.append(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._access_order.remove(key)
                return True
            return False

    def clear(self, prefix: str = "") -> int:
        with self._lock:
            if not prefix:
                count = len(self._data)
                self._data.clear()
                self._access_order.clear()
                return count
            keys_to_remove = [
                k for k in self._data if k.startswith(prefix)
            ]
            for k in keys_to_remove:
                del self._data[k]
                self._access_order.remove(k)
            return len(keys_to_remove)

    def size(self) -> int:
        with self._lock:
            return len(self._data)


class GraphComputationCache:
    """Cache for graph computation results — adjacency lists, edge features,
    node features, and intermediate outputs keyed by data version and window.

    Usage::

        cache = GraphComputationCache()

        @cache.cached_adjacency(version="v3", window="7d")
        def build_adjacency(window_edges):
            ...

        adj = build_adjacency(edges)  # cached per (version, window, edges_hash)
    """

    _instance: GraphComputationCache | None = None

    def __new__(cls, config: GraphCacheConfig | None = None) -> GraphComputationCache:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: GraphCacheConfig | None = None) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.config = config or GraphCacheConfig()
        self._stats = GraphCacheStats()
        self._store: _MemoryGraphStore | None = None
        self._redis_client = None
        self._initialized = True

        if self.config.backend == GraphCacheBackend.MEMORY:
            self._store = _MemoryGraphStore(self.config.max_size)
        elif self.config.backend == GraphCacheBackend.REDIS:
            try:
                import redis

                self._redis_client = redis.from_url(self.config.redis_url)
                self._redis_client.ping()
            except Exception as e:
                logger.warning(
                    "Redis unavailable for graph cache, falling back to memory: %s", e
                )
                self._config.backend = GraphCacheBackend.MEMORY
                self._store = _MemoryGraphStore(self.config.max_size)

    @staticmethod
    def _hash_args(*args: Any, **kwargs: Any) -> str:
        """Generate a deterministic hash from function arguments."""
        parts: list[str] = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                parts.append(f"list:{len(arg)}")
            elif isinstance(arg, dict):
                parts.append(f"dict:{len(arg)}")
            else:
                parts.append(str(arg))
        for k, v in sorted(kwargs.items()):
            parts.append(f"{k}:{v}")
        combined = "|".join(parts)
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def get(self, prefix: str, key: str) -> Any | None:
        full_key = f"{prefix}:{key}"
        if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
            try:
                import pickle as _pickle

                data = self._redis_client.get(full_key)
                if data is not None:
                    self._stats.hits += 1
                    return _pickle.loads(data)
                self._stats.misses += 1
                return None
            except Exception as e:
                logger.warning("Redis graph cache GET error: %s", e)
                self._stats.misses += 1
                return None
        else:
            value = self._store.get(full_key)  # type: ignore[union-attr]
            if value is not None:
                self._stats.hits += 1
            else:
                self._stats.misses += 1
            return value

    def set(self, prefix: str, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        full_key = f"{prefix}:{key}"
        ttl = ttl_seconds or self.config.default_ttl_seconds
        if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
            try:
                import pickle as _pickle

                self._redis_client.setex(full_key, ttl, _pickle.dumps(value))
                self._stats.sets += 1
            except Exception as e:
                logger.warning("Redis graph cache SET error: %s", e)
        else:
            self._store.set(full_key, value, ttl)  # type: ignore[union-attr]
            self._stats.sets += 1

    def invalidate(self, prefix: str, key: str | None = None) -> int:
        if key:
            full_key = f"{prefix}:{key}"
            if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
                try:
                    return 1 if self._redis_client.delete(full_key) else 0
                except Exception:
                    return 0
            else:
                return 1 if self._store.delete(full_key) else 0  # type: ignore[union-attr]
        else:
            pattern = f"{prefix}:*"
            if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
                try:
                    keys = self._redis_client.keys(pattern)
                    if keys:
                        return self._redis_client.delete(*keys)
                    return 0
                except Exception:
                    return 0
            else:
                return self._store.clear(prefix)  # type: ignore[union-attr]

    def get_stats(self) -> GraphCacheStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = GraphCacheStats()

    # -- Convenience decorators -----------------------------------------------

    def cached_adjacency(
        self,
        version: str = "latest",
        window: str = "7d",
        ttl_seconds: int | None = None,
    ) -> Callable[[F], F]:
        """Cache adjacency list computation per data version and window."""

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                arg_hash = self._hash_args(*args, **kwargs)
                cache_key = f"adj:{version}:{window}:{arg_hash}"
                cached_value = self.get("graph:adjacency", cache_key)
                if cached_value is not None:
                    return cached_value
                result = func(*args, **kwargs)
                self.set(
                    "graph:adjacency",
                    cache_key,
                    result,
                    ttl_seconds or self.config.adjacency_ttl,
                )
                return result

            return wrapper  # type: ignore[return-value]

        return decorator

    def cached_edge_features(
        self,
        version: str = "latest",
        window: str = "7d",
        ttl_seconds: int | None = None,
    ) -> Callable[[F], F]:
        """Cache edge feature computation per data version and window."""

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                arg_hash = self._hash_args(*args, **kwargs)
                cache_key = f"ef:{version}:{window}:{arg_hash}"
                cached_value = self.get("graph:edge_features", cache_key)
                if cached_value is not None:
                    return cached_value
                result = func(*args, **kwargs)
                self.set(
                    "graph:edge_features",
                    cache_key,
                    result,
                    ttl_seconds or self.config.edge_feature_ttl,
                )
                return result

            return wrapper  # type: ignore[return-value]

        return decorator

    def cached_node_features(
        self,
        version: str = "latest",
        window: str = "7d",
        ttl_seconds: int | None = None,
    ) -> Callable[[F], F]:
        """Cache node feature computation per data version and window."""

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                arg_hash = self._hash_args(*args, **kwargs)
                cache_key = f"nf:{version}:{window}:{arg_hash}"
                cached_value = self.get("graph:node_features", cache_key)
                if cached_value is not None:
                    return cached_value
                result = func(*args, **kwargs)
                self.set(
                    "graph:node_features",
                    cache_key,
                    result,
                    ttl_seconds or self.config.node_feature_ttl,
                )
                return result

            return wrapper  # type: ignore[return-value]

        return decorator


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------

_graph_cache: GraphComputationCache | None = None
_graph_cache_lock = threading.Lock()


def get_graph_cache(config: GraphCacheConfig | None = None) -> GraphComputationCache:
    """Get or create the singleton graph computation cache."""
    global _graph_cache
    if _graph_cache is None:
        with _graph_cache_lock:
            if _graph_cache is None:
                _graph_cache = GraphComputationCache(config)
    return _graph_cache


def invalidate_graph_cache(prefix: str = "", key: str | None = None) -> int:
    """Invalidate graph cache entries.

    Args:
        prefix: Cache prefix (e.g. ``'graph:adjacency'``). Empty string clears all.
        key: Specific key within prefix. ``None`` clears all for the prefix.

    Returns:
        Number of entries invalidated.
    """
    cache = get_graph_cache()
    if prefix:
        return cache.invalidate(prefix, key)
    count = 0
    for p in ("graph:adjacency", "graph:edge_features", "graph:node_features"):
        count += cache.invalidate(p)
    return count
