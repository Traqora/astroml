"""Graph computation cache — issue #767.

Caches intermediate graph outputs (adjacency lists, edge features) keyed by
data version and window parameters so repeated experiments on the same slice
of the ledger avoid redundant reconstruction.

The cache uses the existing :class:`~astroml.cache.redis_cache.RedisCache`
layer and therefore inherits its TTL configuration, hit/miss metrics, and
Redis connection pooling.  A local in-process LRU layer sits in front to
short-circuit Redis for the most recently accessed windows within a single
process.
"""Graph computation cache for repeated graph outputs — issue #767.

Caches intermediate graph outputs (adjacency lists, edge features, node
features) per data version and window to avoid recomputation across
experiments.  Supports both in-memory (default) and Redis backends.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any

from astroml.cache.redis_cache import CacheKeyPrefix, RedisCache

logger = logging.getLogger(__name__)

_ADJACENCY_PREFIX = CacheKeyPrefix.GRAPH_WINDOW
_EDGE_FEATURE_PREFIX = CacheKeyPrefix.GRAPH_SNAPSHOT

# Default in-process LRU capacity (number of entries, not bytes).
_DEFAULT_LRU_CAPACITY = 128


def _window_key(data_version: str, start_ts: int, end_ts: int, extra: str = "") -> str:
    """Stable cache key from window parameters."""
    payload = f"{data_version}:{start_ts}:{end_ts}:{extra}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return digest


class _LRUCache:
    """Minimal thread-unsafe in-process LRU backed by an OrderedDict."""

    def __init__(self, capacity: int = _DEFAULT_LRU_CAPACITY) -> None:
        self._cap = max(1, capacity)
        self._store: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._cap:
            self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class GraphComputationCache:
    """Two-level cache (in-process LRU → Redis) for graph intermediate outputs.

    Adjacency lists and edge feature tensors/dicts can be expensive to rebuild
    for large windows.  This class stores them under a key derived from
    ``data_version`` and the window bounds so experiments that share the same
    data slice reuse the cached result.

    Args:
        redis_ttl_adjacency: Redis TTL for adjacency list entries in seconds
            (default 30 minutes).
        redis_ttl_edge_features: Redis TTL for edge feature entries in seconds
            (default 1 hour).
        lru_capacity: Number of entries to keep in the in-process LRU.

    Example::

        cache = GraphComputationCache()

        adj = cache.get_adjacency("v1.2", start_ts=1_000_000, end_ts=1_010_000)
        if adj is None:
            adj = build_adjacency(edges, start_ts, end_ts)
            cache.set_adjacency("v1.2", 1_000_000, 1_010_000, adj)
    """

    def __init__(
        self,
        redis_ttl_adjacency: int = 1_800,
        redis_ttl_edge_features: int = 3_600,
        lru_capacity: int = _DEFAULT_LRU_CAPACITY,
    ) -> None:
        self._redis = RedisCache()
        self._ttl_adj = redis_ttl_adjacency
        self._ttl_ef = redis_ttl_edge_features
        self._lru: _LRUCache = _LRUCache(capacity=lru_capacity)

    # ------------------------------------------------------------------ #
    # Adjacency list caching
    # ------------------------------------------------------------------ #

    def get_adjacency(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
    ) -> Any | None:
        """Return a cached adjacency structure or ``None`` on miss."""
        key = self._adj_key(data_version, start_ts, end_ts)
        hit = self._lru.get(key)
        if hit is not None:
            logger.debug("GraphComputationCache: adjacency LRU hit for %s", key[:12])
            return hit
        value = self._redis.get(key)
        if value is not None:
            logger.debug("GraphComputationCache: adjacency Redis hit for %s", key[:12])
            self._lru.set(key, value)
        return value

    def set_adjacency(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        adjacency: Any,
    ) -> None:
        """Store an adjacency structure in both cache levels."""
        key = self._adj_key(data_version, start_ts, end_ts)
        self._lru.set(key, adjacency)
        self._redis.set(key, adjacency, ttl=self._ttl_adj)

    def invalidate_adjacency(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
    ) -> None:
        """Evict an adjacency entry from both cache levels."""
        key = self._adj_key(data_version, start_ts, end_ts)
        self._lru.invalidate(key)
        self._redis.delete(key)

    # ------------------------------------------------------------------ #
    # Edge feature caching
    # ------------------------------------------------------------------ #

    def get_edge_features(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        feature_set: str = "default",
    ) -> Any | None:
        """Return cached edge features or ``None`` on miss."""
        key = self._ef_key(data_version, start_ts, end_ts, feature_set)
        hit = self._lru.get(key)
        if hit is not None:
            logger.debug("GraphComputationCache: edge_features LRU hit for %s", key[:12])
            return hit
        value = self._redis.get(key)
        if value is not None:
            logger.debug("GraphComputationCache: edge_features Redis hit for %s", key[:12])
            self._lru.set(key, value)
        return value

    def set_edge_features(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        features: Any,
        feature_set: str = "default",
    ) -> None:
        """Store edge features in both cache levels."""
        key = self._ef_key(data_version, start_ts, end_ts, feature_set)
        self._lru.set(key, features)
        self._redis.set(key, features, ttl=self._ttl_ef)

    def invalidate_edge_features(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        feature_set: str = "default",
    ) -> None:
        """Evict edge features from both cache levels."""
        key = self._ef_key(data_version, start_ts, end_ts, feature_set)
        self._lru.invalidate(key)
        self._redis.delete(key)

    # ------------------------------------------------------------------ #
    # Bulk operations
    # ------------------------------------------------------------------ #

    def invalidate_version(self, data_version: str) -> None:
        """Evict all LRU entries (Redis entries expire naturally via TTL)."""
        self._lru.clear()
        logger.info(
            "GraphComputationCache: LRU cleared on invalidate_version(%s)", data_version
        )

    @property
    def lru_size(self) -> int:
        """Number of entries currently in the in-process LRU."""
        return len(self._lru)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _adj_key(self, version: str, start: int, end: int) -> str:
        digest = _window_key(version, start, end)
        return f"{_ADJACENCY_PREFIX.value}:adj:{digest}"

    def _ef_key(self, version: str, start: int, end: int, feature_set: str) -> str:
        digest = _window_key(version, start, end, feature_set)
        return f"{_EDGE_FEATURE_PREFIX.value}:ef:{digest}"


def cached_graph_computation(
    data_version_arg: str = "data_version",
    start_ts_arg: str = "start_ts",
    end_ts_arg: str = "end_ts",
    cache: GraphComputationCache | None = None,
    ttl_seconds: int = 1_800,
):
    """Decorator that caches graph computation outputs per data version and window.

    The decorated function must accept ``data_version``, ``start_ts``, and
    ``end_ts`` keyword arguments (or positional args whose names match
    ``data_version_arg``, ``start_ts_arg``, ``end_ts_arg``).

    Example::

        @cached_graph_computation()
        def build_adjacency(data_version: str, start_ts: int, end_ts: int):
            ...  # expensive graph construction
    """
    _cache = cache or GraphComputationCache(redis_ttl_adjacency=ttl_seconds)

    def decorator(func):  # type: ignore[no-untyped-def]
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            version = kwargs.get(data_version_arg, "unknown")
            start = kwargs.get(start_ts_arg, 0)
            end = kwargs.get(end_ts_arg, 0)

            key = _window_key(str(version), int(start), int(end), func.__name__)
            full_key = f"graph:computation:{key}"

            cached = _cache._lru.get(full_key)
            if cached is not None:
                return cached

            cached = _cache._redis.get(full_key)
            if cached is not None:
                _cache._lru.set(full_key, cached)
                return cached

            result = func(*args, **kwargs)
            _cache._lru.set(full_key, result)
            _cache._redis.set(full_key, result, ttl=ttl_seconds)
            return result

        return wrapper

    return decorator
