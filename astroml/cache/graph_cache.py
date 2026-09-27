"""Graph computation cache for repeated graph outputs — issue #767.

Caches intermediate graph outputs (adjacency lists, edge features, node
features) per data version and window to avoid recomputation across
experiments.  Supports both in-memory (default) and Redis backends.

A small per-process LRU (``_LRUCache``) sits in front of the configured
backend so the most recently accessed windows short-circuit the store (and
Redis, when present) entirely within a single process.  LRU entries inherit
the TTL of the value they mirror so stale windows expire lazily.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Default in-process LRU capacity (number of entries, not bytes).
_DEFAULT_LRU_CAPACITY = 128


def _window_key(data_version: str, start_ts: int, end_ts: int, extra: str = "") -> str:
    """Stable cache key from window parameters."""
    payload = f"{data_version}:{start_ts}:{end_ts}:{extra}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return digest


class _LRUCache:
    """Minimal in-process LRU backed by an OrderedDict, with optional TTLs."""

    def __init__(self, capacity: int = _DEFAULT_LRU_CAPACITY) -> None:
        self._cap = max(1, capacity)
        self._store: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()

    def get(self, key: str) -> Any:
        import time

        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() > expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        import time

        if key in self._store:
            self._store.move_to_end(key)
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        self._store[key] = (value, expires_at)
        if len(self._store) > self._cap:
            self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)


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

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
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
            keys_to_remove = [k for k in self._data if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._data[k]
                self._access_order.remove(k)
            return len(keys_to_remove)

    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def touch(self, key: str) -> None:
        """Promote ``key`` to most-recently-used if present.

        Keeps layered eviction coherent: when the front LRU serves a hit, the
        backing store must observe the same access or the two layers can
        evict different victims and resurrect evicted values.
        """
        with self._lock:
            if key in self._data:
                self._access_order.remove(key)
                self._access_order.append(key)


class GraphComputationCache:
    """Cache for graph computation results — adjacency lists, edge features,
    node features, and intermediate outputs keyed by data version and window.

    Reads are served from the in-process LRU first, then the configured
    backend (memory store or Redis), then — best-effort — any raw Redis
    client attached at ``self._redis`` (used by tests and by callers that
    bring their own client).

    Usage::

        cache = GraphComputationCache()

        @cache.cached_adjacency(version="v3", window="7d")
        def build_adjacency(window_edges):
            ...

        adj = build_adjacency(edges)  # cached per (version, window, edges_hash)
    """

    _instance: GraphComputationCache | None = None

    def __new__(
        cls, config: GraphCacheConfig | None = None, **kwargs: Any
    ) -> GraphComputationCache:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        config: GraphCacheConfig | None = None,
        lru_capacity: int | None = None,
    ) -> None:
        # Plain re-instantiation (no explicit arguments) reuses the existing
        # singleton state; passing an explicit config or LRU capacity rebuilds
        # every layer so callers always get a freshly configured cache.
        reconfigure = config is not None or lru_capacity is not None
        if getattr(self, "_initialized", False) and not reconfigure:
            return
        self.config = config or GraphCacheConfig()
        self._stats = GraphCacheStats()
        self._store: _MemoryGraphStore | None = None
        self._redis_client: Any = None
        self._redis: Any = None  # optional raw client (tests / caller-supplied)
        self._initialized = True

        if self.config.backend == GraphCacheBackend.MEMORY:
            self._store = _MemoryGraphStore(self.config.max_size)
        elif self.config.backend == GraphCacheBackend.REDIS:
            try:
                import redis

                self._redis_client = redis.from_url(self.config.redis_url)
                self._redis_client.ping()
            except Exception as e:
                logger.warning("Redis unavailable for graph cache, falling back to memory: %s", e)
                self.config.backend = GraphCacheBackend.MEMORY
                self._store = _MemoryGraphStore(self.config.max_size)

        self._lru = _LRUCache(
            capacity=lru_capacity if lru_capacity is not None else self.config.max_size
        )

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
        lru_value = self._lru.get(full_key)
        if lru_value is not None:
            self._stats.hits += 1
            if self._store is not None:
                self._store.touch(full_key)
            return lru_value

        if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
            try:
                import pickle as _pickle

                data = self._redis_client.get(full_key)
                if data is not None:
                    value = _pickle.loads(data)
                    self._stats.hits += 1
                    self._lru.set(full_key, value)
                    return value
                self._stats.misses += 1
                return None
            except Exception as e:
                logger.warning("Redis graph cache GET error: %s", e)
                self._stats.misses += 1
                return None

        value = self._store.get(full_key) if self._store is not None else None
        if value is not None:
            self._stats.hits += 1
            self._lru.set(full_key, value)
            return value

        if self._redis is not None:
            try:
                raw = self._redis.get(full_key)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Raw Redis graph cache GET error: %s", e)
                raw = None
            if raw is not None:
                self._stats.hits += 1
                self._lru.set(full_key, raw)
                return raw

        self._stats.misses += 1
        return None

    def set(self, prefix: str, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        full_key = f"{prefix}:{key}"
        ttl = ttl_seconds if ttl_seconds is not None else self.config.default_ttl_seconds

        if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
            try:
                import pickle as _pickle

                self._redis_client.setex(full_key, int(ttl), _pickle.dumps(value))
                self._stats.sets += 1
                self._lru.set(full_key, value, ttl_seconds=ttl)
                return
            except Exception as e:
                logger.warning("Redis graph cache SET error: %s", e)

        if self._store is not None:
            self._store.set(full_key, value, ttl)
            self._stats.sets += 1
            self._lru.set(full_key, value, ttl_seconds=ttl)

        if self._redis is not None:
            try:
                if ttl_seconds is not None:
                    self._redis.set(full_key, value, ttl=ttl_seconds)
                else:
                    self._redis.set(full_key, value)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Raw Redis graph cache SET error: %s", e)

    def invalidate(self, prefix: str, key: str | None = None) -> int:
        if key:
            full_key = f"{prefix}:{key}"
            if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
                try:
                    deleted = 1 if self._redis_client.delete(full_key) else 0
                except Exception:
                    deleted = 0
            else:
                deleted = 1 if (self._store is not None and self._store.delete(full_key)) else 0
            self._lru.invalidate(full_key)
            return deleted

        pattern = f"{prefix}:"
        if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
            try:
                keys = self._redis_client.keys(f"{prefix}:*")
                deleted = self._redis_client.delete(*keys) if keys else 0
            except Exception:
                deleted = 0
        else:
            deleted = self._store.clear(prefix) if self._store is not None else 0
        for lru_key in self._lru.keys():
            if lru_key.startswith(pattern):
                self._lru.invalidate(lru_key)
        return deleted

    def clear(self) -> None:
        """Purge every layer (LRU, backend store, Redis when configured)."""
        if self._store is not None:
            self._store.clear()
        self._lru.clear()
        self._stats = GraphCacheStats()
        if self.config.backend == GraphCacheBackend.REDIS and self._redis_client:
            try:
                keys = self._redis_client.keys("graph:*")
                if keys:
                    self._redis_client.delete(*keys)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Redis graph cache CLEAR error: %s", e)

    def get_stats(self) -> GraphCacheStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = GraphCacheStats()

    # -- Convenience accessors ------------------------------------------------

    def get_adjacency(self, data_version: str, start_ts: int, end_ts: int) -> Any | None:
        return self.get("graph:adjacency", _window_key(data_version, start_ts, end_ts))

    def set_adjacency(self, data_version: str, start_ts: int, end_ts: int, value: Any) -> None:
        self.set(
            "graph:adjacency",
            _window_key(data_version, start_ts, end_ts),
            value,
            ttl_seconds=self.config.adjacency_ttl,
        )

    def invalidate_adjacency(self, data_version: str, start_ts: int, end_ts: int) -> None:
        self.invalidate("graph:adjacency", _window_key(data_version, start_ts, end_ts))

    def get_edge_features(
        self, data_version: str, start_ts: int, end_ts: int, feature_set: str = ""
    ) -> Any | None:
        key = _window_key(data_version, start_ts, end_ts, extra=feature_set)
        return self.get("graph:edge_features", key)

    def set_edge_features(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        value: Any,
        feature_set: str = "",
    ) -> None:
        key = _window_key(data_version, start_ts, end_ts, extra=feature_set)
        self.set(
            "graph:edge_features",
            key,
            value,
            ttl_seconds=self.config.edge_feature_ttl,
        )

    def get_node_features(
        self, data_version: str, start_ts: int, end_ts: int, feature_set: str = ""
    ) -> Any | None:
        key = _window_key(data_version, start_ts, end_ts, extra=feature_set)
        return self.get("graph:node_features", key)

    def set_node_features(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        value: Any,
        feature_set: str = "",
    ) -> None:
        key = _window_key(data_version, start_ts, end_ts, extra=feature_set)
        self.set(
            "graph:node_features",
            key,
            value,
            ttl_seconds=self.config.node_feature_ttl,
        )

    def invalidate_version(self, data_version: str) -> None:
        """Evict all locally cached entries (Redis entries expire naturally via TTL)."""
        self._lru.clear()
        if self._store is not None:
            self._store.clear()
        logger.info(
            "GraphComputationCache: local layers cleared on invalidate_version(%s)", data_version
        )

    @property
    def lru_size(self) -> int:
        return len(self._lru)

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

_graph_cache_lock = threading.Lock()


def get_graph_cache(config: GraphCacheConfig | None = None) -> GraphComputationCache:
    """Get or create the singleton graph computation cache.

    The class-level ``GraphComputationCache._instance`` is the single source
    of truth; resetting it (as tests do) makes the next call construct a
    fresh cache.
    """
    if GraphComputationCache._instance is not None:
        return GraphComputationCache._instance
    with _graph_cache_lock:
        if GraphComputationCache._instance is None:
            GraphComputationCache._instance = GraphComputationCache(config)
    return GraphComputationCache._instance


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


def cached_graph_computation(
    data_version_arg: str = "data_version",
    start_ts_arg: str = "start_ts",
    end_ts_arg: str = "end_ts",
    cache: GraphComputationCache | None = None,
    ttl_seconds: int = 1_800,
) -> Callable[[F], F]:
    """Decorator that caches graph computation outputs per data version and window.

    The decorated function must accept ``data_version``, ``start_ts``, and
    ``end_ts`` keyword arguments (or positional args whose names match
    ``data_version_arg``, ``start_ts_arg``, ``end_ts_arg``).

    Example::

        @cached_graph_computation()
        def build_adjacency(data_version: str, start_ts: int, end_ts: int):
            ...  # expensive graph construction
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            target = cache if cache is not None else get_graph_cache()
            version = str(kwargs.get(data_version_arg, "unknown"))
            start = int(kwargs.get(start_ts_arg, 0))
            end = int(kwargs.get(end_ts_arg, 0))

            key = _window_key(version, start, end, func.__name__)
            full_key = f"graph:computation:{key}"

            cached_value = target._lru.get(full_key)
            if cached_value is not None:
                return cached_value

            if target._redis is not None:
                try:
                    raw = target._redis.get(full_key)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("Raw Redis graph cache GET error: %s", e)
                    raw = None
                if raw is not None:
                    target._lru.set(full_key, raw, ttl_seconds=ttl_seconds)
                    return raw

            result = func(*args, **kwargs)
            target._lru.set(full_key, result, ttl_seconds=ttl_seconds)
            if target._redis is not None:
                try:
                    target._redis.set(full_key, result, ttl=ttl_seconds)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("Raw Redis graph cache SET error: %s", e)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
