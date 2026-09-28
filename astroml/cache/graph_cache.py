"""Graph computation cache for repeated graph outputs — issue #767.

Caches intermediate graph outputs (adjacency lists, edge features, node
features) so repeated experiments over the same slice of the ledger avoid
redundant reconstruction.

Two access styles are layered on a single cache instance:

Prefix/key API
    :meth:`GraphComputationCache.get`, :meth:`GraphComputationCache.set` and
    :meth:`GraphComputationCache.invalidate` store arbitrary values under a
    ``"<prefix>:<key>"`` name.  Entries are bounded by
    :attr:`GraphCacheConfig.max_size` and honour a per-entry TTL.  The backing
    store is the process-local :class:`_MemoryGraphStore` by default, or a Redis
    client when :attr:`GraphCacheConfig.backend` is
    :attr:`GraphCacheBackend.REDIS` (falling back to memory if Redis is
    unreachable).

Window API
    :meth:`GraphComputationCache.get_adjacency`,
    :meth:`GraphComputationCache.set_adjacency` and their ``*_edge_features``
    counterparts key entries by ``data_version`` plus window bounds through
    :func:`_window_key`, and short-circuit the shared
    :class:`~astroml.cache.redis_cache.RedisCache` behind a small in-process
    :class:`_LRUCache`.  Because the shared level is ``RedisCache``, the window
    API inherits that layer's connection pooling and TTL configuration, and
    degrades to LRU-only when Redis is unavailable.

Both styles feed the same :class:`GraphCacheStats` counters, and
:func:`get_graph_cache` / :func:`invalidate_graph_cache` expose the
process-wide default instance for callers that would rather not thread a cache
object through their call graph.

Example::

    cache = GraphComputationCache()

    adj = cache.get_adjacency("v1.2", start_ts=1_000_000, end_ts=1_010_000)
    if adj is None:
        adj = build_adjacency(edges, start_ts, end_ts)
        cache.set_adjacency("v1.2", 1_000_000, 1_010_000, adj)
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

from astroml.cache.redis_cache import CacheKeyPrefix, RedisCache

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_ADJACENCY_PREFIX = CacheKeyPrefix.GRAPH_WINDOW
_EDGE_FEATURE_PREFIX = CacheKeyPrefix.GRAPH_SNAPSHOT

# Default in-process LRU capacity for the window API (entry count, not bytes).
_DEFAULT_LRU_CAPACITY = 128

# Prefixes the graph decorators write under, used by the bulk purge helpers.
ADJACENCY_PREFIX = "graph:adjacency"
EDGE_FEATURE_PREFIX = "graph:edge_features"
NODE_FEATURE_PREFIX = "graph:node_features"


class GraphCacheBackend(Enum):
    """Backend backing the prefix/key API of :class:`GraphComputationCache`."""

    MEMORY = "memory"
    REDIS = "redis"


@dataclass
class GraphCacheConfig:
    """Configuration for the graph computation cache.

    Attributes:
        backend: In-process memory store (default) or Redis.
        max_size: Entry ceiling for the memory store before LRU eviction.
        default_ttl_seconds: TTL applied by :meth:`GraphComputationCache.set`
            when the caller does not pass one.
        redis_url: Redis connection URL, used when ``backend`` is Redis.
        adjacency_ttl: TTL for adjacency entries.
        edge_feature_ttl: TTL for edge feature entries.
        node_feature_ttl: TTL for node feature entries.
        snapshot_ttl: TTL for graph snapshot entries.
    """

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
        """Fraction of lookups that were hits (``0.0`` before any lookup)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the counters for logging or a metrics endpoint."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }


def _window_key(data_version: str, start_ts: int, end_ts: int, extra: str = "") -> str:
    """Build a stable cache key from window parameters.

    The same ``(data_version, start_ts, end_ts, extra)`` always yields the same
    digest, so two processes caching the same ledger slice agree on the key.
    """
    payload = f"{data_version}:{start_ts}:{end_ts}:{extra}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class _LRUCache:
    """Minimal thread-unsafe in-process LRU backed by an OrderedDict.

    The single-threaded reader/writer contract is deliberate: every access
    happens on the event loop that owns the owning
    :class:`GraphComputationCache`, so no lock is needed on the hot path.
    """

    def __init__(self, capacity: int = _DEFAULT_LRU_CAPACITY) -> None:
        self._cap = max(1, capacity)
        self._store: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0

    def get(self, key: str) -> Any:
        """Return the value for ``key``, or ``None`` on a miss."""
        if key not in self._store:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return self._store[key]

    def set(self, key: str, value: Any) -> None:
        """Store ``value``, evicting the least recently used entry if full."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        self.sets += 1
        if len(self._store) > self._cap:
            self._store.popitem(last=False)
            self.evictions += 1

    def invalidate(self, key: str) -> None:
        """Drop ``key`` if present."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Drop every entry."""
        self._store.clear()

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were hits (``0.0`` before any lookup)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the counters for logging or a metrics endpoint."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }

    def __len__(self) -> int:
        return len(self._store)


class _MemoryGraphStore:
    """Thread-safe in-memory LRU cache with per-entry TTL for graph computations."""

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._data: dict[str, tuple[Any, float | None]] = {}  # key -> (value, expires_at)
        self._access_order: list[str] = []
        self.evictions = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Return the live value for ``key``, or ``None`` if absent or expired.

        An expired entry is pruned as a side effect, so callers never observe
        a stale value after its TTL elapses.
        """
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
        """Store ``value`` under ``key`` with an optional TTL, evicting if full."""
        with self._lock:
            if key in self._data:
                self._access_order.remove(key)
            elif len(self._data) >= self._max_size:
                # Evict LRU
                oldest = self._access_order.pop(0)
                del self._data[oldest]
                self.evictions += 1

            expires_at = time.time() + ttl_seconds if ttl_seconds else None
            self._data[key] = (value, expires_at)
            self._access_order.append(key)

    def delete(self, key: str) -> bool:
        """Remove ``key``; return whether it was present."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._access_order.remove(key)
                return True
            return False

    def clear(self, prefix: str = "") -> int:
        """Remove every entry, or every entry starting with ``prefix``.

        Returns:
            Number of entries removed.
        """
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
        """Number of entries currently held."""
        with self._lock:
            return len(self._data)


class GraphComputationCache:
    """Cache for graph computation results — adjacency lists, edge features,
    node features, and intermediate outputs keyed by data version and window.

    Constructing with no arguments returns the process-wide shared instance
    (see :func:`get_graph_cache`); passing a :class:`GraphCacheConfig` or a
    non-default ``lru_capacity`` builds an isolated instance, which is what
    tests and multi-tenant callers want.

    Prefix/key usage::

        cache = GraphComputationCache()
        cache.set("graph:adjacency", "window_1", adjacency, ttl_seconds=600)
        assert cache.get("graph:adjacency", "window_1") == adjacency

    Decorator usage::

        @cache.cached_adjacency(version="v3", window="7d")
        def build_adjacency(window_edges):
            ...

        adj = build_adjacency(edges)  # cached per (version, window, edges_hash)
    """

    _instance: GraphComputationCache | None = None
    _instance_lock = threading.Lock()

    def __new__(
        cls,
        config: GraphCacheConfig | None = None,
        *,
        lru_capacity: int = _DEFAULT_LRU_CAPACITY,
    ) -> GraphComputationCache:
        """Return the shared instance for default construction, else a fresh one.

        Isolating explicitly-configured instances keeps ``max_size``/TTL tuning
        and cache-invalidation tests from sharing state with the process-wide
        cache that :func:`get_graph_cache` hands out.
        """
        if config is not None or lru_capacity != _DEFAULT_LRU_CAPACITY:
            return super().__new__(cls)
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        config: GraphCacheConfig | None = None,
        *,
        lru_capacity: int = _DEFAULT_LRU_CAPACITY,
    ) -> None:
        """Wire up the configured backend and the in-process LRU.

        Args:
            config: Backend/TTL configuration; defaults to :class:`GraphCacheConfig`.
            lru_capacity: Entry ceiling for the window API's in-process LRU.
        """
        if getattr(self, "_initialized", False):
            return
        self.config = config or GraphCacheConfig()
        self._stats = GraphCacheStats()
        self._store: _MemoryGraphStore | None = None
        self._redis_client: Any = None
        self._lru = _LRUCache(capacity=lru_capacity)
        self._redis: Any = None
        self._initialized = True

        if self.config.backend == GraphCacheBackend.MEMORY:
            self._store = _MemoryGraphStore(self.config.max_size)
        elif self.config.backend == GraphCacheBackend.REDIS:
            self._connect_redis()

    def _connect_redis(self) -> None:
        """Attach a Redis client, degrading to the memory store if it is unusable."""
        try:
            import redis

            self._redis_client = redis.from_url(self.config.redis_url)
            self._redis_client.ping()
        except Exception as e:
            logger.warning("Redis unavailable for graph cache, falling back to memory: %s", e)
            self.config.backend = GraphCacheBackend.MEMORY
            self._store = _MemoryGraphStore(self.config.max_size)

    @property
    def _uses_redis(self) -> bool:
        """Whether the prefix/key API is currently served by Redis."""
        return self.config.backend == GraphCacheBackend.REDIS and self._redis_client is not None

    def _memory_store(self) -> _MemoryGraphStore:
        """Return the memory store, creating it if the backend was downgraded."""
        if self._store is None:
            self._store = _MemoryGraphStore(self.config.max_size)
        return self._store

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

    # ------------------------------------------------------------------ #
    # Prefix/key API
    # ------------------------------------------------------------------ #

    def get(self, prefix: str, key: str) -> Any | None:
        """Return the value cached under ``prefix``/``key``, or ``None`` on a miss."""
        full_key = f"{prefix}:{key}"
        if self._uses_redis:
            try:
                data = self._redis_client.get(full_key)
                if data is not None:
                    self._stats.hits += 1
                    return self._deserialize(data)
                self._stats.misses += 1
                return None
            except Exception as e:
                logger.warning("Redis graph cache GET error for %s: %s", full_key, e)
                self._stats.misses += 1
                return None
        value = self._memory_store().get(full_key)
        if value is not None:
            self._stats.hits += 1
        else:
            self._stats.misses += 1
        return value

    def set(self, prefix: str, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Cache ``value`` under ``prefix``/``key``.

        Args:
            prefix: Namespace for the entry, e.g. ``"graph:adjacency"``.
            key: Entry key within the namespace.
            value: Any picklable value.
            ttl_seconds: Entry lifetime; defaults to
                :attr:`GraphCacheConfig.default_ttl_seconds`.
        """
        full_key = f"{prefix}:{key}"
        ttl = ttl_seconds or self.config.default_ttl_seconds
        if self._uses_redis:
            try:
                self._redis_client.setex(full_key, ttl, self._serialize(value))
                self._stats.sets += 1
            except Exception as e:
                logger.warning("Redis graph cache SET error for %s: %s", full_key, e)
            return
        store = self._memory_store()
        store.set(full_key, value, ttl)
        self._stats.sets += 1
        self._stats.evictions = store.evictions

    def invalidate(self, prefix: str, key: str | None = None) -> int:
        """Drop one key, or every key under ``prefix`` when ``key`` is ``None``.

        Returns:
            Number of entries removed.
        """
        if self._uses_redis:
            return self._invalidate_redis(f"{prefix}:{key}" if key else f"{prefix}:*")
        store = self._memory_store()
        if key is None:
            return store.clear(prefix)
        return 1 if store.delete(f"{prefix}:{key}") else 0

    def _invalidate_redis(self, pattern: str) -> int:
        """Delete ``pattern`` (``*``-suffixed) from Redis, logging any failure."""
        try:
            if pattern.endswith(":*"):
                keys = self._redis_client.keys(pattern)
                return self._redis_client.delete(*keys) if keys else 0
            return 1 if self._redis_client.delete(pattern) else 0
        except Exception:
            logger.warning("Redis graph cache INVALIDATE error for %s", pattern, exc_info=True)
            return 0

    def clear(self) -> int:
        """Clear every entry and reset statistics.

        Returns:
            Number of entries removed.
        """
        if self._uses_redis:
            try:
                keys = self._redis_client.keys("graph:*")
                count = len(keys)
                if keys:
                    self._redis_client.delete(*keys)
                self.reset_stats()
                return count
            except Exception:
                logger.warning("Redis graph cache CLEAR error", exc_info=True)
                self.reset_stats()
                return 0
        count = self._memory_store().clear("")
        self.reset_stats()
        return count

    def get_stats(self) -> GraphCacheStats:
        """Return the live hit/miss/set counters."""
        return self._stats

    def reset_stats(self) -> None:
        """Zero the hit/miss/set/eviction counters."""
        self._stats = GraphCacheStats()

    # ------------------------------------------------------------------ #
    # Window API (in-process LRU in front of the shared RedisCache)
    # ------------------------------------------------------------------ #

    def _redis_cache(self) -> Any:
        """Return the shared :class:`RedisCache`, creating it on first use.

        Construction is deferred so a memory-only deployment never pays for a
        Redis connection, and so tests can inject their own double via
        ``cache._redis``.
        """
        if self._redis is None:
            self._redis = RedisCache()
        return self._redis

    def _lru_redis_get(self, key: str, label: str) -> Any | None:
        """Read ``key`` from the LRU, falling back to the shared Redis layer."""
        hit = self._lru.get(key)
        if hit is not None:
            logger.debug("GraphComputationCache: %s LRU hit for %s", label, key[:12])
            return hit
        value = self._redis_cache().get(key)
        if value is not None:
            logger.debug("GraphComputationCache: %s Redis hit for %s", label, key[:12])
            self._lru.set(key, value)
        return value

    def _lru_redis_set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Write ``key`` to the LRU and to the shared Redis layer."""
        self._lru.set(key, value)
        self._redis_cache().set(key, value, ttl_seconds=ttl_seconds)

    def get_adjacency(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
    ) -> Any | None:
        """Return a cached adjacency structure or ``None`` on miss."""
        return self._lru_redis_get(self._adj_key(data_version, start_ts, end_ts), "adjacency")

    def set_adjacency(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        adjacency: Any,
    ) -> None:
        """Store an adjacency structure in both cache levels."""
        self._lru_redis_set(
            self._adj_key(data_version, start_ts, end_ts),
            adjacency,
            self.config.adjacency_ttl,
        )

    def invalidate_adjacency(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
    ) -> None:
        """Evict an adjacency entry from both cache levels."""
        self._drop(self._adj_key(data_version, start_ts, end_ts))

    def get_edge_features(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        feature_set: str = "default",
    ) -> Any | None:
        """Return cached edge features or ``None`` on miss."""
        key = self._ef_key(data_version, start_ts, end_ts, feature_set)
        return self._lru_redis_get(key, "edge_features")

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
        self._lru_redis_set(key, features, self.config.edge_feature_ttl)

    def invalidate_edge_features(
        self,
        data_version: str,
        start_ts: int,
        end_ts: int,
        feature_set: str = "default",
    ) -> None:
        """Evict edge features from both cache levels."""
        self._drop(self._ef_key(data_version, start_ts, end_ts, feature_set))

    def invalidate_version(self, data_version: str) -> None:
        """Evict in-process window entries (Redis entries expire via TTL)."""
        self._lru.clear()
        logger.info(
            "GraphComputationCache: LRU cleared on invalidate_version(%s)",
            data_version,
        )

    def _drop(self, key: str) -> None:
        """Remove ``key`` from the in-process LRU and the shared Redis layer."""
        self._lru.invalidate(key)
        self._redis_cache().delete(key)

    @property
    def lru_size(self) -> int:
        """Number of entries currently in the in-process LRU."""
        return len(self._lru)

    def lru_stats(self) -> dict[str, Any]:
        """Serialise the in-process LRU counters."""
        return self._lru.to_dict()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _serialize(value: Any) -> bytes:
        """Pickle ``value`` for transport over the Redis wire protocol."""
        import pickle as _pickle

        return _pickle.dumps(value)

    @staticmethod
    def _deserialize(data: Any) -> Any:
        """Unpickle a value read back from Redis, tolerating raw payloads."""
        import pickle as _pickle

        if isinstance(data, bytes):
            try:
                return _pickle.loads(data)
            except (_pickle.PickleError, TypeError, ValueError):
                return data
        return data

    def _adj_key(self, version: str, start: int, end: int) -> str:
        """Namespace the adjacency key for ``version`` and the window bounds."""
        digest = _window_key(version, start, end)
        return f"{_ADJACENCY_PREFIX.value}:adj:{digest}"

    def _ef_key(self, version: str, start: int, end: int, feature_set: str) -> str:
        """Namespace the edge-feature key for ``version``, window and feature set."""
        digest = _window_key(version, start, end, feature_set)
        return f"{_EDGE_FEATURE_PREFIX.value}:ef:{digest}"

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
                return self._cached_call(
                    ADJACENCY_PREFIX,
                    f"adj:{version}:{window}:{self._hash_args(*args, **kwargs)}",
                    ttl_seconds or self.config.adjacency_ttl,
                    func,
                    args,
                    kwargs,
                )

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
                return self._cached_call(
                    EDGE_FEATURE_PREFIX,
                    f"ef:{version}:{window}:{self._hash_args(*args, **kwargs)}",
                    ttl_seconds or self.config.edge_feature_ttl,
                    func,
                    args,
                    kwargs,
                )

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
                return self._cached_call(
                    NODE_FEATURE_PREFIX,
                    f"nf:{version}:{window}:{self._hash_args(*args, **kwargs)}",
                    ttl_seconds or self.config.node_feature_ttl,
                    func,
                    args,
                    kwargs,
                )

            return wrapper  # type: ignore[return-value]

        return decorator

    def _cached_call(
        self,
        prefix: str,
        key: str,
        ttl_seconds: int,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Return the cached result for ``prefix``/``key`` or compute and store it."""
        cached_value = self.get(prefix, key)
        if cached_value is not None:
            return cached_value
        result = func(*args, **kwargs)
        self.set(prefix, key, result, ttl_seconds=ttl_seconds)
        return result

    def reset(self) -> None:
        """Drop every entry and counter, in both the prefix and window APIs."""
        self.clear()
        self._lru.clear()
        self._redis = None


# ---------------------------------------------------------------------------
# Module-level accessors for the process-wide default instance
# ---------------------------------------------------------------------------


def get_graph_cache(config: GraphCacheConfig | None = None) -> GraphComputationCache:
    """Get or create the shared graph computation cache.

    Args:
        config: Optional configuration. Supplying one builds an isolated
            instance rather than mutating the shared one.
    """
    return GraphComputationCache(config)


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
    return cache.clear()


def cached_graph_computation(
    data_version_arg: str = "data_version",
    start_ts_arg: str = "start_ts",
    end_ts_arg: str = "end_ts",
    cache: GraphComputationCache | None = None,
    ttl_seconds: int = 1_800,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Cache a graph computation per data version and window.

    The decorated function is called with the same arguments on a miss; the
    ``data_version``/``start_ts``/``end_ts`` keyword arguments (names given by
    ``data_version_arg``/``start_ts_arg``/``end_ts_arg``) select the cache entry.
    They default to ``"unknown"``/``0``/``0`` when absent, so a function called
    without them shares a single entry.

    Args:
        data_version_arg: Keyword holding the data version.
        start_ts_arg: Keyword holding the window start.
        end_ts_arg: Keyword holding the window end.
        cache: Cache to store into; defaults to the shared instance.
        ttl_seconds: TTL for stored results.

    Example::

        @cached_graph_computation()
        def build_adjacency(data_version: str, start_ts: int, end_ts: int):
            ...  # expensive graph construction
    """
    _cache = cache or GraphComputationCache()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            version = kwargs.get(data_version_arg, "unknown")
            start = kwargs.get(start_ts_arg, 0)
            end = kwargs.get(end_ts_arg, 0)

            digest = _window_key(str(version), int(start), int(end), func.__name__)
            key = f"graph:computation:{digest}"

            value = _cache._lru.get(key)
            if value is not None:
                return value
            value = _cache._redis_cache().get(key)
            if value is not None:
                _cache._lru.set(key, value)
                return value

            result = func(*args, **kwargs)
            _cache._lru.set(key, result)
            _cache._redis_cache().set(key, result, ttl_seconds=ttl_seconds)
            return result

        return wrapper

    return decorator
