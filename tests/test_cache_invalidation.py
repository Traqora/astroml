"""Tests for cache invalidation in astroml/cache (issue #711).

Covers:
- TTL expiry on cached items
- Stale writes / overwriting keys with fresh values
- Explicit purge at key, prefix, and global levels
- Cache hits returning fresh values after recompute across both RedisCache and GraphComputationCache
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from astroml.cache.graph_cache import (
    GraphCacheConfig,
    GraphComputationCache,
    _MemoryGraphStore,
    get_graph_cache,
    invalidate_graph_cache,
)
from astroml.cache.redis_cache import (
    CacheConfig,
    CacheKeyPrefix,
    RedisCache,
    cached,
    clear_all_caches,
    invalidate_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache_singletons():
    """Ensure clean singleton state before and after each test."""
    GraphComputationCache._instance = None
    RedisCache._instance = None
    yield
    GraphComputationCache._instance = None
    RedisCache._instance = None


# ===========================================================================
# 1. GraphComputationCache: TTL Expiry Tests
# ===========================================================================


class TestGraphCacheTTLExpiry:
    """Test TTL expiration and lazy eviction in GraphComputationCache."""

    def test_memory_store_ttl_expiry(self):
        """Expired keys in _MemoryGraphStore return None and are pruned on access."""
        store = _MemoryGraphStore(max_size=10)
        store.set("key_temp", "val_temp", ttl_seconds=0.05)

        # Before expiry: key exists and is retrievable
        assert store.get("key_temp") == "val_temp"

        # Wait past TTL
        time.sleep(0.08)

        # After expiry: returns None and key is deleted from storage
        assert store.get("key_temp") is None
        assert "key_temp" not in store._data

    def test_graph_cache_ttl_expiry_and_recompute_fresh_value(self):
        """After TTL expires, the cache returns None, allowing recompute of fresh values."""
        cache = GraphComputationCache()
        prefix = "graph:test_ttl"
        key = "window_1"

        # Initial computation
        cache.set(prefix, key, {"version": 1, "data": "initial"}, ttl_seconds=0.05)
        assert cache.get(prefix, key) == {"version": 1, "data": "initial"}

        # Wait for TTL to expire
        time.sleep(0.08)

        # Stale entry has expired
        assert cache.get(prefix, key) is None

        # Recompute and store fresh value
        cache.set(prefix, key, {"version": 2, "data": "fresh"}, ttl_seconds=60)
        fresh = cache.get(prefix, key)
        assert fresh == {"version": 2, "data": "fresh"}

    def test_stats_track_miss_after_ttl_expiry(self):
        """Reading an expired item increments miss count in stats."""
        cache = GraphComputationCache()
        prefix = "graph:stats_ttl"

        cache.set(prefix, "k1", "v1", ttl_seconds=0.05)
        time.sleep(0.08)

        result = cache.get(prefix, "k1")
        assert result is None
        assert cache.get_stats().misses == 1


# ===========================================================================
# 2. GraphComputationCache: Stale Writes & Explicit Purge
# ===========================================================================


class TestGraphCacheInvalidationAndStaleWrites:
    """Test stale write overwrite, explicit key/prefix invalidation, and global purge."""

    def test_stale_write_overwrite_returns_fresh_value(self):
        """Writing a newer computation for the same key overwrites the stale value."""
        cache = GraphComputationCache()
        prefix = "graph:adjacency"
        key = "graph_snap_1"

        # Initial write
        cache.set(prefix, key, {"edges": 100})
        assert cache.get(prefix, key) == {"edges": 100}

        # Stale write update with newer graph snapshot
        cache.set(prefix, key, {"edges": 250})
        fresh = cache.get(prefix, key)
        assert fresh == {"edges": 250}

    def test_explicit_purge_specific_key(self):
        """Invalidating a specific key purges only that key and preserves siblings."""
        cache = GraphComputationCache()
        prefix = "graph:node_features"

        cache.set(prefix, "node_A", [1.0, 0.5])
        cache.set(prefix, "node_B", [0.2, 0.8])

        # Purge only node_A
        invalidated = cache.invalidate(prefix, "node_A")
        assert invalidated == 1
        assert cache.get(prefix, "node_A") is None
        assert cache.get(prefix, "node_B") == [0.2, 0.8]

    def test_explicit_purge_entire_prefix(self):
        """Invalidating a prefix purges all keys for that prefix and leaves others."""
        cache = GraphComputationCache()

        cache.set("graph:adjacency", "k1", {"matrix": "adj1"})
        cache.set("graph:adjacency", "k2", {"matrix": "adj2"})
        cache.set("graph:edge_features", "ef1", {"weights": [1, 2]})

        # Purge adjacency prefix
        invalidated = cache.invalidate("graph:adjacency")
        assert invalidated == 2
        assert cache.get("graph:adjacency", "k1") is None
        assert cache.get("graph:adjacency", "k2") is None
        assert cache.get("graph:edge_features", "ef1") == {"weights": [1, 2]}

    def test_explicit_global_purge(self):
        """cache.clear() empties all stored items and resets stats."""
        cache = GraphComputationCache()
        cache.set("p1", "k1", "val1")
        cache.set("p2", "k2", "val2")
        cache.get("p1", "k1")  # record a hit

        cache.clear()

        assert cache.get("p1", "k1") is None
        assert cache.get("p2", "k2") is None
        assert cache.get_stats().hits == 0
        assert cache.get_stats().sets == 0

    def test_invalidate_graph_cache_helper_purges_all_prefixes(self):
        """The top-level invalidate_graph_cache() utility clears all graph prefixes."""
        cache = get_graph_cache()
        cache.set("graph:adjacency", "k1", "adj_data")
        cache.set("graph:edge_features", "k2", "ef_data")
        cache.set("graph:node_features", "k3", "nf_data")

        purged = invalidate_graph_cache()
        assert purged == 3
        assert cache.get("graph:adjacency", "k1") is None
        assert cache.get("graph:edge_features", "k2") is None
        assert cache.get("graph:node_features", "k3") is None

    def test_lru_eviction_when_capacity_exceeded(self):
        """Adding items beyond max_size evicts the least recently used item."""
        config = GraphCacheConfig(max_size=2)
        cache = GraphComputationCache(config)

        cache.set("lru", "k1", "v1")
        cache.set("lru", "k2", "v2")

        # Access k1 to make k2 the least recently used
        assert cache.get("lru", "k1") == "v1"

        # Insert k3, exceeding capacity (2)
        cache.set("lru", "k3", "v3")

        # k2 should be evicted; k1 and k3 remain
        assert cache.get("lru", "k2") is None
        assert cache.get("lru", "k1") == "v1"
        assert cache.get("lru", "k3") == "v3"


# ===========================================================================
# 3. Decorator Fresh Values After Invalidation
# ===========================================================================


class TestDecoratorRecomputeAfterInvalidation:
    """Test that decorated functions recompute and return fresh values when cache is cleared."""

    def test_cached_adjacency_recomputes_after_invalidation(self):
        """cached_adjacency invokes underlying function again after invalidation."""
        cache = GraphComputationCache()
        call_counter = {"count": 0, "current_weight": 1.0}

        @cache.cached_adjacency(version="v1", window="24h", ttl_seconds=3600)
        def compute_adj(edges: list[tuple[str, str]]):
            call_counter["count"] += 1
            return {"edges": len(edges), "weight": call_counter["current_weight"]}

        edges = [("A", "B"), ("B", "C")]

        # First call: computes value
        res1 = compute_adj(edges)
        assert res1["weight"] == 1.0
        assert call_counter["count"] == 1

        # Second call: cache hit, no new computation
        res2 = compute_adj(edges)
        assert res2["weight"] == 1.0
        assert call_counter["count"] == 1

        # Invalidate cache for adjacency prefix
        cache.invalidate("graph:adjacency")

        # Update underlying data state
        call_counter["current_weight"] = 5.0

        # Third call: recomputed and returns fresh value
        res3 = compute_adj(edges)
        assert res3["weight"] == 5.0
        assert call_counter["count"] == 2


# ===========================================================================
# 4. RedisCache: Invalidation, Purge & Fresh Values (Mocked Backend)
# ===========================================================================


class TestRedisCacheInvalidation:
    """Test RedisCache explicit invalidation, pattern deletion, and purge with mocked Redis."""

    def _setup_mock_redis(self):
        """Create a RedisCache instance backed by an in-memory dictionary mock."""
        mock_client = MagicMock()
        storage: dict[str, Any] = {}

        def mock_get(k):
            return storage.get(k)

        def mock_set(k, v):
            storage[k] = v
            return True

        def mock_setex(k, ttl, v):
            storage[k] = v
            return True

        def mock_delete(*keys):
            deleted = 0
            for k in keys:
                if k in storage:
                    del storage[k]
                    deleted += 1
            return deleted

        def mock_keys(pattern):
            import fnmatch

            return [k for k in storage if fnmatch.fnmatch(k, pattern)]

        def mock_flushdb():
            storage.clear()
            return True

        def mock_exists(k):
            return 1 if k in storage else 0

        mock_client.get.side_effect = mock_get
        mock_client.set.side_effect = mock_set
        mock_client.setex.side_effect = mock_setex
        mock_client.delete.side_effect = mock_delete
        mock_client.keys.side_effect = mock_keys
        mock_client.flushdb.side_effect = mock_flushdb
        mock_client.exists.side_effect = mock_exists

        with patch.object(RedisCache, "_connect"):
            cache = RedisCache()
            cache._client = mock_client
            return cache, storage

    def test_rediscache_delete_specific_key(self):
        """cache.delete() removes the specified key."""
        cache, storage = self._setup_mock_redis()

        cache.set("feature:acc_1", {"balance": 100})
        cache.set("feature:acc_2", {"balance": 200})

        assert cache.delete("feature:acc_1") is True
        assert cache.get("feature:acc_1") is None
        assert cache.get("feature:acc_2") == {"balance": 200}

    def test_rediscache_delete_pattern_purge(self):
        """cache.delete_pattern() removes all matching keys."""
        cache, storage = self._setup_mock_redis()

        cache.set("feature:user:1", "data1")
        cache.set("feature:user:2", "data2")
        cache.set("prediction:user:1", "pred1")

        deleted_count = cache.delete_pattern("feature:*")
        assert deleted_count == 2
        assert cache.get("feature:user:1") is None
        assert cache.get("feature:user:2") is None
        assert cache.get("prediction:user:1") == "pred1"

    def test_invalidate_cache_helper_by_prefix_and_key(self):
        """invalidate_cache utility removes specific key or entire prefix namespace."""
        cache, storage = self._setup_mock_redis()

        with patch("astroml.cache.redis_cache.RedisCache", return_value=cache):
            cache.set(f"{CacheKeyPrefix.FEATURE.value}:tx_1", "feat1")
            cache.set(f"{CacheKeyPrefix.FEATURE.value}:tx_2", "feat2")

            # Invalidate specific key
            count = invalidate_cache(CacheKeyPrefix.FEATURE, key="tx_1")
            assert count == 1
            assert cache.get(f"{CacheKeyPrefix.FEATURE.value}:tx_1") is None
            assert cache.get(f"{CacheKeyPrefix.FEATURE.value}:tx_2") == "feat2"

            # Invalidate entire prefix
            count_all = invalidate_cache(CacheKeyPrefix.FEATURE)
            assert count_all == 1
            assert cache.get(f"{CacheKeyPrefix.FEATURE.value}:tx_2") is None

    def test_clear_all_caches_helper(self):
        """clear_all_caches utility purges the entire Redis cache."""
        cache, storage = self._setup_mock_redis()

        with patch("astroml.cache.redis_cache.RedisCache", return_value=cache):
            cache.set("k1", "v1")
            cache.set("k2", "v2")

            assert clear_all_caches() is True
            assert len(storage) == 0
            assert cache.get("k1") is None

    def test_cached_decorator_returns_fresh_value_after_invalidation(self):
        """cached decorator function recomputes and returns fresh values after cache invalidation."""
        cache, storage = self._setup_mock_redis()
        eval_count = {"n": 0, "prediction": 0.12}

        with patch("astroml.cache.redis_cache.RedisCache", return_value=cache):

            @cached(CacheKeyPrefix.PREDICTION, ttl_seconds=600)
            def predict_score(account_id: str) -> float:
                eval_count["n"] += 1
                return eval_count["prediction"]

            # First computation: computes value
            first = predict_score("ACC100")
            assert first == 0.12
            assert eval_count["n"] == 1

            # Second call: cache hit, no computation
            second = predict_score("ACC100")
            assert second == 0.12
            assert eval_count["n"] == 1

            # Invalidate prediction cache
            invalidate_cache(CacheKeyPrefix.PREDICTION)

            # Update model output
            eval_count["prediction"] = 0.99

            # Third call: recomputed fresh result
            third = predict_score("ACC100")
            assert third == 0.99
            assert eval_count["n"] == 2
