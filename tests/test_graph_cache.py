"""Tests for GraphComputationCache — issue #767."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from astroml.cache.graph_cache import (
    GraphComputationCache,
    _LRUCache,
    _window_key,
    cached_graph_computation,
)


# ---------------------------------------------------------------------------
# _window_key
# ---------------------------------------------------------------------------

class TestWindowKey:
    def test_deterministic(self):
        k1 = _window_key("v1", 1000, 2000)
        k2 = _window_key("v1", 1000, 2000)
        assert k1 == k2

    def test_different_versions_produce_different_keys(self):
        assert _window_key("v1", 0, 100) != _window_key("v2", 0, 100)

    def test_different_windows_produce_different_keys(self):
        assert _window_key("v1", 0, 100) != _window_key("v1", 0, 200)


# ---------------------------------------------------------------------------
# _LRUCache
# ---------------------------------------------------------------------------

class TestLRUCache:
    def test_get_miss_returns_none(self):
        lru = _LRUCache(capacity=4)
        assert lru.get("missing") is None

    def test_set_and_get(self):
        lru = _LRUCache(capacity=4)
        lru.set("k", {"adj": [1, 2]})
        assert lru.get("k") == {"adj": [1, 2]}

    def test_capacity_evicts_oldest(self):
        lru = _LRUCache(capacity=2)
        lru.set("a", 1)
        lru.set("b", 2)
        lru.set("c", 3)  # evicts "a"
        assert lru.get("a") is None
        assert lru.get("b") == 2
        assert lru.get("c") == 3

    def test_access_promotes_to_recent(self):
        lru = _LRUCache(capacity=2)
        lru.set("a", 1)
        lru.set("b", 2)
        lru.get("a")       # promote "a"
        lru.set("c", 3)    # should evict "b", not "a"
        assert lru.get("a") == 1
        assert lru.get("b") is None

    def test_invalidate(self):
        lru = _LRUCache(capacity=4)
        lru.set("k", 99)
        lru.invalidate("k")
        assert lru.get("k") is None

    def test_clear(self):
        lru = _LRUCache(capacity=4)
        lru.set("a", 1)
        lru.set("b", 2)
        lru.clear()
        assert len(lru) == 0


# ---------------------------------------------------------------------------
# GraphComputationCache
# ---------------------------------------------------------------------------

def _make_cache_no_redis():
    """Return a GraphComputationCache whose Redis client is fully mocked."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None  # always a miss
    mock_redis.set.return_value = None
    mock_redis.delete.return_value = None

    cache = GraphComputationCache(lru_capacity=8)
    cache._redis = mock_redis
    return cache


class TestGraphComputationCacheAdjacency:
    def test_miss_returns_none(self):
        cache = _make_cache_no_redis()
        result = cache.get_adjacency("v1", 1000, 2000)
        assert result is None

    def test_set_then_get_from_lru(self):
        cache = _make_cache_no_redis()
        adj = {"0x1": ["0x2", "0x3"]}
        cache.set_adjacency("v1", 1000, 2000, adj)
        result = cache.get_adjacency("v1", 1000, 2000)
        assert result == adj
        # LRU hit: Redis.get should NOT be called again after the set
        assert cache._redis.get.call_count == 0

    def test_lru_miss_falls_through_to_redis(self):
        cache = _make_cache_no_redis()
        redis_adj = {"0xA": ["0xB"]}
        cache._redis.get.return_value = redis_adj

        result = cache.get_adjacency("v1", 1000, 2000)
        assert result == redis_adj
        cache._redis.get.assert_called_once()

    def test_invalidate_removes_from_lru(self):
        cache = _make_cache_no_redis()
        cache.set_adjacency("v1", 0, 100, {"data": True})
        cache.invalidate_adjacency("v1", 0, 100)
        # After invalidation LRU is empty; cache falls through to (mocked) Redis
        cache._redis.get.return_value = None
        assert cache.get_adjacency("v1", 0, 100) is None

    def test_different_versions_isolated(self):
        cache = _make_cache_no_redis()
        cache.set_adjacency("v1", 0, 100, "v1-data")
        cache.set_adjacency("v2", 0, 100, "v2-data")
        assert cache.get_adjacency("v1", 0, 100) == "v1-data"
        assert cache.get_adjacency("v2", 0, 100) == "v2-data"


class TestGraphComputationCacheEdgeFeatures:
    def test_miss_returns_none(self):
        cache = _make_cache_no_redis()
        assert cache.get_edge_features("v1", 0, 100) is None

    def test_set_then_get(self):
        cache = _make_cache_no_redis()
        features = {"amount": [1.0, 2.0], "type": ["xlm", "xlm"]}
        cache.set_edge_features("v1", 0, 100, features)
        assert cache.get_edge_features("v1", 0, 100) == features

    def test_feature_set_namespace_isolated(self):
        cache = _make_cache_no_redis()
        cache.set_edge_features("v1", 0, 100, "basic", feature_set="basic")
        cache.set_edge_features("v1", 0, 100, "rich", feature_set="rich")
        assert cache.get_edge_features("v1", 0, 100, feature_set="basic") == "basic"
        assert cache.get_edge_features("v1", 0, 100, feature_set="rich") == "rich"

    def test_invalidate_version_clears_lru(self):
        cache = _make_cache_no_redis()
        cache.set_edge_features("v1", 0, 100, {"x": 1})
        cache.invalidate_version("v1")
        assert cache.lru_size == 0


# ---------------------------------------------------------------------------
# cached_graph_computation decorator
# ---------------------------------------------------------------------------

class TestCachedGraphComputationDecorator:
    def test_result_is_cached(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.set.return_value = None

        inner_cache = GraphComputationCache(lru_capacity=8)
        inner_cache._redis = mock_redis

        call_count = 0

        @cached_graph_computation(cache=inner_cache)
        def expensive(data_version, start_ts, end_ts):
            nonlocal call_count
            call_count += 1
            return {"nodes": 42}

        r1 = expensive(data_version="v1", start_ts=0, end_ts=100)
        r2 = expensive(data_version="v1", start_ts=0, end_ts=100)
        assert r1 == r2
        assert call_count == 1  # second call served from LRU

    def test_different_windows_invoke_function(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.set.return_value = None

        inner_cache = GraphComputationCache(lru_capacity=8)
        inner_cache._redis = mock_redis

        call_count = 0

        @cached_graph_computation(cache=inner_cache)
        def build(data_version, start_ts, end_ts):
            nonlocal call_count
            call_count += 1
            return start_ts

        build(data_version="v1", start_ts=0, end_ts=50)
        build(data_version="v1", start_ts=51, end_ts=100)
        assert call_count == 2
