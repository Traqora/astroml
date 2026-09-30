"""Integrity regression tests for :mod:`astroml.cache.graph_cache` (issue #1027).

A merge once concatenated two divergent implementations of
``GraphComputationCache`` into one file, leaving two module docstrings and
interleaved class bodies.  The result was not valid Python, which made
``astroml.cache`` — and therefore all of ``astroml.features`` — unimportable
and turned 46 test modules into collection errors.

These tests guard the shape of the module (parses, single docstring, exports
resolve) and the seams the corrupted file left broken, so a future bad merge of
the same two designs fails here instead of at import time.
"""

from __future__ import annotations

import ast
import py_compile
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import astroml.cache as cache_pkg
from astroml.cache import graph_cache
from astroml.cache.graph_cache import (
    GraphCacheBackend,
    GraphCacheConfig,
    GraphComputationCache,
    _LRUCache,
    _window_key,
    get_graph_cache,
)

_MODULE_PATH = Path(graph_cache.__file__)


class _StrictRedisCacheDouble:
    """RedisCache stand-in that rejects the wrong TTL keyword.

    The merged-away implementation called ``RedisCache.set(key, value, ttl=...)``
    while the real signature is ``set(key, value, ttl_seconds=...)``.  A
    permissive ``MagicMock`` hides that, so this double spells out the accepted
    signature instead.
    """

    def __init__(self) -> None:
        self.store: dict[str, tuple[object, int | None]] = {}

    def get(self, key: str):
        return self.store[key][0] if key in self.store else None

    def set(self, key: str, value, ttl_seconds: int | None = None) -> bool:
        self.store[key] = (value, ttl_seconds)
        return True

    def delete(self, key: str) -> bool:
        return self.store.pop(key, None) is not None


@pytest.fixture(autouse=True)
def _reset_shared_cache():
    """Keep the process-wide instance from leaking between tests."""
    GraphComputationCache._instance = None
    yield
    GraphComputationCache._instance = None


# ---------------------------------------------------------------------------
# Module shape
# ---------------------------------------------------------------------------


def test_graph_cache_is_valid_python():
    """`graph_cache.py` compiles — the corruption that broke #1027."""
    with tempfile.TemporaryDirectory() as tmp:
        py_compile.compile(str(_MODULE_PATH), cfile=str(Path(tmp) / "out.pyc"), doraise=True)


def test_graph_cache_has_exactly_one_module_docstring():
    """A second module docstring means two implementations got concatenated."""
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    docstring_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    ]
    assert len(docstring_nodes) == 1, (
        f"expected 1 module docstring, found {len(docstring_nodes)} — "
        "two divergent implementations were likely merged into one file"
    )


def test_cache_package_imports():
    """`astroml.cache` (and so `astroml.features`) is importable again."""
    assert cache_pkg.GraphComputationCache is GraphComputationCache


def test_every_exported_name_resolves():
    """Nothing in `astroml.cache.__all__` is missing or duplicated."""
    assert len(cache_pkg.__all__) == len(set(cache_pkg.__all__))
    missing = [name for name in cache_pkg.__all__ if not hasattr(cache_pkg, name)]
    assert missing == []


def test_both_design_surfaces_are_exported():
    """The prefix/key and window APIs are both re-exported (issue #1027)."""
    for name in (
        "GraphComputationCache",
        "GraphCacheBackend",
        "GraphCacheConfig",
        "GraphCacheStats",
        "cached_graph_computation",
        "get_graph_cache",
        "invalidate_graph_cache",
    ):
        assert name in cache_pkg.__all__, f"{name} dropped from astroml.cache.__all__"
        assert hasattr(graph_cache, name), f"{name} missing from graph_cache"


# ---------------------------------------------------------------------------
# Instance construction
# ---------------------------------------------------------------------------


def test_default_construction_is_shared():
    """`GraphComputationCache()` and `get_graph_cache()` are the same object."""
    assert GraphComputationCache() is GraphComputationCache()
    assert get_graph_cache() is GraphComputationCache()


def test_explicit_config_builds_isolated_instance():
    """Tuning the config must not mutate the process-wide instance."""
    shared = get_graph_cache()
    isolated = GraphComputationCache(GraphCacheConfig(max_size=7))
    assert isolated is not shared
    assert shared.config.max_size == GraphCacheConfig().max_size


def test_explicit_lru_capacity_builds_isolated_instance():
    """A non-default LRU capacity opts out of the shared instance."""
    shared = get_graph_cache()
    assert GraphComputationCache(lru_capacity=4) is not shared
    assert shared.lru_size == 0


# ---------------------------------------------------------------------------
# Redis fallback
# ---------------------------------------------------------------------------


def test_unreachable_redis_falls_back_to_memory():
    """A dead Redis downgrades the backend instead of raising.

    The corrupted file assigned to ``self._config`` (no such attribute) inside
    the connection-failure handler, so the fallback itself raised
    ``AttributeError``.
    """
    config = GraphCacheConfig(backend=GraphCacheBackend.REDIS, redis_url="redis://127.0.0.1:1")

    failing = MagicMock(side_effect=OSError("connection refused"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "redis", MagicMock(from_url=failing))
        cache = GraphComputationCache(config)

    assert cache.config.backend == GraphCacheBackend.MEMORY
    cache.set("graph:adjacency", "k", {"edges": 1})
    assert cache.get("graph:adjacency", "k") == {"edges": 1}


# ---------------------------------------------------------------------------
# Window API / shared RedisCache contract
# ---------------------------------------------------------------------------


def test_window_api_writes_through_redis_cache_signature():
    """`set_adjacency` uses `RedisCache.set(..., ttl_seconds=...)`.

    A wrong keyword would raise ``TypeError`` against the real
    :class:`~astroml.cache.redis_cache.RedisCache` while passing unnoticed under
    ``MagicMock``.
    """
    cache = GraphComputationCache(lru_capacity=4)
    cache._redis = _StrictRedisCacheDouble()

    cache.set_adjacency("v1", 0, 100, {"0x1": ["0x2"]})

    assert cache.get_adjacency("v1", 0, 100) == {"0x1": ["0x2"]}
    stored_key, (_value, ttl) = next(iter(cache._redis.store.items()))
    assert stored_key.startswith("graph:window:adj:")
    assert ttl == GraphCacheConfig().adjacency_ttl


def test_edge_features_use_the_redis_cache_ttl_keyword():
    """`set_edge_features` also honours the real `ttl_seconds` keyword."""
    cache = GraphComputationCache(lru_capacity=4)
    cache._redis = _StrictRedisCacheDouble()

    cache.set_edge_features("v1", 0, 100, {"w": [1.0]}, feature_set="rich")

    assert cache.get_edge_features("v1", 0, 100, feature_set="rich") == {"w": [1.0]}
    assert cache._redis.store[next(iter(cache._redis.store))][1] == (
        GraphCacheConfig().edge_feature_ttl
    )


def test_redis_cache_is_constructed_lazily():
    """A memory-only deployment never pays for a Redis connection."""
    cache = GraphComputationCache()
    assert cache._redis is None
    cache.set("graph:adjacency", "k", "v")
    assert cache._redis is None


# ---------------------------------------------------------------------------
# In-process LRU counters
# ---------------------------------------------------------------------------


def test_lru_tracks_counters():
    """`_LRUCache.to_dict` reports the counters it used to read but never set."""
    lru = _LRUCache(capacity=2)
    lru.set("a", 1)
    lru.get("a")
    lru.get("missing")
    lru.set("b", 2)
    lru.set("c", 3)  # evicts "a"

    assert lru.to_dict() == {
        "hits": 1,
        "misses": 1,
        "sets": 3,
        "evictions": 1,
        "hit_rate": 0.5,
    }


def test_lru_size_reflects_entries():
    """`lru_size` tracks the window API's in-process entries."""
    cache = GraphComputationCache(lru_capacity=4)
    cache._redis = _StrictRedisCacheDouble()
    cache.set_adjacency("v1", 0, 100, "adj")
    cache.set_edge_features("v1", 0, 100, "ef")
    assert cache.lru_size == 2
    cache.invalidate_adjacency("v1", 0, 100)
    assert cache.lru_size == 1


def test_window_key_unchanged_across_processes():
    """The window digest is a pure function of its inputs."""
    assert _window_key("v1", 0, 100) == _window_key("v1", 0, 100)
    assert _window_key("v1", 0, 100) != _window_key("v2", 0, 100)


def test_graph_cache_stats_report_evictions():
    """`GraphCacheStats.evictions` is populated rather than pinned at zero."""
    cache = GraphComputationCache(GraphCacheConfig(max_size=1))
    cache.set("lru", "k1", "v1")
    cache.set("lru", "k2", "v2")
    assert cache.get_stats().evictions == 1
    assert cache.get_stats().to_dict()["evictions"] == 1
