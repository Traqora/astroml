"""Redis caching layer for AstroML.

This module provides Redis-based caching for frequently accessed data including:
- Graph snapshots
- Feature computation results
- Model predictions
- Artifact metadata

The caching layer supports:
- Configurable TTL per data type
- Cache invalidation on data updates
- Cache hit/miss metrics
- Decorator-based caching
"""

from __future__ import annotations

from astroml.cache.decorators import cache_feature_store
from astroml.cache.graph_cache import (
    GraphComputationCache,
    cached_graph_computation,
)
from astroml.cache.redis_cache import (
    CacheConfig,
    CacheStats,
    RedisCache,
    cached,
    cached_feature,
    cached_graph_snapshot,
    cached_prediction,
    clear_all_caches,
    get_cache_stats,
    invalidate_cache,
)

__all__ = [
    "RedisCache",
    "CacheConfig",
    "CacheStats",
    "cached",
    "cached_feature",
    "cached_prediction",
    "cached_graph_snapshot",
    "cache_feature_store",
    "GraphComputationCache",
    "cached_graph_computation",
    "invalidate_cache",
    "get_cache_stats",
    "clear_all_caches",
]
