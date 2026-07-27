"""Cache decorators for the Redis caching layer.

This module provides convenience decorators for caching specific data types:
- graph snapshots
- feature computations
- model predictions
- artifact metadata
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar

from astroml.cache.redis_cache import RedisCache, CacheKeyPrefix, cached

F = TypeVar('F', bound=Callable[..., Any])


def cache_feature_computation(
    ttl_seconds: Optional[int] = 3600,
    invalidate_on_updates: bool = True,
):
    """Decorator for caching expensive feature computations.
    
    Features are invalidated when underlying data is updated.
    
    Args:
        ttl_seconds: TTL in seconds (default 1 hour)
        invalidate_on_updates: Whether to invalidate on data updates
        
    Returns:
        Decorated function
    """
    return cached(
        CacheKeyPrefix.FEATURE,
        ttl_seconds=ttl_seconds,
    )


def cache_model_prediction(
    ttl_seconds: Optional[int] = 300,
    include_model_version: bool = True,
):
    """Decorator for caching model predictions.
    
    Predictions are cached with model version to avoid stale predictions.
    
    Args:
        ttl_seconds: TTL in seconds (default 5 minutes)
        include_model_version: Whether to include model version in key
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = RedisCache()
            
            # Generate cache key with model version if available
            arg_hash = cache._hash_key(*args, **kwargs)
            
            # Check for model version in kwargs or args
            model_version = kwargs.get('model_version') or 'default'
            if include_model_version:
                cache_key = f"{CacheKeyPrefix.PREDICTION.value}:{func.__name__}:{model_version}:{arg_hash}"
            else:
                cache_key = f"{CacheKeyPrefix.PREDICTION.value}:{func.__name__}:{arg_hash}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Compute and cache
            result = func(*args, **kwargs)
            ttl = ttl_seconds or cache.config.ttl_overrides.get(CacheKeyPrefix.PREDICTION)
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def cache_graph_snapshot(
    ttl_seconds: Optional[int] = 3600,
    include_window_params: bool = True,
):
    """Decorator for caching graph snapshots.
    
    Graph snapshots are cached with window parameters to avoid recomputation.
    
    Args:
        ttl_seconds: TTL in seconds (default 1 hour)
        include_window_params: Whether to include window params in key
        
    Returns:
        Decorated function
    """
    return cached(
        CacheKeyPrefix.GRAPH_SNAPSHOT,
        ttl_seconds=ttl_seconds,
    )


def cache_feature_store(
    ttl_seconds: Optional[int] = 900,
    include_entity_ids: bool = True,
):
    """Decorator for caching feature store queries.
    
    Args:
        ttl_seconds: TTL in seconds (default 15 minutes)
        include_entity_ids: Whether to include entity IDs in key
        
    Returns:
        Decorated function
    """
    return cached(
        CacheKeyPrefix.FEATURE_STORE,
        ttl_seconds=ttl_seconds,
    )


def cache_structural_features(
    ttl_seconds: Optional[int] = 1800,
):
    """Decorator for caching structural graph features.
    
    Args:
        ttl_seconds: TTL in seconds (default 30 minutes)
        
    Returns:
        Decorated function
    """
    return cached(
        CacheKeyPrefix.STRUCTURAL,
        ttl_seconds=ttl_seconds,
    )