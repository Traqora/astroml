"""Redis caching service with decorators and metrics.

This module provides a Redis-based caching service with:
- Configurable TTL per data type
- Cache invalidation strategies
- Cache hit/miss metrics
- Decorator-based caching for functions
- Support for different data types (DataFrame, dict, list, etc.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast
from enum import Enum

import pandas as pd
import redis
from redis.client import Redis

logger = logging.getLogger(__name__)


# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])


class CacheKeyPrefix(Enum):
    """Cache key prefixes for different data types."""
    GRAPH_SNAPSHOT = "graph:snapshot"
    GRAPH_WINDOW = "graph:window"
    FEATURE = "feature"
    NODE_FEATURE = "node:feature"
    STRUCTURAL = "structural"
    PREDICTION = "prediction"
    LINK_PREDICTION = "link:pred"
    FEATURE_STORE = "feature:store"
    ARTIFACT = "artifact"
    INGESTION_STATE = "ingestion:state"


@dataclass
class CacheConfig:
    """Configuration for Redis cache."""
    
    redis_url: str = "redis://localhost:6379"
    default_ttl_seconds: int = 300  # 5 minutes
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    decode_responses: bool = False
    
    # TTL overrides per data type (seconds)
    ttl_overrides: Dict[CacheKeyPrefix, int] = field(default_factory=lambda: {
        CacheKeyPrefix.GRAPH_SNAPSHOT: 3600,    # 1 hour
        CacheKeyPrefix.GRAPH_WINDOW: 1800,      # 30 minutes
        CacheKeyPrefix.FEATURE: 900,            # 15 minutes
        CacheKeyPrefix.NODE_FEATURE: 900,       # 15 minutes
        CacheKeyPrefix.STRUCTURAL: 1800,        # 30 minutes
        CacheKeyPrefix.PREDICTION: 300,         # 5 minutes
        CacheKeyPrefix.LINK_PREDICTION: 600,    # 10 minutes
        CacheKeyPrefix.FEATURE_STORE: 900,      # 15 minutes
        CacheKeyPrefix.ARTIFACT: 3600,          # 1 hour
        CacheKeyPrefix.INGESTION_STATE: 300,    # 5 minutes
    })


@dataclass
class CacheStats:
    """Cache statistics."""
    
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
        }


class RedisCache:
    """Redis-based caching service."""
    
    _instance: Optional[RedisCache] = None
    _lock = None  # Will be set during initialization
    
    def __new__(cls, config: Optional[CacheConfig] = None) -> "RedisCache":
        """Singleton pattern - ensure only one RedisCache instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize Redis cache.
        
        Args:
            config: Cache configuration
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.config = config or CacheConfig()
        self._stats = CacheStats()
        self._client: Optional[Redis] = None
        self._initialized = True
        
        # Initialize Redis connection
        self._connect()
        
        logger.info(f"RedisCache initialized with URL: {self.config.redis_url}")
    
    def _connect(self) -> None:
        """Establish Redis connection."""
        try:
            self._client = redis.Redis(
                connection_pool=redis.ConnectionPool(
                    connection_class=redis.Connection,
                    max_connections=self.config.max_connections,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    retry_on_timeout=self.config.retry_on_timeout,
                    decode_responses=self.config.decode_responses,
                ),
                host=self._parse_host(),
                port=self._parse_port(),
                password=self._parse_password(),
                db=self._parse_db(),
            )
            # Test connection
            self._client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._client = None
    
    def _parse_host(self) -> str:
        """Parse host from redis_url."""
        import re
        match = re.search(r'redis://(?:[^:]+:[^@]+@)?([^:/]+)', self.config.redis_url)
        if match:
            return match.group(1)
        if 'unix://' in self.config.redis_url:
            return 'localhost'
        return 'localhost'
    
    def _parse_port(self) -> int:
        """Parse port from redis_url."""
        import re
        match = re.search(r':(\d+)', self.config.redis_url)
        if match and '/redis' not in self.config.redis_url:
            return int(match.group(1))
        return 6379
    
    def _parse_password(self) -> Optional[str]:
        """Parse password from redis_url."""
        import re
        match = re.search(r'redis://([^:]+):([^@]+)@', self.config.redis_url)
        if match:
            return match.group(2)
        return None
    
    def _parse_db(self) -> int:
        """Parse database from redis_url."""
        import re
        match = re.search(r'/(\d+)$', self.config.redis_url)
        if match:
            return int(match.group(1))
        return 0
    
    @property
    def client(self) -> Redis:
        """Get Redis client."""
        if self._client is None:
            self._connect()
        if self._client is None:
            raise RuntimeError("Redis client is not available")
        return self._client
    
    def _make_key(
        self,
        prefix: CacheKeyPrefix,
        key: str,
        version: Optional[str] = None,
    ) -> str:
        """Construct cache key.
        
        Args:
            prefix: Cache key prefix
            key: Base key
            version: Optional version suffix
            
        Returns:
            Full cache key
        """
        key_parts = [prefix.value, key]
        if version:
            key_parts.append(version)
        return ":".join(key_parts)
    
    def _hash_key(self, *args, **kwargs) -> str:
        """Generate a consistent hash for function arguments."""
        # Convert args to string representation
        key_parts = []
        
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # Use shape and hash of first few rows as identifier
                key_parts.append(f"df:{arg.shape[0]}x{arg.shape[1]}")
                try:
                    # Use first 100 rows as a quick hash
                    sample = arg.head(100).to_csv(index=False).encode('utf-8')
                    key_parts.append(hashlib.md5(sample).hexdigest()[:8])
                except:
                    pass
            elif isinstance(arg, (list, tuple, set)):
                key_parts.append(f"list:{len(arg)}")
            else:
                key_parts.append(str(arg))
        
        # Add sorted kwargs
        for k, v in sorted(kwargs.items()):
            if isinstance(v, pd.DataFrame):
                key_parts.append(f"{k}:df:{v.shape[0]}x{v.shape[1]}")
            elif isinstance(v, (list, tuple, set)):
                key_parts.append(f"{k}:list:{len(v)}")
            else:
                key_parts.append(f"{k}:{v}")
        
        # Create hash for the combined string
        combined = "|".join(key_parts)
        return hashlib.md5(combined.encode('utf-8')).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        try:
            data = self.client.get(key)
            if data is not None:
                self._stats.hits += 1
                try:
                    # Try to unpickle
                    return pickle.loads(data)
                except (pickle.PickleError, TypeError, ValueError):
                    # If not pickle, try JSON
                    try:
                        return json.loads(data.decode('utf-8'))
                    except:
                        return data
            else:
                self._stats.misses += 1
                return None
        except Exception as e:
            self._stats.errors += 1
            logger.warning(f"Redis GET error: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Set value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds (uses default if None)
            
        Returns:
            True if successful
        """
        try:
            # Serialize value
            try:
                serialized = pickle.dumps(value)
            except (pickle.PickleError, TypeError):
                try:
                    serialized = json.dumps(value).encode('utf-8')
                except (TypeError, ValueError):
                    serialized = str(value).encode('utf-8')
            
            # Set with TTL
            if ttl_seconds is not None:
                result = self.client.setex(key, ttl_seconds, serialized)
            else:
                result = self.client.set(key, serialized)
            
            self._stats.sets += 1
            return result
        except Exception as e:
            self._stats.errors += 1
            logger.warning(f"Redis SET error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        try:
            result = self.client.delete(key)
            self._stats.deletes += 1
            return result > 0
        except Exception as e:
            self._stats.errors += 1
            logger.warning(f"Redis DELETE error: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern.
        
        Args:
            pattern: Key pattern (e.g., "feature:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            self._stats.errors += 1
            logger.warning(f"Redis DELETE_PATTERN error: {e}")
            return 0
    
    def clear(self) -> bool:
        """Clear all cache entries.
        
        Returns:
            True if successful
        """
        try:
            self.client.flushdb()
            self._stats = CacheStats()
            return True
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Redis CLEAR error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            self._stats.errors += 1
            return False
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = CacheStats()
    
    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds, None if no TTL
        """
        try:
            ttl = self.client.ttl(key)
            return ttl if ttl > 0 else None
        except Exception as e:
            self._stats.errors += 1
            return None


# ---------------------------------------------------------------------------
# Decorators for caching
# ---------------------------------------------------------------------------

def cached(
    prefix: CacheKeyPrefix,
    ttl_seconds: Optional[int] = None,
    key_func: Optional[Callable[..., str]] = None,
):
    """Decorator to cache function results in Redis.
    
    Args:
        prefix: Cache key prefix
        ttl_seconds: TTL in seconds (uses configured default if None)
        key_func: Custom key generation function
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = RedisCache()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: prefix + function name + argument hash
                arg_hash = cache._hash_key(*args, **kwargs)
                cache_key = f"{prefix.value}:{func.__name__}:{arg_hash}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Compute and cache result
            result = func(*args, **kwargs)
            
            # Determine TTL
            ttl = ttl_seconds or cache.config.ttl_overrides.get(prefix)
            
            # Cache the result
            cache.set(cache_key, result, ttl)
            
            return result
        return cast(F, wrapper)
    return decorator


def cached_feature(
    ttl_seconds: Optional[int] = None,
    key_func: Optional[Callable[..., str]] = None,
):
    """Decorator for caching feature computation results.
    
    Args:
        ttl_seconds: TTL in seconds
        key_func: Custom key generation function
        
    Returns:
        Decorated function
    """
    return cached(CacheKeyPrefix.FEATURE, ttl_seconds, key_func)


def cached_prediction(
    ttl_seconds: Optional[int] = None,
    key_func: Optional[Callable[..., str]] = None,
):
    """Decorator for caching model prediction results.
    
    Args:
        ttl_seconds: TTL in seconds
        key_func: Custom key generation function
        
    Returns:
        Decorated function
    """
    return cached(CacheKeyPrefix.PREDICTION, ttl_seconds, key_func)


def cached_graph_snapshot(
    ttl_seconds: Optional[int] = None,
    key_func: Optional[Callable[..., str]] = None,
):
    """Decorator for caching graph snapshot results.
    
    Args:
        ttl_seconds: TTL in seconds
        key_func: Custom key generation function
        
    Returns:
        Decorated function
    """
    return cached(CacheKeyPrefix.GRAPH_SNAPSHOT, ttl_seconds, key_func)


# ---------------------------------------------------------------------------
# Cache management functions
# ---------------------------------------------------------------------------

def invalidate_cache(prefix: CacheKeyPrefix, key: Optional[str] = None) -> int:
    """Invalidate cache entries.
    
    Args:
        prefix: Cache key prefix
        key: Optional specific key to invalidate
        
    Returns:
        Number of entries invalidated
    """
    cache = RedisCache()
    if key:
        full_key = f"{prefix.value}:{key}"
        return 1 if cache.delete(full_key) else 0
    else:
        pattern = f"{prefix.value}:*"
        return cache.delete_pattern(pattern)


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    cache = RedisCache()
    return cache.get_stats().to_dict()


def clear_all_caches() -> bool:
    """Clear all cache entries."""
    cache = RedisCache()
    return cache.clear()