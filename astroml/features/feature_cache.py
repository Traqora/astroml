"""Feature caching and storage optimization for the Feature Store.

This module provides advanced caching mechanisms, storage optimization,
and retrieval strategies for efficient feature access.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Set
from enum import Enum
import threading
import time
from contextlib import contextmanager
from functools import lru_cache, wraps

import pandas as pd
import numpy as np
from cachetools import TTLCache, LRUCache
import redis
from pyarrow import parquet as pq
from pyarrow import Table as ArrowTable

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Caching strategies."""
    LRU = "lru"
    TTL = "ttl"
    LFU = "lfu"
    REDIS = "redis"
    DISK = "disk"


class StorageFormat(Enum):
    """Storage formats for feature data."""
    PARQUET = "parquet"
    FEATHER = "feather"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    CSV = "csv"


@dataclass
class CacheConfig:
    """Configuration for feature caching.
    
    Attributes:
        strategy: Caching strategy
        max_size: Maximum cache size
        ttl_seconds: Time-to-live in seconds (for TTL cache)
        redis_url: Redis connection URL (for Redis cache)
        disk_path: Disk cache path (for disk cache)
        compression: Whether to use compression
    """
    
    strategy: CacheStrategy = CacheStrategy.LRU
    max_size: int = 1000
    ttl_seconds: Optional[int] = None
    redis_url: Optional[str] = None
    disk_path: Optional[str] = None
    compression: bool = True


@dataclass
class StorageConfig:
    """Configuration for feature storage.
    
    Attributes:
        format: Storage format
        compression: Compression algorithm
        partition_cols: Columns to partition by
        index_cols: Columns to index
        chunk_size: Chunk size for large datasets
    """
    
    format: StorageFormat = StorageFormat.PARQUET
    compression: str = "snappy"
    partition_cols: Optional[List[str]] = None
    index_cols: Optional[List[str]] = None
    chunk_size: Optional[int] = None


@dataclass
class CacheEntry:
    """Cache entry with metadata.
    
    Attributes:
        key: Cache key
        value: Cached value
        timestamp: Cache timestamp
        access_count: Number of accesses
        size_bytes: Size in bytes
        ttl_seconds: Time-to-live
        metadata: Additional metadata
    """
    
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.utcnow() > self.timestamp + timedelta(seconds=self.ttl_seconds)
    
    def access(self) -> Any:
        """Access the cached value."""
        self.access_count += 1
        return self.value


class MemoryCache:
    """In-memory cache implementation."""
    
    def __init__(self, config: CacheConfig):
        """Initialize memory cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        
        if config.strategy == CacheStrategy.LRU:
            self._cache = LRUCache(maxsize=config.max_size)
        elif config.strategy == CacheStrategy.TTL:
            self._cache = TTLCache(maxsize=config.max_size, ttl=config.ttl_seconds or 3600)
        else:
            raise ValueError(f"Unsupported memory cache strategy: {config.strategy}")
        
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired
        """
        with self._lock:
            if key in self._cache:
                if isinstance(self._cache[key], CacheEntry):
                    entry = self._cache[key]
                    if not entry.is_expired:
                        return entry.access()
                    else:
                        # Remove expired entry
                        del self._cache[key]
                else:
                    return self._cache[key]
            return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Custom TTL override
        """
        with self._lock:
            if self.config.strategy == CacheStrategy.TTL or ttl_seconds:
                entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl_seconds=ttl_seconds or self.config.ttl_seconds,
                )
                self._cache[key] = entry
            else:
                self._cache[key] = value
    
    def remove(self, key: str) -> bool:
        """Remove value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if value was removed
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        with self._lock:
            return len(self._cache)
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())


class RedisCache:
    """Redis-based distributed cache implementation."""
    
    def __init__(self, config: CacheConfig):
        """Initialize Redis cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.redis_client = redis.from_url(config.redis_url or "redis://localhost:6379")
        self._prefix = "feature_store:"
    
    def _make_key(self, key: str) -> str:
        """Make Redis key."""
        return f"{self._prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found
        """
        try:
            data = self.redis_client.get(self._make_key(key))
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Put value in Redis cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds
        """
        try:
            data = pickle.dumps(value)
            redis_key = self._make_key(key)
            
            if ttl_seconds:
                self.redis_client.setex(redis_key, ttl_seconds, data)
            else:
                self.redis_client.set(redis_key, data)
        except Exception as e:
            logger.error(f"Redis put error: {e}")
    
    def remove(self, key: str) -> bool:
        """Remove value from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if value was removed
        """
        try:
            result = self.redis_client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis remove error: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            pattern = f"{self._prefix}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def size(self) -> int:
        """Get cache size."""
        try:
            pattern = f"{self._prefix}*"
            keys = self.redis_client.keys(pattern)
            return len(keys)
        except Exception as e:
            logger.error(f"Redis size error: {e}")
            return 0


class DiskCache:
    """Disk-based cache implementation."""
    
    def __init__(self, config: CacheConfig):
        """Initialize disk cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.cache_path = Path(config.disk_path or "./feature_cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata database
        self.db_path = self.cache_path / "cache_metadata.db"
        self._init_metadata_db()
    
    def _init_metadata_db(self) -> None:
        """Initialize metadata database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER,
                    ttl_seconds INTEGER,
                    metadata TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_cache_entries_timestamp 
                    ON cache_entries(timestamp);
            """)
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash of key for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_path / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_path, timestamp, ttl_seconds FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    file_path, timestamp_str, ttl_seconds = row
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    # Check TTL
                    if ttl_seconds and datetime.utcnow() > timestamp + timedelta(seconds=ttl_seconds):
                        # Remove expired entry
                        self.remove(key)
                        return None
                    
                    # Load value from file
                    file_path_obj = Path(file_path)
                    if file_path_obj.exists():
                        with open(file_path_obj, 'rb') as f:
                            value = pickle.load(f)
                        
                        # Update access count
                        conn.execute(
                            "UPDATE cache_entries SET access_count = access_count + 1 WHERE key = ?",
                            (key,)
                        )
                        conn.commit()
                        
                        return value
                    else:
                        # File doesn't exist, remove metadata
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        conn.commit()
        
        except Exception as e:
            logger.error(f"Disk cache get error: {e}")
        
        return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Put value in disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds
        """
        try:
            file_path = self._get_file_path(key)
            
            # Save value to file
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update metadata
            file_size = file_path.stat().st_size
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries 
                    (key, file_path, timestamp, access_count, size_bytes, ttl_seconds, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        str(file_path),
                        datetime.utcnow().isoformat(),
                        0,
                        file_size,
                        ttl_seconds,
                        json.dumps({})
                    )
                )
                conn.commit()
        
        except Exception as e:
            logger.error(f"Disk cache put error: {e}")
    
    def remove(self, key: str) -> bool:
        """Remove value from disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if value was removed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_path FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    file_path = Path(row[0])
                    
                    # Remove file
                    if file_path.exists():
                        file_path.unlink()
                    
                    # Remove metadata
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    
                    return True
        
        except Exception as e:
            logger.error(f"Disk cache remove error: {e}")
        
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            # Remove all cache files
            for cache_file in self.cache_path.glob("*.cache"):
                cache_file.unlink()
            
            # Clear metadata
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
        
        except Exception as e:
            logger.error(f"Disk cache clear error: {e}")
    
    def size(self) -> int:
        """Get cache size."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Disk cache size error: {e}")
            return 0
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries.
        
        Returns:
            Number of expired entries removed
        """
        try:
            removed_count = 0
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT key, file_path, timestamp, ttl_seconds 
                    FROM cache_entries 
                    WHERE ttl_seconds IS NOT NULL
                    """
                )
                
                for row in cursor.fetchall():
                    key, file_path, timestamp_str, ttl_seconds = row
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    if datetime.utcnow() > timestamp + timedelta(seconds=ttl_seconds):
                        # Remove expired entry
                        file_path_obj = Path(file_path)
                        if file_path_obj.exists():
                            file_path_obj.unlink()
                        
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        removed_count += 1
                
                conn.commit()
            
            return removed_count
        
        except Exception as e:
            logger.error(f"Disk cache cleanup error: {e}")
            return 0


class FeatureCache:
    """Unified feature cache interface."""
    
    def __init__(self, config: CacheConfig):
        """Initialize feature cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        
        # Initialize cache backend
        if config.strategy == CacheStrategy.REDIS:
            self._backend = RedisCache(config)
        elif config.strategy == CacheStrategy.DISK:
            self._backend = DiskCache(config)
        else:
            self._backend = MemoryCache(config)
        
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
        }
        self._lock = threading.RLock()
    
    def _make_key(self, feature_name: str, entity_ids: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Make cache key.
        
        Args:
            feature_name: Feature name
            entity_ids: List of entity IDs
            **kwargs: Additional parameters
            
        Returns:
            Cache key
        """
        key_parts = [feature_name]
        
        if entity_ids:
            # Sort entity IDs for consistent key
            sorted_ids = sorted(entity_ids)
            key_parts.append(f"entities:{','.join(sorted_ids[:10])}")  # Limit for key length
            if len(sorted_ids) > 10:
                key_parts.append(f"count:{len(sorted_ids)}")
        
        # Add relevant parameters to key
        for param_name in ["timestamp", "version", "window_size"]:
            if param_name in kwargs:
                key_parts.append(f"{param_name}:{kwargs[param_name]}")
        
        return ":".join(key_parts)
    
    def get(self, feature_name: str, entity_ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[pd.DataFrame]:
        """Get feature from cache.
        
        Args:
            feature_name: Feature name
            entity_ids: List of entity IDs
            **kwargs: Additional parameters
            
        Returns:
            Cached feature data if found
        """
        key = self._make_key(feature_name, entity_ids, **kwargs)
        value = self._backend.get(key)
        
        with self._lock:
            if value is not None:
                self._stats["hits"] += 1
                return value
            else:
                self._stats["misses"] += 1
                return None
    
    def put(
        self,
        feature_name: str,
        data: pd.DataFrame,
        entity_ids: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Put feature in cache.
        
        Args:
            feature_name: Feature name
            data: Feature data
            entity_ids: List of entity IDs
            ttl_seconds: TTL override
            **kwargs: Additional parameters
        """
        key = self._make_key(feature_name, entity_ids, **kwargs)
        self._backend.put(key, data, ttl_seconds)
        
        with self._lock:
            self._stats["sets"] += 1
    
    def remove(self, feature_name: str, entity_ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Remove feature from cache.
        
        Args:
            feature_name: Feature name
            entity_ids: List of entity IDs
            **kwargs: Additional parameters
            
        Returns:
            True if feature was removed
        """
        key = self._make_key(feature_name, entity_ids, **kwargs)
        result = self._backend.remove(key)
        
        with self._lock:
            if result:
                self._stats["deletes"] += 1
        
        return result
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._backend.clear()
        
        with self._lock:
            self._stats = {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "deletes": 0,
            }
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        with self._lock:
            stats = self._stats.copy()
            stats["size"] = self._backend.size()
            
            if stats["hits"] + stats["misses"] > 0:
                stats["hit_rate"] = stats["hits"] / (stats["hits"] + stats["misses"])
            else:
                stats["hit_rate"] = 0.0
            
            return stats
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries.
        
        Returns:
            Number of expired entries removed
        """
        if hasattr(self._backend, 'cleanup_expired'):
            return self._backend.cleanup_expired()
        return 0


class FeatureStorageOptimizer:
    """Optimizes feature storage for efficient access."""
    
    def __init__(self, storage_config: StorageConfig):
        """Initialize storage optimizer.
        
        Args:
            storage_config: Storage configuration
        """
        self.config = storage_config
    
    def optimize_dataframe(self, data: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """Optimize DataFrame for storage.
        
        Args:
            data: Input DataFrame
            feature_name: Feature name
            
        Returns:
            Optimized DataFrame
        """
        optimized = data.copy()
        
        # Optimize data types
        for col in optimized.columns:
            if optimized[col].dtype == 'object':
                # Try to convert to categorical if low cardinality
                unique_ratio = optimized[col].nunique() / len(optimized)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimized[col] = optimized[col].astype('category')
            
            elif optimized[col].dtype in ['int64', 'float64']:
                # Downcast numeric types
                if optimized[col].dtype == 'int64':
                    optimized[col] = pd.to_numeric(optimized[col], downcast='integer')
                elif optimized[col].dtype == 'float64':
                    optimized[col] = pd.to_numeric(optimized[col], downcast='float')
        
        # Set appropriate index
        if optimized.index.name != feature_name:
            optimized.index.name = feature_name
        
        return optimized
    
    def save_dataframe(self, data: pd.DataFrame, filepath: Path) -> None:
        """Save DataFrame with optimal format.
        
        Args:
            data: DataFrame to save
            filepath: Output file path
        """
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.format == StorageFormat.PARQUET:
            data.to_parquet(
                filepath,
                engine='pyarrow',
                compression=self.config.compression,
                index=True
            )
        elif self.config.format == StorageFormat.FEATHER:
            data.to_feather(filepath)
        elif self.config.format == StorageFormat.HDF5:
            data.to_hdf(
                filepath,
                key='features',
                mode='w',
                complevel=9 if self.config.compression else 0,
                complib='blosc' if self.config.compression else None
            )
        elif self.config.format == StorageFormat.PICKLE:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif self.config.format == StorageFormat.CSV:
            data.to_csv(filepath, index=True)
        else:
            raise ValueError(f"Unsupported storage format: {self.config.format}")
    
    def load_dataframe(self, filepath: Path) -> pd.DataFrame:
        """Load DataFrame from file.
        
        Args:
            filepath: File path
            
        Returns:
            Loaded DataFrame
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if self.config.format == StorageFormat.PARQUET:
            return pd.read_parquet(filepath)
        elif self.config.format == StorageFormat.FEATHER:
            return pd.read_feather(filepath)
        elif self.config.format == StorageFormat.HDF5:
            return pd.read_hdf(filepath, key='features')
        elif self.config.format == StorageFormat.PICKLE:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif self.config.format == StorageFormat.CSV:
            return pd.read_csv(filepath, index_col=0)
        else:
            raise ValueError(f"Unsupported storage format: {self.config.format}")
    
    def estimate_size(self, data: pd.DataFrame) -> int:
        """Estimate storage size in bytes.
        
        Args:
            data: DataFrame
            
        Returns:
            Estimated size in bytes
        """
        # Use memory usage as estimate
        return data.memory_usage(deep=True).sum()


# Decorators for caching

def cached_feature(
    cache: FeatureCache,
    ttl_seconds: Optional[int] = None,
    key_func: Optional[Callable] = None,
):
    """Decorator for caching feature computation functions.
    
    Args:
        cache: Feature cache instance
        ttl_seconds: TTL override
        key_func: Custom key generation function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        key_parts.append(f"df:{len(arg)}")
                    else:
                        key_parts.append(str(arg))
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}:{v}")
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache._backend.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache._backend.put(cache_key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator


# Convenience functions

def create_feature_cache(
    strategy: CacheStrategy = CacheStrategy.LRU,
    max_size: int = 1000,
    ttl_seconds: Optional[int] = None,
    **kwargs: Any,
) -> FeatureCache:
    """Create a feature cache instance.
    
    Args:
        strategy: Caching strategy
        max_size: Maximum cache size
        ttl_seconds: TTL in seconds
        **kwargs: Additional configuration
        
    Returns:
        Feature cache instance
    """
    config = CacheConfig(
        strategy=strategy,
        max_size=max_size,
        ttl_seconds=ttl_seconds,
        **kwargs
    )
    return FeatureCache(config)


def create_storage_optimizer(
    format: StorageFormat = StorageFormat.PARQUET,
    compression: str = "snappy",
    **kwargs: Any,
) -> FeatureStorageOptimizer:
    """Create a storage optimizer instance.
    
    Args:
        format: Storage format
        compression: Compression algorithm
        **kwargs: Additional configuration
        
    Returns:
        Storage optimizer instance
    """
    config = StorageConfig(
        format=format,
        compression=compression,
        **kwargs
    )
    return FeatureStorageOptimizer(config)
