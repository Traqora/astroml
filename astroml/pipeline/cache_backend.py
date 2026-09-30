from astroml.utils.exceptions import AstroMLError
"""Cache backends for pipeline caching — local filesystem, Redis, and S3.

Issue #636: Multi-backend cache support for pipeline caching system.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BackendStats:
    """Statistics for a cache backend instance."""

    hits: int = 0
    misses: int = 0
    puts: int = 0
    evictions: int = 0
    errors: int = 0
    total_bytes_stored: int = 0
    total_bytes_retrieved: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "puts": self.puts,
            "evictions": self.evictions,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "total_bytes_stored": self.total_bytes_stored,
            "total_bytes_retrieved": self.total_bytes_retrieved,
        }


class CacheBackend(ABC):
    """Abstract base class for cache backends.

    All backends must implement get, put, delete, exists, and clear.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Retrieve a value from the cache."""
        ...

    @abstractmethod
    def put(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to store (must be picklable).
            ttl_seconds: Optional time-to-live in seconds.

        Returns:
            True on success.
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a specific key from the cache."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        ...

    @abstractmethod
    def clear(self) -> bool:
        """Clear all entries from the cache."""
        ...

    @abstractmethod
    def stats(self) -> BackendStats:
        """Return current backend statistics."""
        ...

    def close(self) -> None:
        """Release any resources held by the backend."""
        pass


class LocalFileBackend(CacheBackend):
    """Filesystem-based cache backend.

    Stores serialized values in a directory on the local filesystem.
    Suitable for development and single-machine deployments.

    Example:
        backend = LocalFileBackend("/tmp/astroml_cache", max_size_gb=10)
        backend.put("my-key", my_data, ttl_seconds=3600)
        val = backend.get("my-key")
    """

    def __init__(
        self,
        cache_dir: str | Path = "/tmp/astroml_cache",
        max_size_gb: float = 10.0,
        default_ttl_seconds: int = 3600,
    ) -> None:
        """Initialize local filesystem cache backend.

        Args:
            cache_dir: Directory to store cache files.
            max_size_gb: Maximum total cache size in GB (enforced as soft limit).
            default_ttl_seconds: Default TTL when none is specified.
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.default_ttl = default_ttl_seconds
        self._stats = BackendStats()

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert a cache key to a filesystem path using hash bucketing."""
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        # Use first 3 chars as subdirectory to avoid too many files in one dir
        subdir = safe_key[:3]
        bucket_dir = self.cache_dir / subdir
        bucket_dir.mkdir(exist_ok=True)
        return bucket_dir / f"{safe_key}.cache"

    def _serialize(self, value: Any) -> bytes:
        """Serialize a value to bytes."""
        return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes back to a value."""
        return pickle.loads(data)

    def _read_metadata(self, path: Path) -> dict[str, Any] | None:
        """Read cache metadata file."""
        meta_path = path.with_suffix(".meta")
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _write_metadata(self, path: Path, metadata: dict[str, Any]) -> None:
        """Write cache metadata file."""
        meta_path = path.with_suffix(".meta")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

    def get(self, key: str) -> Any | None:
        """Retrieve a value from the local filesystem cache."""
        path = self._key_to_path(key)
        if not path.exists():
            self._stats.misses += 1
            return None

        metadata = self._read_metadata(path)
        if metadata is None:
            self._stats.misses += 1
            return None

        # Check TTL
        expiry = metadata.get("expiry")
        if expiry is not None and time.time() > expiry:
            self._evict(path)
            self._stats.misses += 1
            return None

        try:
            data = path.read_bytes()
            self._stats.hits += 1
            self._stats.total_bytes_retrieved += len(data)
            return self._deserialize(data)
        except (OSError, pickle.PickleError) as e:
            logger.warning(f"LocalFileBackend GET error for key {key}: {e}")
            self._stats.errors += 1
            return None

    def put(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Store a value in the local filesystem cache."""
        path = self._key_to_path(key)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        try:
            serialized = self._serialize(value)
            path.write_bytes(serialized)

            metadata = {
                "key": key,
                "created_at": time.time(),
                "ttl": ttl,
                "expiry": time.time() + ttl,
                "size_bytes": len(serialized),
            }
            self._write_metadata(path, metadata)

            self._stats.puts += 1
            self._stats.total_bytes_stored += len(serialized)

            # Lazy eviction if over size limit
            self._maybe_evict()
            return True
        except (OSError, pickle.PickleError, TypeError) as e:
            logger.warning(f"LocalFileBackend PUT error for key {key}: {e}")
            self._stats.errors += 1
            return False

    def delete(self, key: str) -> bool:
        """Delete a specific key from the local cache."""
        path = self._key_to_path(key)
        meta_path = path.with_suffix(".meta")
        existed = path.exists()
        try:
            if path.exists():
                path.unlink()
                self._stats.total_bytes_stored -= path.stat().st_size if path.exists() else 0
            if meta_path.exists():
                meta_path.unlink()
        except OSError:
            pass
        return existed

    def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        path = self._key_to_path(key)
        if not path.exists():
            return False

        metadata = self._read_metadata(path)
        if metadata is None:
            return False

        expiry = metadata.get("expiry")
        if expiry is not None and time.time() > expiry:
            self._evict(path)
            return False

        return True

    def clear(self) -> bool:
        """Clear all entries from the local cache."""
        try:
            for item in self.cache_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    for f in item.iterdir():
                        f.unlink()
                    item.rmdir()
            self._stats = BackendStats()
            return True
        except OSError as e:
            logger.warning(f"LocalFileBackend CLEAR error: {e}")
            self._stats.errors += 1
            return False

    def stats(self) -> BackendStats:
        """Return current backend statistics."""
        return self._stats

    def _evict(self, path: Path) -> None:
        """Remove a cache entry and its metadata."""
        try:
            if path.exists():
                path.unlink()
            meta_path = path.with_suffix(".meta")
            if meta_path.exists():
                meta_path.unlink()
            self._stats.evictions += 1
        except OSError:
            pass

    def _maybe_evict(self) -> None:
        """Evict oldest entries if we're over the size limit."""
        total_size = self._compute_total_size()
        if total_size <= self.max_size_bytes:
            return

        # Collect all cache entries with timestamps
        entries: list[tuple[float, Path]] = []
        for bucket in self.cache_dir.iterdir():
            if not bucket.is_dir():
                continue
            for f in bucket.glob("*.cache"):
                meta = self._read_metadata(f)
                if meta:
                    entries.append((meta["created_at"], f))

        # Sort oldest first
        entries.sort(key=lambda e: e[0])

        # Evict until under limit
        for _, path in entries:
            if self._compute_total_size() <= self.max_size_bytes * 0.8:  # evict to 80%
                break
            self._evict(path)

    def _compute_total_size(self) -> int:
        """Compute total size of cache in bytes."""
        total = 0
        for bucket in self.cache_dir.iterdir():
            if not bucket.is_dir():
                continue
            for f in bucket.glob("*.cache"):
                total += f.stat().st_size
        return total


class RedisBackend(CacheBackend):
    """Redis-based cache backend.

    Uses the existing RedisCache infrastructure for high-performance,
    network-accessible caching.

    Example:
        backend = RedisBackend(redis_url="redis://localhost:6379")
        backend.put("my-key", my_data, ttl_seconds=300)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "pipeline",
        default_ttl_seconds: int = 3600,
    ) -> None:
        """Initialize Redis cache backend.

        Args:
            redis_url: Redis connection URL.
            prefix: Key prefix for namespacing.
            default_ttl_seconds: Default TTL when none specified.
        """
        from astroml.cache.redis_cache import CacheConfig, RedisCache

        self.prefix = prefix
        self.default_ttl = default_ttl_seconds
        self._stats = BackendStats()

        config = CacheConfig(
            redis_url=redis_url,
            default_ttl_seconds=default_ttl_seconds,
        )
        self._redis = RedisCache(config)

    def _make_key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def get(self, key: str) -> Any | None:
        """Retrieve a value from Redis."""
        result = self._redis.get(self._make_key(key))
        if result is not None:
            self._stats.hits += 1
        else:
            self._stats.misses += 1
        return result

    def put(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Store a value in Redis."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        success = self._redis.set(self._make_key(key), value, ttl)
        if success:
            self._stats.puts += 1
        else:
            self._stats.errors += 1
        return success

    def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        return self._redis.delete(self._make_key(key))

    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        return self._redis.exists(self._make_key(key))

    def clear(self) -> bool:
        """Clear all pipeline keys from Redis."""
        return self._redis.delete_pattern(f"{self.prefix}:*") > 0

    def stats(self) -> BackendStats:
        """Return current backend statistics."""
        redis_stats = self._redis.get_stats()
        self._stats.hits = redis_stats.hits
        self._stats.misses = redis_stats.misses
        self._stats.errors = redis_stats.errors
        return self._stats


class S3Backend(CacheBackend):
    """S3-compatible cache backend for distributed pipeline caching.

    Stores cached values as S3 objects. Requires boto3.

    Example:
        backend = S3Backend(
            bucket="astroml-cache",
            prefix="pipeline/",
            region="us-east-1",
        )
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "pipeline/",
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        default_ttl_seconds: int = 3600,
    ) -> None:
        """Initialize S3 cache backend.

        Args:
            bucket: S3 bucket name.
            prefix: Object key prefix.
            region: AWS region.
            endpoint_url: Optional S3-compatible endpoint (e.g., MinIO).
            aws_access_key_id: Optional AWS access key.
            aws_secret_access_key: Optional AWS secret key.
            default_ttl_seconds: Default TTL when none specified.
        """
        import boto3

        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.default_ttl = default_ttl_seconds
        self._stats = BackendStats()

        client_kwargs: dict[str, Any] = {"region_name": region}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if aws_access_key_id:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self._client = boto3.client("s3", **client_kwargs)

    def _make_key(self, key: str) -> str:
        safe = hashlib.sha256(key.encode()).hexdigest()
        return f"{self.prefix}{safe}"

    def get(self, key: str) -> Any | None:
        """Retrieve a value from S3."""
        s3_key = self._make_key(key)
        try:
            # Check metadata for TTL
            head = self._client.head_object(Bucket=self.bucket, Key=s3_key)
            expiry_tag = head.get("Metadata", {}).get("expiry")
            if expiry_tag and time.time() > float(expiry_tag):
                self._client.delete_object(Bucket=self.bucket, Key=s3_key)
                self._stats.misses += 1
                return None

            response = self._client.get_object(Bucket=self.bucket, Key=s3_key)
            data = response["Body"].read()
            self._stats.hits += 1
            self._stats.total_bytes_retrieved += len(data)
            return pickle.loads(data)
        except self._client.exceptions.NoSuchKey:
            self._stats.misses += 1
            return None
        except AstroMLError as e:
            logger.warning(f"S3Backend GET error for key {key}: {e}")
            self._stats.errors += 1
            return None

    def put(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Store a value in S3."""
        s3_key = self._make_key(key)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        try:
            serialized = pickle.dumps(value)
            expiry = time.time() + ttl

            self._client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=serialized,
                ContentType="application/octet-stream",
                Metadata={
                    "expiry": str(expiry),
                    "created_at": str(time.time()),
                },
            )
            self._stats.puts += 1
            self._stats.total_bytes_stored += len(serialized)
            return True
        except AstroMLError as e:
            logger.warning(f"S3Backend PUT error for key {key}: {e}")
            self._stats.errors += 1
            return False

    def delete(self, key: str) -> bool:
        """Delete a key from S3."""
        s3_key = self._make_key(key)
        try:
            self._client.head_object(Bucket=self.bucket, Key=s3_key)
            self._client.delete_object(Bucket=self.bucket, Key=s3_key)
            return True
        except self._client.exceptions.NoSuchKey:
            return False
        except AstroMLError as e:
            logger.warning(f"S3Backend DELETE error: {e}")
            self._stats.errors += 1
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists in S3 and is not expired."""
        s3_key = self._make_key(key)
        try:
            head = self._client.head_object(Bucket=self.bucket, Key=s3_key)
            expiry_tag = head.get("Metadata", {}).get("expiry")
            if expiry_tag and time.time() > float(expiry_tag):
                return False
            return True
        except self._client.exceptions.NoSuchKey:
            return False
        except AstroMLError:
            self._stats.errors += 1
            return False

    def clear(self) -> bool:
        """Clear all pipeline keys from S3.

        Warning: this can be slow for large buckets. Use with caution.
        """
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
                objects = page.get("Contents", [])
                if objects:
                    self._client.delete_objects(
                        Bucket=self.bucket,
                        Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
                    )
            self._stats = BackendStats()
            return True
        except AstroMLError as e:
            logger.warning(f"S3Backend CLEAR error: {e}")
            self._stats.errors += 1
            return False

    def stats(self) -> BackendStats:
        """Return current backend statistics."""
        return self._stats

    def close(self) -> None:
        """No-op for S3 client (managed by boto3 session)."""
        pass


def create_backend(
    backend_type: str = "local",
    **kwargs: Any,
) -> CacheBackend:
    """Factory function to create a cache backend.

    Args:
        backend_type: One of 'local', 'redis', 's3'.
        **kwargs: Backend-specific configuration.

    Returns:
        Initialized CacheBackend instance.

    Raises:
        ValueError: If backend_type is unknown.
    """
    if backend_type == "local":
        return LocalFileBackend(**kwargs)
    elif backend_type == "redis":
        return RedisBackend(**kwargs)
    elif backend_type == "s3":
        return S3Backend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type!r}. "
            f"Use one of: 'local', 'redis', 's3'."
        )


__all__ = [
    "BackendStats",
    "CacheBackend",
    "LocalFileBackend",
    "RedisBackend",
    "S3Backend",
    "create_backend",
]