"""Storage backends for cache."""
import json
import logging
import os
import pickle
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CacheStore(ABC):
    """Abstract base class for cache storage backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Retrieve value by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: str, ttl: int = 3600, metadata: Dict[str, Any] = None) -> None:
        """Store value with optional TTL and metadata."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete entry by key."""
        pass

    @abstractmethod
    def scan_prefix(self, prefix: str) -> List[Tuple[str, str]]:
        """Scan all keys with given prefix."""
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a key."""
        pass


class RedisStore(CacheStore):
    """Redis-backed cache store (hot cache)."""

    def __init__(self, redis_url: str = None):
        """Initialize Redis store.

        Args:
            redis_url: Redis connection URL
        """
        try:
            import redis
        except ImportError:
            raise ImportError("redis package required for RedisStore")

        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = redis.Redis.from_url(self.redis_url, decode_responses=True)

    def get(self, key: str) -> Optional[str]:
        """Retrieve value from Redis."""
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: str, ttl: int = 3600, metadata: Dict[str, Any] = None) -> None:
        """Store value in Redis with TTL."""
        try:
            self.client.setex(key, ttl, value)
            if metadata:
                meta_key = f"{key}:meta"
                self.client.setex(meta_key, ttl, json.dumps(metadata))
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            result = self.client.delete(key, f"{key}:meta")
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def scan_prefix(self, prefix: str) -> List[Tuple[str, str]]:
        """Scan keys matching prefix."""
        try:
            keys = []
            for key in self.client.scan_iter(match=f"{prefix}*"):
                if not key.endswith(":meta"):
                    value = self.client.get(key)
                    if value:
                        keys.append((key, value))
            return keys
        except Exception as e:
            logger.error(f"Redis scan error: {e}")
            return []

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for key."""
        try:
            meta_key = f"{key}:meta"
            data = self.client.get(meta_key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.debug(f"Redis metadata error: {e}")
            return None


class SQLiteStore(CacheStore):
    """SQLite-backed cache store (warm cache)."""

    def __init__(self, db_path: str = None):
        """Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or os.path.expanduser("~/.astroml/cache.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                metadata TEXT,
                expires_at INTEGER NOT NULL
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
        self.conn.commit()

    def get(self, key: str) -> Optional[str]:
        """Retrieve value from SQLite."""
        now = int(time.time())
        cursor = self.conn.execute(
            "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
            (key, now),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str, ttl: int = 3600, metadata: Dict[str, Any] = None) -> None:
        """Store value in SQLite."""
        expires_at = int(time.time()) + ttl
        meta_json = json.dumps(metadata) if metadata else None

        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, metadata, expires_at) VALUES (?, ?, ?, ?)",
            (key, value, meta_json, expires_at),
        )
        self.conn.commit()

    def delete(self, key: str) -> bool:
        """Delete key from SQLite."""
        cursor = self.conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        self.conn.commit()
        return cursor.rowcount > 0

    def scan_prefix(self, prefix: str) -> List[Tuple[str, str]]:
        """Scan keys matching prefix."""
        now = int(time.time())
        cursor = self.conn.execute(
            "SELECT key, value FROM cache WHERE key LIKE ? AND expires_at > ?",
            (f"{prefix}%", now),
        )
        return cursor.fetchall()

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for key."""
        cursor = self.conn.execute("SELECT metadata FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return None

    def cleanup_expired(self):
        """Remove expired entries."""
        now = int(time.time())
        self.conn.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
        self.conn.commit()


class DiskStore(CacheStore):
    """Disk-backed cache store (cold storage)."""

    def __init__(self, cache_dir: str = None):
        """Initialize disk store.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.astroml/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        # Use first 2 chars for directory sharding
        import hashlib
        h = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.pkl"

    def get(self, key: str) -> Optional[str]:
        """Retrieve value from disk."""
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            # Check expiration
            if data["expires_at"] <= time.time():
                path.unlink()
                return None

            return data["value"]
        except Exception as e:
            logger.error(f"Disk read error: {e}")
            return None

    def set(self, key: str, value: str, ttl: int = 3600, metadata: Dict[str, Any] = None) -> None:
        """Store value on disk."""
        path = self._key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "value": value,
            "metadata": metadata,
            "expires_at": time.time() + ttl,
        }

        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Disk write error: {e}")

    def delete(self, key: str) -> bool:
        """Delete file from disk."""
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def scan_prefix(self, prefix: str) -> List[Tuple[str, str]]:
        """Scan files (slow, not recommended for disk store)."""
        # Not efficiently implemented for disk store
        logger.warning("scan_prefix is slow for DiskStore")
        return []

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata from disk."""
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data.get("metadata")
        except Exception:
            return None
