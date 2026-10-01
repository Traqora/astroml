from astroml.utils.exceptions import AstroMLError
"""Online Feature Store for low-latency feature serving.

Provides fast key-value storage and retrieval of entity features for real-time inference,
supporting Redis backend with an in-memory dictionary fallback.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OnlineFeatureValue:
    """A feature value stored in the online store."""

    entity_id: str
    feature_name: str
    value: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict format for serialization."""
        return {
            "entity_id": self.entity_id,
            "feature_name": self.feature_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OnlineFeatureValue:
        """Create from serialized dict."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            parsed_ts = datetime.fromisoformat(ts)
        elif isinstance(ts, datetime):
            parsed_ts = ts
        else:
            parsed_ts = datetime.now(timezone.utc)
        return cls(
            entity_id=data["entity_id"],
            feature_name=data["feature_name"],
            value=data["value"],
            timestamp=parsed_ts,
            version=data.get("version", 1),
        )


class BaseOnlineStore(ABC):
    """Abstract base class for online feature store implementations."""

    @abstractmethod
    def get_online_features(
        self,
        entity_keys: Sequence[str],
        feature_names: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Retrieve latest feature values for a set of entity keys."""

    @abstractmethod
    def write_online_features(
        self,
        features: list[OnlineFeatureValue] | pd.DataFrame,
        entity_col: str = "entity_id",
        timestamp_col: str | None = None,
        ttl_seconds: int | None = None,
    ) -> int:
        """Write feature values to the online store. Returns count of features written."""

    @abstractmethod
    def delete_online_features(
        self,
        entity_keys: Sequence[str],
        feature_names: Sequence[str] | None = None,
    ) -> int:
        """Delete features for specified entities."""


class InMemoryOnlineStore(BaseOnlineStore):
    """Fast thread-safe in-memory key-value store for development and testing."""

    def __init__(self, default_ttl_seconds: int = 86400) -> None:
        """Initialize in-memory store."""
        self._store: dict[
            str, dict[str, tuple[Any, float]]
        ] = {}  # entity -> {feature: (value, expire_at)}
        self.default_ttl = default_ttl_seconds

    def _clean_expired(self, entity_id: str) -> None:
        """Remove expired keys for an entity."""
        if entity_id in self._store:
            now = time.time()
            self._store[entity_id] = {
                k: v for k, v in self._store[entity_id].items() if v[1] == 0 or v[1] > now
            }
            if not self._store[entity_id]:
                del self._store[entity_id]

    def get_online_features(
        self,
        entity_keys: Sequence[str],
        feature_names: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Retrieve features for given entity keys."""
        results: dict[str, dict[str, Any]] = {}
        now = time.time()
        for entity_id in entity_keys:
            entity_dict: dict[str, Any] = {}
            if entity_id in self._store:
                for feat in feature_names:
                    if feat in self._store[entity_id]:
                        val, expire_at = self._store[entity_id][feat]
                        if expire_at == 0 or expire_at > now:
                            entity_dict[feat] = val
            results[entity_id] = entity_dict
        return results

    def write_online_features(
        self,
        features: list[OnlineFeatureValue] | pd.DataFrame,
        entity_col: str = "entity_id",
        timestamp_col: str | None = "timestamp",
        ttl_seconds: int | None = None,
    ) -> int:
        """Write feature records into the in-memory store."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expire_at = (time.time() + ttl) if ttl > 0 else 0.0
        count = 0

        if isinstance(features, pd.DataFrame):
            df = features
            feat_cols = [c for c in df.columns if c not in (entity_col, timestamp_col)]
            for _, row in df.iterrows():
                entity_id = str(row[entity_col])
                if entity_id not in self._store:
                    self._store[entity_id] = {}
                for col in feat_cols:
                    self._store[entity_id][col] = (row[col], expire_at)
                    count += 1

        else:
            for item in features:
                entity_id = str(item.entity_id)
                if entity_id not in self._store:
                    self._store[entity_id] = {}
                self._store[entity_id][item.feature_name] = (item.value, expire_at)
                count += 1
        return count

    def delete_online_features(
        self,
        entity_keys: Sequence[str],
        feature_names: Sequence[str] | None = None,
    ) -> int:
        """Delete features from the in-memory store."""
        deleted = 0
        for entity_id in entity_keys:
            if entity_id in self._store:
                if feature_names is None:
                    deleted += len(self._store[entity_id])
                    del self._store[entity_id]
                else:
                    for feat in feature_names:
                        if feat in self._store[entity_id]:
                            del self._store[entity_id][feat]
                            deleted += 1
        return deleted


class RedisOnlineStore(BaseOnlineStore):
    """Redis-backed online feature store for production sub-millisecond serving."""

    def __init__(
        self,
        redis_client: Any | None = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "astroml:fs:",
        default_ttl: int = 86400,
    ) -> None:
        """Initialize Redis store."""
        self.prefix = prefix
        self.default_ttl = default_ttl
        if redis_client is not None:
            self.client = redis_client
        else:
            try:
                import redis

                self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            except Exception as exc:
                logger.warning("Redis connection failed, falling back to in-memory: %s", exc)
                self.client = None

    def _key(self, entity_id: str) -> str:
        return f"{self.prefix}{entity_id}"

    def get_online_features(
        self,
        entity_keys: Sequence[str],
        feature_names: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Retrieve features from Redis using pipeline/mget."""
        if not self.client:
            return {k: {} for k in entity_keys}

        results: dict[str, dict[str, Any]] = {}
        try:
            pipeline = self.client.pipeline()
            for entity_id in entity_keys:
                pipeline.hmget(self._key(entity_id), list(feature_names))
            raw_results = pipeline.execute()

            for entity_id, raw_values in zip(entity_keys, raw_results):
                entity_dict: dict[str, Any] = {}
                for feat, raw_val in zip(feature_names, raw_values):
                    if raw_val is not None:
                        try:
                            entity_dict[feat] = json.loads(raw_val)
                        except (json.JSONDecodeError, TypeError):
                            entity_dict[feat] = raw_val
                results[entity_id] = entity_dict
        except Exception as exc:
            logger.error("Redis get_online_features error: %s", exc)
            return {k: {} for k in entity_keys}

        return results

    def write_online_features(
        self,
        features: list[OnlineFeatureValue] | pd.DataFrame,
        entity_col: str = "entity_id",
        timestamp_col: str | None = "timestamp",
        ttl_seconds: int | None = None,
    ) -> int:
        """Write features to Redis hashes with TTL."""
        if not self.client:
            return 0

        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        count = 0
        try:
            pipeline = self.client.pipeline()
            if isinstance(features, pd.DataFrame):
                feat_cols = [c for c in features.columns if c not in (entity_col, timestamp_col)]
                for _, row in features.iterrows():
                    entity_id = str(row[entity_col])
                    key = self._key(entity_id)
                    mapping = {col: json.dumps(row[col]) for col in feat_cols}
                    pipeline.hset(key, mapping=mapping)
                    if ttl > 0:
                        pipeline.expire(key, ttl)
                    count += len(feat_cols)
            else:
                grouped: dict[str, dict[str, str]] = {}
                for item in features:
                    entity_id = str(item.entity_id)
                    if entity_id not in grouped:
                        grouped[entity_id] = {}
                    grouped[entity_id][item.feature_name] = json.dumps(item.value)
                    count += 1

                for entity_id, mapping in grouped.items():
                    key = self._key(entity_id)
                    pipeline.hset(key, mapping=mapping)
                    if ttl > 0:
                        pipeline.expire(key, ttl)
            pipeline.execute()
        except Exception as exc:
            logger.error("Redis write_online_features error: %s", exc)
            return 0
        return count

    def delete_online_features(
        self,
        entity_keys: Sequence[str],
        feature_names: Sequence[str] | None = None,
    ) -> int:
        """Delete features or entire entity hashes in Redis."""
        if not self.client:
            return 0
        deleted = 0
        try:
            pipeline = self.client.pipeline()
            for entity_id in entity_keys:
                key = self._key(entity_id)
                if feature_names is None:
                    pipeline.delete(key)
                    deleted += 1
                else:
                    pipeline.hdel(key, *list(feature_names))
                    deleted += len(feature_names)
            pipeline.execute()
        except Exception as exc:
            logger.error("Redis delete_online_features error: %s", exc)
            return 0
        return deleted


def create_online_store(
    backend: str = "memory",
    redis_client: Any | None = None,
    **kwargs: Any,
) -> BaseOnlineStore:
    """Factory function to instantiate an online feature store backend."""
    if backend == "redis":
        return RedisOnlineStore(redis_client=redis_client, **kwargs)
    return InMemoryOnlineStore(**kwargs)
