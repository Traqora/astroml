from astroml.utils.exceptions import AstroMLError
"""Feature Store implementation for AstroML.

See ADR-002 (docs/adr/002-sqlite-feature-store-metadata.md) for feature store metadata architecture.

This module provides a comprehensive feature store that centralizes feature computation,
storage, versioning, and retrieval for machine learning workflows. It integrates with
existing feature modules while adding enterprise-grade feature management capabilities.

Key Features:
- Feature definition and registration
- Computed feature storage and caching
- Feature versioning and lineage tracking
- Time-travel and point-in-time queries
- Feature metadata and documentation
- Integration with existing feature modules

Key components:
- FeatureStore: Main feature store interface with caching
- FeatureRegistry: Feature computer registration and discovery
- FeatureStorage: SQLite-backed storage backend
- FeatureDefinition: Feature metadata and configuration
- FeatureValue: Computed feature value container

Dependencies:
- pandas: Data manipulation for feature values
- cachetools: TTL-based caching
- sqlite3: Metadata storage
- astroml.features.schema_validation: Feature validation
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import sqlite3
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd
from cachetools import TTLCache

from astroml.features.schema_validation import (
    FEATURE_VALUE_SCHEMA,
    ValidationResult,
    dry_run_ingestion,
)

from ..cache import cache_feature_store

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Supported feature data types."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    VECTOR = "vector"
    TIME_SERIES = "time_series"


class FeatureStatus(Enum):
    """Feature lifecycle status."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class FeatureDefinition:
    """Definition of a feature in the feature store.

    Attributes:
        name: Unique feature name
        description: Human-readable description
        feature_type: Data type of the feature
        computation_function: Function to compute the feature
        parameters: Parameters for the computation function
        tags: List of tags for categorization
        owner: Feature owner/team
        status: Feature lifecycle status
        version: Feature version
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional metadata
    """

    name: str
    description: str
    feature_type: FeatureType
    computation_function: Callable | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    owner: str = ""
    status: FeatureStatus = FeatureStatus.DEVELOPMENT
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate feature ID and validate definition."""
        if not self.name:
            raise ValueError("FeatureDefinition.name must not be empty")
        if self.version < 1:
            raise ValueError("FeatureDefinition.version must be at least 1")

    @property
    def feature_id(self) -> str:
        """Unique feature identifier."""
        return f"{self.name}_v{self.version}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "feature_type": self.feature_type.value,
            "parameters": self.parameters,
            "tags": self.tags,
            "owner": self.owner,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureDefinition:
        """Create from dictionary representation."""
        data = data.copy()
        data.pop("feature_id", None)
        if isinstance(data.get("feature_type"), str):
            data["feature_type"] = FeatureType(data["feature_type"])
        if isinstance(data.get("status"), str):
            data["status"] = FeatureStatus(data["status"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class FeatureValue:
    """Container for computed feature values with metadata.

    Attributes:
        feature_id: Feature identifier
        entity_id: Entity identifier (account, transaction, etc.)
        value: Feature value
        timestamp: Feature computation timestamp
        validity_period: Period during which feature is valid
        metadata: Additional metadata
    """

    feature_id: str
    entity_id: str
    value: Any
    timestamp: datetime
    validity_period: timedelta | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expires_at(self) -> datetime | None:
        """Expiration timestamp for the feature value."""
        if self.validity_period:
            return self.timestamp + self.validity_period
        return None

    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if feature value is valid at given timestamp."""
        if self.expires_at and timestamp > self.expires_at:
            return False
        return timestamp >= self.timestamp


@dataclass
class FeatureSet:
    """Collection of related features for a specific use case.

    Attributes:
        name: Feature set name
        description: Feature set description
        feature_ids: List of feature identifiers
        entity_type: Type of entity (account, transaction, etc.)
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional metadata
    """

    name: str
    description: str
    feature_ids: list[str]
    entity_type: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "feature_ids": self.feature_ids,
            "entity_type": self.entity_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@runtime_checkable
class FeatureComputer(Protocol):
    """Protocol for feature computation functions."""

    def __call__(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute features from input data.

        Args:
            data: Input DataFrame
            entity_col: Entity identifier column
            timestamp_col: Timestamp column
            **kwargs: Additional parameters

        Returns:
            DataFrame with computed features indexed by entity
        """
        ...


def _safe_json_loads(value: Optional[str], default: Any) -> Any:
    """Deserialize a JSON-encoded string safely."""
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class FeatureStorage:
    """Storage backend for feature values and metadata."""

    # Column names used across multiple methods for feature definition queries.
    _FEATURE_DEFINITION_COLUMNS: list[str] = [
        "feature_id",
        "name",
        "version",
        "description",
        "feature_type",
        "parameters",
        "tags",
        "owner",
        "status",
        "created_at",
        "updated_at",
        "metadata",
    ]

    def __init__(self, storage_path -> Any: str | Path):
        """Initialize storage backend.

        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite database for metadata
        self.db_path = self.storage_path / "feature_store.db"
        self._init_database()

        # Directory for feature data
        self.data_path = self.storage_path / "data"
        self.data_path.mkdir(exist_ok=True)

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS feature_definitions (
                    feature_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    description TEXT,
                    feature_type TEXT NOT NULL,
                    parameters TEXT,
                    tags TEXT,
                    owner TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS feature_sets (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    feature_ids TEXT,
                    entity_type TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS feature_lineage (
                    feature_id TEXT,
                    parent_feature_id TEXT,
                    relationship_type TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (feature_id, parent_feature_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_feature_definitions_name 
                    ON feature_definitions(name);
                
                CREATE INDEX IF NOT EXISTS idx_feature_definitions_status 
                    ON feature_definitions(status);
            """)

    def store_feature_definition(self, feature_def: FeatureDefinition) -> None:
        """Store feature definition in database.

        Args:
            feature_def: Feature definition to store
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feature_definitions 
                (feature_id, name, version, description, feature_type, 
                 parameters, tags, owner, status, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feature_def.feature_id,
                    feature_def.name,
                    feature_def.version,
                    feature_def.description,
                    feature_def.feature_type.value,
                    json.dumps(feature_def.parameters),
                    json.dumps(feature_def.tags),
                    feature_def.owner,
                    feature_def.status.value,
                    feature_def.created_at.isoformat(),
                    feature_def.updated_at.isoformat(),
                    json.dumps(feature_def.metadata),
                ),
            )

    @staticmethod
    def _row_to_dict(row: tuple[Any, ...], columns: List[str]) -> Dict[str, Any]:
        """Map a SQLite row to a dictionary using column names."""
        data = dict(zip(columns, row))
        data["parameters"] = _safe_json_loads(data["parameters"], default={})
        data["tags"] = _safe_json_loads(data["tags"], default=[])
        data["metadata"] = _safe_json_loads(data["metadata"], default={})
        return data

    @staticmethod
    def _deserialize_feature_definition(row: tuple[Any, ...]) -> FeatureDefinition:
        """Deserialize a feature definition row into a dataclass."""
        data = FeatureStorage._row_to_dict(row, FeatureStorage._FEATURE_DEFINITION_COLUMNS)
        data.pop("feature_id", None)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return FeatureDefinition.from_dict(data)

    def get_feature_definition(self, feature_id: str) -> Optional[FeatureDefinition]:
        """Retrieve feature definition by ID.

        Args:
            feature_id: Feature identifier

        Returns:
            Feature definition if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM feature_definitions WHERE feature_id = ?",
                (feature_id,),
            )
            row = cursor.fetchone()

            if row:
                return FeatureStorage._deserialize_feature_definition(row)
            return None

    @staticmethod
    def _matches_tags(feature_tags: list[str], required_tags: list[str]) -> bool:
        """Return True if *feature_tags* contains all *required_tags*."""
        tag_set = set(feature_tags)
        return all(tag in tag_set for tag in required_tags)

    def list_feature_definitions(
        self,
        status: FeatureStatus | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
    ) -> list[FeatureDefinition]:
        """List feature definitions with optional filtering.

        Args:
            status: Filter by status
            tags: Filter by tags (must contain all specified tags)
            owner: Filter by owner

        Returns:
            List of feature definitions
        """
        query, params = self._build_feature_list_query(status, owner)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            features: List[FeatureDefinition] = []
            for row in rows:
                data = FeatureStorage._row_to_dict(row, self._FEATURE_DEFINITION_COLUMNS)
                if tags and not self._matches_tags(data["tags"], tags):
                    continue
                features.append(FeatureDefinition.from_dict(data))

            return features

    @staticmethod
    def _build_feature_list_query(
        status: FeatureStatus | None,
        owner: str | None,
    ) -> tuple[str, list[Any]]:
        """Build a parameterized SQL query for listing feature definitions."""
        query = "SELECT * FROM feature_definitions WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status.value)

        if owner:
            query += " AND owner = ?"
            params.append(owner)

        return query, params

    def store_feature_values(
        self,
        feature_id: str,
        values: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store computed feature values.

        Args:
            feature_id: Feature identifier
            values: DataFrame with feature values indexed by entity
            metadata: Additional metadata
        """
        # Store as parquet file for efficient storage and retrieval
        file_path = self.data_path / f"{feature_id}.parquet"

        # Add metadata to DataFrame
        if metadata:
            values.attrs["metadata"] = metadata
            values.attrs["feature_id"] = feature_id
            values.attrs["stored_at"] = datetime.utcnow().isoformat()

        values.to_parquet(file_path, index=True)
        logger.info(f"Stored {len(values)} feature values for {feature_id}")

    def get_feature_values(
        self,
        feature_id: str,
        entity_ids: list[str] | None = None,
        timestamp: datetime | None = None,
    ) -> pd.DataFrame | None:
        """Retrieve stored feature values.

        Args:
            feature_id: Feature identifier
            entity_ids: Optional list of entity IDs to filter
            timestamp: Optional timestamp for point-in-time queries

        Returns:
            DataFrame with feature values if found, None otherwise
        """
        file_path = self.data_path / f"{feature_id}.parquet"

        if not file_path.exists():
            return None

        values = pd.read_parquet(file_path)

        # Filter by entity IDs if specified
        if entity_ids:
            values = values[values.index.isin(entity_ids)]

        # TODO: Implement point-in-time filtering if timestamp is provided
        # This would require storing multiple versions of feature values

        return values

    def store_feature_set(self, feature_set: FeatureSet) -> None:
        """Store feature set definition.

        Args:
            feature_set: Feature set to store
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feature_sets 
                (name, description, feature_ids, entity_type, 
                 created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feature_set.name,
                    feature_set.description,
                    json.dumps(feature_set.feature_ids),
                    feature_set.entity_type,
                    feature_set.created_at.isoformat(),
                    feature_set.updated_at.isoformat(),
                    json.dumps(feature_set.metadata),
                ),
            )

    def get_feature_set(self, name: str) -> FeatureSet | None:
        """Retrieve feature set by name.

        Args:
            name: Feature set name

        Returns:
            Feature set if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM feature_sets WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()

            if row:
                columns = [
                    "name",
                    "description",
                    "feature_ids",
                    "entity_type",
                    "created_at",
                    "updated_at",
                    "metadata",
                ]
                data = dict(zip(columns, row))
                data["feature_ids"] = json.loads(data["feature_ids"])
                data["metadata"] = json.loads(data["metadata"])
                data["created_at"] = datetime.fromisoformat(data["created_at"])
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])

                return FeatureSet(**data)

            return None


class FeatureRegistry:
    """Registry for managing feature definitions and computations."""

    _global_computers: dict[str, FeatureComputer] = {}

    def __init__(self, storage -> Any: FeatureStorage):
        """Initialize feature registry.

        Args:
            storage: Storage backend
        """
        self.storage = storage
        self._computers: dict[str, FeatureComputer] = {}
        self._plugin_computers: dict[str, str] = {}
        self._register_builtin_features()
        self._discover_plugins()

    def _discover_plugins(self) -> None:
        """Discover and register feature computer plugins via entry points."""
        try:
            from importlib.metadata import entry_points

            plugins = entry_points(group="astroml.feature_computers")
            for ep in plugins:
                try:
                    computer_cls = ep.load()
                    self._validate_plugin(computer_cls)
                    name = getattr(computer_cls, "name", ep.name)
                    if hasattr(computer_cls, "compute"):
                        computer = computer_cls()
                        self.register_computer(
                            name,
                            computer.compute if hasattr(computer, "compute") else computer,
                            {
                                "description": getattr(
                                    computer_cls, "__doc__", f"Plugin: {ep.name}"
                                ),
                                "feature_type": FeatureType.NUMERIC,
                                "tags": ["plugin", ep.name],
                                "owner": "plugin",
                            },
                        )
                        self._plugin_computers[name] = ep.module or ep.name
                    elif callable(computer_cls):
                        metadata = {
                            "description": getattr(computer_cls, "__doc__", f"Plugin: {ep.name}"),
                            "feature_type": FeatureType.NUMERIC,
                            "tags": ["plugin", ep.name],
                            "owner": "plugin",
                        }
                        self.register_computer(ep.name, computer_cls, metadata)
                        self._plugin_computers[ep.name] = ep.module or ep.name
                except AstroMLError as e:
                    logger.warning(f"Failed to load plugin '{ep.name}': {e}")
        except ImportError:
            logger.debug("importlib.metadata not available for plugin discovery")
        except AstroMLError as e:
            logger.warning(f"Plugin discovery failed: {e}")

    def _validate_plugin(self, plugin_cls: type) -> None:
        """Validate that a plugin class has required methods.

        Args:
            plugin_cls: Plugin class to validate

        Raises:
            TypeError: If plugin is missing required methods
        """
        has_compute = hasattr(plugin_cls, "compute") and callable(getattr(plugin_cls, "compute"))
        is_callable_cls = callable(plugin_cls)
        if not (has_compute or is_callable_cls):
            raise TypeError(
                f"Plugin {plugin_cls.__name__} must either be callable "
                f"or have a callable 'compute' method"
            )

    def _register_builtin_features(self) -> None:
        """Register built-in feature computers from existing modules."""
        try:
            from astroml.features import (
                asset_diversity,
                frequency,
                node_features,
                structural_importance,
            )

            # Feature descriptors: (name, computer_fn, description, feature_type, tags)
            builtin_features = [
                (
                    "daily_transaction_count",
                    frequency.compute_daily_transaction_counts,
                    "Daily transaction count per account",
                    FeatureType.NUMERIC,
                    ["frequency", "activity"],
                ),
                (
                    "transaction_burstiness",
                    frequency.compute_burstiness,
                    "Transaction burstiness metric",
                    FeatureType.NUMERIC,
                    ["frequency", "behavior"],
                ),
                (
                    "degree_centrality",
                    structural_importance.compute_degree_centrality,
                    "Degree centrality in transaction graph",
                    FeatureType.NUMERIC,
                    ["graph", "centrality"],
                ),
                (
                    "betweenness_centrality",
                    structural_importance.compute_betweenness_centrality,
                    "Betweenness centrality in transaction graph",
                    FeatureType.NUMERIC,
                    ["graph", "centrality"],
                ),
                (
                    "pagerank",
                    structural_importance.compute_pagerank,
                    "PageRank score in transaction graph",
                    FeatureType.NUMERIC,
                    ["graph", "importance"],
                ),
                (
                    "node_features",
                    node_features.compute_node_features,
                    "Basic node features (degree, volume, age)",
                    FeatureType.TIME_SERIES,
                    ["node", "basic"],
                ),
                (
                    "asset_diversity",
                    asset_diversity.compute_asset_diversity,
                    "Asset diversity metrics",
                    FeatureType.NUMERIC,
                    ["asset", "diversity"],
                ),
            ]

            for name, computer_fn, desc, ftype, ftags in builtin_features:
                self.register_computer(
                    name,
                    computer_fn,
                    {
                        "description": desc,
                        "feature_type": ftype,
                        "tags": ftags,
                    },
                )

            logger.info("Registered built-in feature computers")

        except ImportError as e:
            logger.warning(f"Could not import some feature modules: {e}")

    def register_computer(
        self,
        name: str,
        computer: FeatureComputer,
        metadata: dict[str, Any],
    ) -> None:
        """Register a feature computer.

        Args:
            name: Feature name
            computer: Computation function
            metadata: Feature metadata
        """
        self._computers[name] = computer
        FeatureRegistry._global_computers[name] = computer

        # Create feature definition
        feature_def = FeatureDefinition(
            name=name,
            description=metadata.get("description", ""),
            feature_type=metadata.get("feature_type", FeatureType.NUMERIC),
            parameters=metadata.get("parameters", {}),
            tags=metadata.get("tags", []),
            owner=metadata.get("owner", "system"),
        )

        self.storage.store_feature_definition(feature_def)
        logger.info(f"Registered feature computer: {name}")

    def get_computer(self, name: str) -> FeatureComputer | None:
        """Get registered feature computer.

        Args:
            name: Feature name

        Returns:
            Feature computer if found, None otherwise
        """
        if name in self._computers:
            return self._computers[name]
        return FeatureRegistry._global_computers.get(name)

    def list_features(self) -> list[str]:
        """List all registered feature names."""
        all_keys = set(self._computers.keys()) | set(FeatureRegistry._global_computers.keys())
        return list(all_keys)


class FeatureStore:
    """Main feature store interface.

    Provides a high-level API for feature registration, computation,
    storage, and retrieval with an LRU+TTL cache backed by
    :class:`cachetools.TTLCache`.

    Cache behaviour
    ---------------
    * **maxsize** – upper bound on the number of features held in memory at
      once (LRU eviction when full).
    * **TTL** – entries older than *cache_ttl_seconds* are considered stale
      and will be re-fetched from storage on the next access.
    * **Metrics** – hit, miss, and eviction counters are maintained and
      exposed via :meth:`get_cache_stats`.
    * **Thread safety** – a :class:`threading.Lock` guards every cache
      mutation so the store is safe to use from concurrent threads.
    """

    # Default configuration (overridden by config/feature_store.yaml values
    # or constructor arguments).
    _DEFAULT_MAXSIZE: int = 128
    _DEFAULT_TTL: int = 900  # 15 minutes
    _DEFAULT_MAX_WORKERS: int = 4
    _DEFAULT_CHUNK_SIZE: int = 100

    def __init__(
        self,
        storage_path: str | Path = "./feature_store",
        max_cache_size_mb: int = 500,
        cache_ttl_seconds: int = _DEFAULT_TTL,
        cache_maxsize: int = _DEFAULT_MAXSIZE,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        enable_parallel: bool = True,
    ):
        """Initialize feature store.

        Args:
            storage_path: Path to feature store storage.
            max_cache_size_mb: Soft memory cap in MB; entries are still
                subject to TTL-based and LRU-based eviction from the
                TTLCache regardless of this value.
            cache_ttl_seconds: Seconds before a cached entry expires
                (default: 900 = 15 min, matching ``config/feature_store.yaml``).
            cache_maxsize: Maximum number of entries in the TTLCache before
                LRU eviction kicks in (default: 128).
            max_workers: Maximum number of parallel workers for feature computation
                (default: 4). Set to 1 to disable parallelism.
            chunk_size: Number of entities to process per chunk in parallel computation
                (default: 100). Larger chunks reduce overhead but may increase memory usage.
            enable_parallel: Whether to enable parallel feature computation
                (default: True).
        """
        self.storage = FeatureStorage(storage_path)
        self.registry = FeatureRegistry(self.storage)

        from .offline_store import OfflineFeatureStore, create_offline_store
        from .online_store import BaseOnlineStore, create_online_store

        self.online_store: BaseOnlineStore = create_online_store(backend="memory")
        self.offline_store: OfflineFeatureStore = create_offline_store(
            storage_path=Path(storage_path) / "offline"
        )

        # Lightweight metadata cache (feature definitions rarely change).
        self._metadata_cache: dict[str, FeatureDefinition] = {}

        # Cache and parallel settings
        self._init_cache_settings(cache_maxsize, cache_ttl_seconds, max_cache_size_mb)
        self._init_parallel_settings(max_workers, chunk_size, enable_parallel)

    def _init_cache_settings(
        self,
        cache_maxsize: int,
        cache_ttl_seconds: int,
        max_cache_size_mb: int,
    ) -> None:
        """Initialize the value cache (TTL+LRU) and related metrics."""
        self._cache_lock: threading.RLock = threading.RLock()
        self._cache_ttl_seconds: int = cache_ttl_seconds
        self._cache_maxsize: int = cache_maxsize

        self._value_cache: TTLCache = TTLCache(
            maxsize=cache_maxsize,
            ttl=cache_ttl_seconds,
        )

        # Legacy attributes kept for compatibility with code that inspects
        # memory usage directly.
        self._max_cache_size_bytes: int = max_cache_size_mb * 1024 * 1024
        self._current_cache_size_bytes: int = 0

        # Cache metrics counters.
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._cache_evictions: int = 0

    def _init_parallel_settings(
        self,
        max_workers: int,
        chunk_size: int,
        enable_parallel: bool,
    ) -> None:
        """Initialize parallel computation settings."""
        self._max_workers: int = max_workers
        self._chunk_size: int = chunk_size
        self._enable_parallel: bool = enable_parallel and max_workers > 1

    def register_feature(
        self,
        name: str,
        computer: FeatureComputer | None = None,
        description: str = "",
        feature_type: FeatureType = FeatureType.NUMERIC,
        tags: list[str] | None = None,
        owner: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> FeatureDefinition:
        """Register a new feature.

        Args:
            name: Feature name
            computer: Computation function
            description: Feature description
            feature_type: Feature data type
            tags: Feature tags
            owner: Feature owner
            parameters: Feature parameters

        Returns:
            Created feature definition
        """
        metadata = {
            "description": description,
            "feature_type": feature_type,
            "tags": tags or [],
            "owner": owner,
            "parameters": parameters or {},
        }

        if computer is not None:
            self.registry.register_computer(name, computer, metadata)
        else:
            feature_def = FeatureDefinition(
                name=name,
                description=description,
                feature_type=feature_type,
                tags=tags or [],
                owner=owner,
                parameters=parameters or {},
            )
            self.storage.store_feature_definition(feature_def)

        # Return the created feature definition
        feature_def = self.storage.get_feature_definition(f"{name}_v1")
        if feature_def is None:
            raise RuntimeError("Failed to create feature definition")

        return feature_def

    def compute_feature(
        self,
        feature_name: str,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute feature values.

        Args:
            feature_name: Name of feature to compute
            data: Input data
            entity_col: Entity identifier column
            timestamp_col: Timestamp column
            **kwargs: Additional parameters

        Returns:
            DataFrame with computed feature values
        """
        computer = self.registry.get_computer(feature_name)
        if computer is None:
            raise ValueError(f"Feature '{feature_name}' not found")

        logger.info(f"Computing feature: {feature_name}")
        self._validate_input_columns(data, entity_col, timestamp_col)

        result = self._compute_with_parallel_fallback(
            computer, feature_name, data, entity_col, timestamp_col, **kwargs
        )

        # Ensure result is indexed by entity
        if entity_col in result.columns:
            result = result.set_index(entity_col)

        logger.info(f"Computed {len(result)} feature values for {feature_name}")
        return result

    @staticmethod
    def _validate_input_columns(data: pd.DataFrame, entity_col: str, timestamp_col: str) -> None:
        """Raise ValueError if required columns are missing from *data*."""
        missing_cols = [col for col in [entity_col, timestamp_col] if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _compute_with_parallel_fallback(
        self,
        computer: FeatureComputer,
        feature_name: str,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute feature, using parallel execution when appropriate."""
        if self._enable_parallel and len(data) > self._chunk_size:
            try:
                return self._compute_feature_parallel(
                    computer, feature_name, data, entity_col, timestamp_col, **kwargs
                )
            except AstroMLError as e:
                logger.warning(f"Parallel computation failed, falling back to sequential: {e}")
        return self._compute_feature_sequential(
            computer, feature_name, data, entity_col, timestamp_col, **kwargs
        )

    def _compute_feature_sequential(
        self,
        computer: FeatureComputer,
        feature_name: str,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute feature values sequentially.

        Args:
            computer: Feature computation function
            feature_name: Name of feature to compute
            data: Input data
            entity_col: Entity identifier column
            timestamp_col: Timestamp column
            **kwargs: Additional parameters

        Returns:
            DataFrame with computed feature values
        """
        try:
            result = computer(data, entity_col, timestamp_col, **kwargs)
            return result
        except AstroMLError as e:
            logger.error(f"Error computing feature {feature_name}: {e}")
            raise

    def _compute_feature_parallel(
        self,
        computer: FeatureComputer,
        feature_name: str,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute feature values in parallel using chunking.

        Splits the input data into chunks and processes them in parallel
        using ThreadPoolExecutor. Results are combined after all chunks complete.
        """
        chunks = self._split_data_into_chunks(data, entity_col)

        logger.info(
            f"Processing {len(data)} rows in {len(chunks)} chunks with {self._max_workers} workers"
        )

        results = self._process_chunks_parallel(
            chunks, computer, entity_col, timestamp_col, **kwargs
        )

        if results:
            return pd.concat(results, axis=0)
        return pd.DataFrame()

    def _split_data_into_chunks(self, data: pd.DataFrame, entity_col: str) -> list[pd.DataFrame]:
        """Split *data* into chunks of at most ``_chunk_size`` unique entities."""
        unique_entities = data[entity_col].unique()
        chunks: list[pd.DataFrame] = []
        for i in range(0, len(unique_entities), self._chunk_size):
            chunk_entities = unique_entities[i : i + self._chunk_size]
            chunks.append(data[data[entity_col].isin(chunk_entities)].copy())
        return chunks

    def _process_chunks_parallel(
        self,
        chunks: list[pd.DataFrame],
        computer: FeatureComputer,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> list[pd.DataFrame]:
        """Process data chunks in parallel using ThreadPoolExecutor."""

        def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
            try:
                return computer(chunk, entity_col, timestamp_col, **kwargs)
            except AstroMLError as e:
                logger.error(f"Error processing chunk: {e}")
                raise

        results: list[pd.DataFrame] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    results.append(future.result())
                except AstroMLError as e:
                    logger.error(f"Chunk processing failed: {e}")
                    raise

        return results

    def store_feature(
        self,
        feature_name: str,
        values: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
        validate_schema: bool = True,
        dry_run: bool = False,
        auto_register: bool = True,
    ) -> ValidationResult:
        """Store computed feature values.

        Args:
            feature_name: Feature name
            values: Feature values to store
            metadata: Additional metadata
            validate_schema: Whether to validate schema before storing
            dry_run: If True, validate but don't store
            auto_register: Whether to auto-register definition if not found

        Returns:
            ValidationResult if validate_schema=True, otherwise empty ValidationResult
        """
        # Get feature definition
        feature_def = self.storage.get_feature_definition(f"{feature_name}_v1")
        if feature_def is None:
            feature_def = self.storage.get_feature_definition(feature_name)
        if feature_def is None:
            if auto_register:
                feature_def = FeatureDefinition(
                    name=feature_name,
                    description=f"Feature {feature_name}",
                    feature_type=FeatureType.NUMERIC,
                    version=1,
                )
                self.storage.store_feature_definition(feature_def)
            else:
                raise ValueError(f"Feature '{feature_name}' not found")



        # Validate schema if requested
        if validate_schema:
            result = dry_run_ingestion(values, FEATURE_VALUE_SCHEMA, log_issues=True)
            if not result.is_valid and not dry_run:
                logger.error("Schema validation failed, not storing feature")
                return result
        else:
            result = ValidationResult(is_valid=True)

        # Store values if not dry run
        if not dry_run:
            self.storage.store_feature_values(feature_def.feature_id, values, metadata)

            # Invalidate cache entry so the next read fetches fresh data.
            with self._cache_lock:
                if feature_def.feature_id in self._value_cache:
                    cache_entry = self._value_cache.pop(feature_def.feature_id)
                    self._current_cache_size_bytes -= cache_entry.get("size_bytes", 0)
                    self._cache_evictions += 1

            logger.info(f"Stored feature '{feature_name}' with {len(values)} values")
        else:
            logger.info(f"Dry run: would store feature '{feature_name}' with {len(values)} values")

        return result

    def _get_dataframe_size_bytes(self, df: pd.DataFrame) -> int:
        """Estimate DataFrame memory usage in bytes."""
        return int(df.memory_usage(deep=True).sum())

    def _evict_lru_features(self, required_bytes: int) -> None:
        """Evict the least-recently-used entries until *required_bytes* are freed.

        TTLCache performs LRU eviction automatically when *maxsize* is reached,
        so this method is only needed for the soft MB cap.  It iterates over the
        cache in insertion order (oldest first for LRUCache ordering) and removes
        entries until enough space is reclaimed.

        Args:
            required_bytes: Number of bytes to free.
        """
        with self._cache_lock:
            freed_bytes = 0
            # list() snapshot avoids "dictionary changed size" errors during iteration
            for feature_id in list(self._value_cache.keys()):
                if freed_bytes >= required_bytes:
                    break
                cache_entry = self._value_cache.pop(feature_id, None)
                if cache_entry is not None:
                    freed_bytes += cache_entry.get("size_bytes", 0)
                    self._current_cache_size_bytes -= cache_entry.get("size_bytes", 0)
                    self._cache_evictions += 1
                    logger.debug(f"Evicted feature {feature_id} from cache (memory cap)")

    def _is_cache_expired(self, feature_id: str) -> bool:
        """Return True if *feature_id* is absent from the TTLCache (expired or missing).

        TTLCache handles expiry transparently on key access; this helper exists
        for explicit pre-checks without triggering a read.

        Args:
            feature_id: Feature identifier.

        Returns:
            True if the entry has expired or was never cached, False otherwise.
        """
        with self._cache_lock:
            return feature_id not in self._value_cache

    def get_feature(
        self,
        feature_name: str,
        entity_ids: list[str] | None = None,
        timestamp: datetime | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame | None:
        """Retrieve stored feature values with lazy loading.

        Uses lazy loading: only loads feature values on-demand and caches
        recently accessed features.  The underlying :class:`cachetools.TTLCache`
        provides both TTL-based expiry and LRU eviction automatically.
        """
        feature_def = self._resolve_feature_def(feature_name)
        if feature_def is None:
            return None
        feature_id = feature_def.feature_id

        # Cache lookup
        cached_values = self._lookup_cached_values(feature_id, feature_name, entity_ids, use_cache)
        if cached_values is not None:
            return cached_values

        # Cache miss — load from storage
        self._cache_misses += 1
        logger.debug(f"Cache miss for feature '{feature_name}' — loading from storage")
        values = self.storage.get_feature_values(feature_id, entity_ids, timestamp)

        if values is not None and use_cache:
            self._cache_feature_values(feature_id, feature_name, values)

        return values

    def _resolve_feature_def(self, feature_name: str) -> FeatureDefinition | None:
        """Resolve a feature definition, using the metadata cache when possible."""
        if feature_name in self._metadata_cache:
            return self._metadata_cache[feature_name]

        feature_def = self.storage.get_feature_definition(f"{feature_name}_v1")
        if feature_def is None:
            feature_def = self.storage.get_feature_definition(feature_name)
        if feature_def is not None:
            self._metadata_cache[feature_name] = feature_def
        return feature_def

    def _lookup_cached_values(
        self,
        feature_id: str,
        feature_name: str,
        entity_ids: list[str] | None,
        use_cache: bool,
    ) -> pd.DataFrame | None:
        """Try to return cached values. Returns None on cache miss or if caching is disabled."""
        if not use_cache:
            return None

        with self._cache_lock:
            cache_entry = self._value_cache.get(feature_id)

        if cache_entry is None:
            return None

        self._cache_hits += 1
        values = cache_entry["data"].copy()
        logger.debug(f"Cache hit for feature '{feature_name}'")
        if entity_ids:
            values = values[values.index.isin(entity_ids)]
        return values

    def _cache_feature_values(
        self,
        feature_id: str,
        feature_name: str,
        values: pd.DataFrame,
    ) -> None:
        """Insert *values* into the cache, evicting LRU entries if the memory cap is exceeded."""
        value_size_bytes = self._get_dataframe_size_bytes(values)

        with self._cache_lock:
            if self._current_cache_size_bytes + value_size_bytes > self._max_cache_size_bytes:
                required_space = (
                    self._current_cache_size_bytes + value_size_bytes - self._max_cache_size_bytes
                )
                logger.debug(f"Cache at memory cap, freeing {required_space} bytes")
                self._evict_lru_features(required_space)

            self._value_cache[feature_id] = {
                "data": values.copy(),
                "size_bytes": value_size_bytes,
                "loaded_at": datetime.now(),
            }
            self._current_cache_size_bytes += value_size_bytes

        logger.debug(
            f"Cached feature '{feature_name}' "
            f"({value_size_bytes / 1024 / 1024:.2f} MB, "
            f"total: {self._current_cache_size_bytes / 1024 / 1024:.2f} MB)"
        )

    def compute_and_store(
        self,
        feature_name: str,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute and store feature values in one step.

        Args:
            feature_name: Feature name
            data: Input data
            entity_col: Entity identifier column
            timestamp_col: Timestamp column
            metadata: Additional metadata
            **kwargs: Additional parameters

        Returns:
            Computed feature values
        """
        values = self.compute_feature(feature_name, data, entity_col, timestamp_col, **kwargs)
        self.store_feature(feature_name, values, metadata)
        return values

    def create_feature_set(
        self,
        name: str,
        feature_names: list[str],
        description: str,
        entity_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureSet:
        """Create a feature set.

        Args:
            name: Feature set name
            feature_names: List of feature names
            description: Feature set description
            entity_type: Entity type
            metadata: Additional metadata

        Returns:
            Created feature set
        """
        # Get feature IDs
        feature_ids = []
        for feature_name in feature_names:
            feature_def = self.storage.get_feature_definition(f"{feature_name}_v1")
            if feature_def is None:
                raise ValueError(f"Feature '{feature_name}' not found")
            feature_ids.append(feature_def.feature_id)

        feature_set = FeatureSet(
            name=name,
            description=description,
            feature_ids=feature_ids,
            entity_type=entity_type,
            metadata=metadata or {},
        )

        self.storage.store_feature_set(feature_set)
        return feature_set

    def get_feature_set(self, name: str) -> FeatureSet | None:
        """Retrieve feature set.

        Args:
            name: Feature set name

        Returns:
            Feature set if found, None otherwise
        """
        return self.storage.get_feature_set(name)

    def get_features_for_entities(
        self,
        feature_names: list[str],
        entity_ids: list[str],
        timestamp: datetime | None = None,
        parallel: bool = True,
    ) -> pd.DataFrame:
        """Get multiple features for specific entities.

        Args:
            feature_names: List of feature names
            entity_ids: List of entity IDs
            timestamp: Optional timestamp for point-in-time queries
            parallel: Whether to fetch features in parallel

        Returns:
            DataFrame with features indexed by entity
        """
        feature_data: Dict[str, Any] = {}

        if parallel and self._enable_parallel and len(feature_names) > 1:
            feature_data = self._fetch_features_parallel(feature_names, entity_ids, timestamp)
        else:
            feature_data = self._fetch_features_sequential(feature_names, entity_ids, timestamp)

        if not feature_data:
            return pd.DataFrame()

        return pd.DataFrame(feature_data, index=entity_ids)

    @staticmethod
    def _add_feature_values_to_dict(
        feature_data: dict[str, Any],
        feature_name: str,
        values: pd.DataFrame,
    ) -> None:
        """Add feature values to *feature_data* dict, handling multi-column results."""
        if len(values.columns) == 1:
            feature_data[feature_name] = values.iloc[:, 0]
        else:
            for col in values.columns:
                feature_data[f"{feature_name}_{col}"] = values[col]

    def _fetch_features_sequential(
        self,
        feature_names: list[str],
        entity_ids: list[str],
        timestamp: datetime | None,
    ) -> dict[str, Any]:
        """Fetch features one at a time."""
        feature_data: dict[str, Any] = {}
        for feature_name in feature_names:
            values = self.get_feature(feature_name, entity_ids, timestamp)
            if values is not None:
                self._add_feature_values_to_dict(feature_data, feature_name, values)
        return feature_data

    def _fetch_features_parallel(
        self,
        feature_names: list[str],
        entity_ids: list[str],
        timestamp: datetime | None,
    ) -> dict[str, Any]:
        """Fetch features concurrently using ThreadPoolExecutor."""
        feature_data: dict[str, Any] = {}

        def fetch_feature(feature_name: str) -> tuple[str, pd.DataFrame | None]:
            values = self.get_feature(feature_name, entity_ids, timestamp)
            return feature_name, values

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(feature_names))
            ) as executor:
                future_to_feature = {executor.submit(fetch_feature, fn): fn for fn in feature_names}
                for future in concurrent.futures.as_completed(future_to_feature):
                    feature_name = future_to_feature[future]
                    try:
                        fn, values = future.result()
                        if values is not None:
                            self._add_feature_values_to_dict(feature_data, fn, values)
                    except AstroMLError as e:
                        logger.error(f"Failed to fetch feature {feature_name}: {e}")
        except AstroMLError as e:
            logger.warning(f"Parallel fetch failed, falling back to sequential: {e}")
            return self._fetch_features_sequential(feature_names, entity_ids, timestamp)

        return feature_data

    def list_features(
        self,
        status: FeatureStatus | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
    ) -> list[FeatureDefinition]:
        """List available features.

        Args:
            status: Filter by status
            tags: Filter by tags
            owner: Filter by owner

        Returns:
            List of feature definitions
        """
        return self.storage.list_feature_definitions(status, tags, owner)

    def get_online_features(
        self,
        entity_keys: Sequence[str],
        feature_names: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Retrieve low-latency online features for real-time inference."""
        return self.online_store.get_online_features(entity_keys, feature_names)

    def write_online_features(
        self,
        features: list[Any] | pd.DataFrame,
        entity_col: str = "entity_id",
        timestamp_col: str | None = None,
        ttl_seconds: int | None = None,
    ) -> int:
        """Write features to the online feature store."""
        return self.online_store.write_online_features(
            features,
            entity_col=entity_col,
            timestamp_col=timestamp_col,
            ttl_seconds=ttl_seconds,
        )

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_names: Sequence[str],
        entity_col: str = "entity_id",
        timestamp_col: str = "timestamp",
        lookback_seconds: int | None = None,
    ) -> pd.DataFrame:
        """Retrieve point-in-time correct historical features for training datasets."""
        return self.offline_store.get_historical_features(
            entity_df=entity_df,
            feature_names=feature_names,
            entity_col=entity_col,
            timestamp_col=timestamp_col,
            lookback_seconds=lookback_seconds,
        )

    def write_offline_features(
        self,
        feature_name: str,
        df: pd.DataFrame,
        entity_col: str = "entity_id",
        timestamp_col: str = "timestamp",
    ) -> int:
        """Write feature values to offline Parquet storage."""
        return self.offline_store.write_offline_features(
            feature_name=feature_name,
            df=df,
            entity_col=entity_col,
            timestamp_col=timestamp_col,
        )

    def materialize_to_online(
        self,
        feature_names: Sequence[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        ttl_seconds: int | None = None,
    ) -> int:
        """Materialize offline features into the online store for low-latency serving."""
        total_written = 0
        for feat in feature_names:
            df = self.offline_store.read_offline_feature(
                feat, start_time=start_time, end_time=end_time
            )
            if not df.empty:
                count = self.online_store.write_online_features(df, ttl_seconds=ttl_seconds)
                total_written += count
        return total_written

    def get_feature_statistics(
        self,
        feature_name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get monitoring and statistical summary for a feature."""
        return self.offline_store.get_feature_statistics(
            feature_name, start_time=start_time, end_time=end_time
        )

    def clear_cache(self) -> None:
        """Clear all in-process caches (metadata and value) and reset metrics."""
        self._metadata_cache.clear()
        with self._cache_lock:
            self._value_cache.clear()
            self._current_cache_size_bytes = 0
        logger.info("Feature cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Return cache statistics including hit/miss/eviction metrics.

        Returns:
            Dictionary with the following keys:

            * ``cached_features`` – number of entries currently in the cache.
            * ``cache_size_mb`` – estimated memory occupied by cached data.
            * ``max_cache_size_mb`` – configured soft memory cap.
            * ``cache_utilization_pct`` – percentage of soft cap used.
            * ``cache_maxsize`` – maximum number of TTLCache entries (LRU cap).
            * ``cache_ttl_seconds`` – TTL in seconds for each entry.
            * ``metadata_cached`` – number of feature definitions in the
              lightweight metadata cache.
            * ``hits`` – cumulative cache hits since last :meth:`clear_cache`.
            * ``misses`` – cumulative cache misses since last :meth:`clear_cache`.
            * ``evictions`` – cumulative evictions (TTL + memory cap) since
              last :meth:`clear_cache`.
            * ``hit_rate`` – fraction of lookups that were hits (0.0–1.0).
            * ``miss_rate`` – fraction of lookups that were misses (0.0–1.0).
        """
        total_lookups = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_lookups if total_lookups > 0 else 0.0
        miss_rate = self._cache_misses / total_lookups if total_lookups > 0 else 0.0

        with self._cache_lock:
            cached_features = len(self._value_cache)
            current_bytes = self._current_cache_size_bytes

        return {
            "cached_features": cached_features,
            "cache_size_mb": current_bytes / 1024 / 1024,
            "max_cache_size_mb": self._max_cache_size_bytes / 1024 / 1024,
            "cache_utilization_pct": (
                (current_bytes / self._max_cache_size_bytes) * 100
                if self._max_cache_size_bytes > 0
                else 0.0
            ),
            "cache_maxsize": self._cache_maxsize,
            "cache_ttl_seconds": self._cache_ttl_seconds,
            "metadata_cached": len(self._metadata_cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "evictions": self._cache_evictions,
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
        }

    @contextmanager
    def batch_mode(self) -> Any:
        """Context manager for batch operations.

        Clears the cache before and after the batch so that stale entries
        do not bleed across batch boundaries.  Metric counters are also
        reset so per-batch hit/miss rates can be measured independently.
        """
        self.clear_cache()
        # Reset metrics for the new batch window.
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        try:
            yield
        finally:
            self.clear_cache()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def _load_feature_store_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load feature store configuration from YAML.

    Looks for ``config/feature_store.yaml`` relative to the current working
    directory unless *config_path* is given explicitly.  Returns an empty dict
    if the file does not exist so callers can apply safe defaults.

    Args:
        config_path: Explicit path to the YAML file (optional).

    Returns:
        Parsed configuration dict (may be empty).
    """
    import yaml  # only imported when needed to avoid hard dep in minimal envs

    if config_path is None:
        config_path = Path("config") / "feature_store.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        logger.debug(f"Feature store config not found at '{config_path}', using defaults")
        return {}

    with open(config_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    logger.info(f"Loaded feature store config from '{config_path}'")
    return data


def create_feature_store(
    storage_path: str = "./feature_store",
    config_path: str | Path | None = None,
    max_workers: int | None = None,
    chunk_size: int | None = None,
    enable_parallel: bool | None = None,
) -> FeatureStore:
    """Create a :class:`FeatureStore` instance, optionally driven by YAML config.

    Constructor keyword arguments take precedence over values read from
    ``config/feature_store.yaml`` (or *config_path*).

    Args:
        storage_path: Path to feature store storage.
        config_path: Override for the YAML config file location.
        max_workers: Maximum number of parallel workers for feature computation.
            If not provided, reads from config or uses default (4).
        chunk_size: Number of entities to process per chunk in parallel computation.
            If not provided, reads from config or uses default (100).
        enable_parallel: Whether to enable parallel feature computation.
            If not provided, reads from config or uses default (True).

    Returns:
        Configured :class:`FeatureStore` instance.
    """
    cfg = _load_feature_store_config(config_path)
    cache_cfg = cfg.get("cache", {})
    parallel_cfg = cfg.get("parallel", {})

    return FeatureStore(
        storage_path=storage_path,
        max_cache_size_mb=cache_cfg.get("max_size_mb", 500),
        cache_ttl_seconds=cache_cfg.get("ttl_seconds", FeatureStore._DEFAULT_TTL),
        cache_maxsize=cache_cfg.get("maxsize", FeatureStore._DEFAULT_MAXSIZE),
        max_workers=max_workers
        or parallel_cfg.get("max_workers", FeatureStore._DEFAULT_MAX_WORKERS),
        chunk_size=chunk_size or parallel_cfg.get("chunk_size", FeatureStore._DEFAULT_CHUNK_SIZE),
        enable_parallel=(
            enable_parallel if enable_parallel is not None else parallel_cfg.get("enable", True)
        ),
    )


def get_feature_store(
    storage_path: str = "./feature_store",
    config_path: str | Path | None = None,
) -> FeatureStore:
    """Alias for :func:`create_feature_store`.

    Args:
        storage_path: Path to feature store storage.
        config_path: Override for the YAML config file location.

    Returns:
        :class:`FeatureStore` instance.
    """
    return create_feature_store(storage_path, config_path=config_path)
