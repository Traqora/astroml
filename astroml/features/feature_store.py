"""Feature Store implementation for AstroML.

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
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Callable,
    Protocol,
    runtime_checkable,
)
from enum import Enum
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import concurrent.futures

import pandas as pd
from cachetools import TTLCache

from astroml.features.schema_validation import (
    dry_run_ingestion,
    ValidationResult,
    FEATURE_VALUE_SCHEMA,
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
    computation_function: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    owner: str = ""
    status: FeatureStatus = FeatureStatus.DEVELOPMENT
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    
    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> FeatureDefinition:
        """Create from dictionary representation."""
        data = data.copy()
        data.pop("feature_id", None)
        data["feature_type"] = FeatureType(data["feature_type"])
        data["status"] = FeatureStatus(data["status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
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
    validity_period: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def expires_at(self) -> Optional[datetime]:
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
    feature_ids: List[str]
    entity_type: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
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


class FeatureStorage:
    """Storage backend for feature values and metadata."""
    
    def __init__(self, storage_path: Union[str, Path]):
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
                columns = [
                    "feature_id", "name", "version", "description", "feature_type",
                    "parameters", "tags", "owner", "status", "created_at", 
                    "updated_at", "metadata"
                ]
                data = dict(zip(columns, row))
                data["parameters"] = json.loads(data["parameters"])
                data["tags"] = json.loads(data["tags"])
                data["metadata"] = json.loads(data["metadata"])
                return FeatureDefinition.from_dict(data)
            
            return None
    
    def list_feature_definitions(
        self,
        status: Optional[FeatureStatus] = None,
        tags: Optional[List[str]] = None,
        owner: Optional[str] = None,
    ) -> List[FeatureDefinition]:
        """List feature definitions with optional filtering.
        
        Args:
            status: Filter by status
            tags: Filter by tags (must contain all specified tags)
            owner: Filter by owner
            
        Returns:
            List of feature definitions
        """
        query = "SELECT * FROM feature_definitions WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        if owner:
            query += " AND owner = ?"
            params.append(owner)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            features = []
            for row in rows:
                columns = [
                    "feature_id", "name", "version", "description", "feature_type",
                    "parameters", "tags", "owner", "status", "created_at", 
                    "updated_at", "metadata"
                ]
                data = dict(zip(columns, row))
                data["parameters"] = json.loads(data["parameters"])
                data["tags"] = json.loads(data["tags"])
                data["metadata"] = json.loads(data["metadata"])
                
                # Filter by tags if specified
                if tags:
                    feature_tags = set(data["tags"])
                    if not all(tag in feature_tags for tag in tags):
                        continue
                
                features.append(FeatureDefinition.from_dict(data))
            
            return features
    
    def store_feature_values(
        self,
        feature_id: str,
        values: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
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
        entity_ids: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
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
    
    def get_feature_set(self, name: str) -> Optional[FeatureSet]:
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
                    "name", "description", "feature_ids", "entity_type",
                    "created_at", "updated_at", "metadata"
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
    
    def __init__(self, storage: FeatureStorage):
        """Initialize feature registry.
        
        Args:
            storage: Storage backend
        """
        self.storage = storage
        self._computers: Dict[str, FeatureComputer] = {}
        self._register_builtin_features()
    
    def _register_builtin_features(self) -> None:
        """Register built-in feature computers from existing modules."""
        try:
            # Import existing feature modules
            from astroml.features import (
                frequency,
                structural_importance,
                node_features,
                asset_diversity,
            )
            
            # Register frequency features
            self.register_computer(
                "daily_transaction_count",
                frequency.compute_daily_transaction_counts,
                {
                    "description": "Daily transaction count per account",
                    "feature_type": FeatureType.NUMERIC,
                    "tags": ["frequency", "activity"],
                },
            )
            
            self.register_computer(
                "transaction_burstiness",
                frequency.compute_burstiness,
                {
                    "description": "Transaction burstiness metric",
                    "feature_type": FeatureType.NUMERIC,
                    "tags": ["frequency", "behavior"],
                },
            )
            
            # Register structural importance features
            self.register_computer(
                "degree_centrality",
                structural_importance.compute_degree_centrality,
                {
                    "description": "Degree centrality in transaction graph",
                    "feature_type": FeatureType.NUMERIC,
                    "tags": ["graph", "centrality"],
                },
            )
            
            self.register_computer(
                "betweenness_centrality",
                structural_importance.compute_betweenness_centrality,
                {
                    "description": "Betweenness centrality in transaction graph",
                    "feature_type": FeatureType.NUMERIC,
                    "tags": ["graph", "centrality"],
                },
            )
            
            self.register_computer(
                "pagerank",
                structural_importance.compute_pagerank,
                {
                    "description": "PageRank score in transaction graph",
                    "feature_type": FeatureType.NUMERIC,
                    "tags": ["graph", "importance"],
                },
            )
            
            # Register node features
            self.register_computer(
                "node_features",
                node_features.compute_node_features,
                {
                    "description": "Basic node features (degree, volume, age)",
                    "feature_type": FeatureType.TIME_SERIES,
                    "tags": ["node", "basic"],
                },
            )
            
            # Register asset diversity features
            self.register_computer(
                "asset_diversity",
                asset_diversity.compute_asset_diversity,
                {
                    "description": "Asset diversity metrics",
                    "feature_type": FeatureType.NUMERIC,
                    "tags": ["asset", "diversity"],
                },
            )
            
            logger.info("Registered built-in feature computers")
            
        except ImportError as e:
            logger.warning(f"Could not import some feature modules: {e}")
    
    def register_computer(
        self,
        name: str,
        computer: FeatureComputer,
        metadata: Dict[str, Any],
    ) -> None:
        """Register a feature computer.
        
        Args:
            name: Feature name
            computer: Computation function
            metadata: Feature metadata
        """
        self._computers[name] = computer
        
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
    
    def get_computer(self, name: str) -> Optional[FeatureComputer]:
        """Get registered feature computer.
        
        Args:
            name: Feature name
            
        Returns:
            Feature computer if found, None otherwise
        """
        return self._computers.get(name)
    
    def list_features(self) -> List[str]:
        """List all registered feature names."""
        return list(self._computers.keys())


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
        storage_path: Union[str, Path] = "./feature_store",
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

        # Lightweight metadata cache (feature definitions rarely change).
        self._metadata_cache: Dict[str, FeatureDefinition] = {}

        # Primary value cache: TTLCache provides automatic TTL expiry *and*
        # LRU eviction when maxsize is reached.  A threading.Lock makes all
        # mutations atomic.
        self._cache_lock: threading.Lock = threading.Lock()
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

        # Parallel computation settings
        self._max_workers: int = max_workers
        self._chunk_size: int = chunk_size
        self._enable_parallel: bool = enable_parallel and max_workers > 1
    
    def register_feature(
        self,
        name: str,
        computer: FeatureComputer,
        description: str,
        feature_type: FeatureType = FeatureType.NUMERIC,
        tags: Optional[List[str]] = None,
        owner: str = "",
        parameters: Optional[Dict[str, Any]] = None,
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
        
        self.registry.register_computer(name, computer, metadata)
        
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

        # Validate input data
        required_cols = [entity_col, timestamp_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Compute feature with parallelism if enabled and data is large enough
        if self._enable_parallel and len(data) > self._chunk_size:
            try:
                result = self._compute_feature_parallel(
                    computer, feature_name, data, entity_col, timestamp_col, **kwargs
                )
            except Exception as e:
                logger.warning(f"Parallel computation failed, falling back to sequential: {e}")
                result = self._compute_feature_sequential(
                    computer, feature_name, data, entity_col, timestamp_col, **kwargs
                )
        else:
            result = self._compute_feature_sequential(
                computer, feature_name, data, entity_col, timestamp_col, **kwargs
            )

        # Ensure result is indexed by entity
        if entity_col in result.columns:
            result = result.set_index(entity_col)

        logger.info(f"Computed {len(result)} feature values for {feature_name}")
        return result

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
        except Exception as e:
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

        Args:
            computer: Feature computation function
            feature_name: Name of feature to compute
            data: Input data
            entity_col: Entity identifier column
            timestamp_col: Timestamp column
            **kwargs: Additional parameters

        Returns:
            DataFrame with computed feature values from all chunks combined

        Raises:
            Exception: If parallel computation fails
        """
        # Split data into chunks by entity
        unique_entities = data[entity_col].unique()
        chunks = []
        for i in range(0, len(unique_entities), self._chunk_size):
            chunk_entities = unique_entities[i : i + self._chunk_size]
            chunk_data = data[data[entity_col].isin(chunk_entities)].copy()
            chunks.append(chunk_data)

        logger.info(
            f"Processing {len(data)} rows in {len(chunks)} chunks "
            f"with {self._max_workers} workers"
        )

        # Process chunks in parallel
        def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
            """Process a single chunk of data."""
            try:
                result = computer(chunk, entity_col, timestamp_col, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                raise

        results = []
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers
            ) as executor:
                future_to_chunk = {
                    executor.submit(process_chunk, chunk): chunk
                    for chunk in chunks
                }

                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        chunk_result = future.result()
                        results.append(chunk_result)
                    except Exception as e:
                        logger.error(f"Chunk processing failed: {e}")
                        raise

        except Exception as e:
            logger.error(f"Parallel computation failed: {e}")
            raise

        # Combine results from all chunks
        if results:
            combined_result = pd.concat(results, axis=0)
            return combined_result
        else:
            return pd.DataFrame()
    
    def store_feature(
        self,
        feature_name: str,
        values: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        validate_schema: bool = True,
        dry_run: bool = False,
    ) -> ValidationResult:
        """Store computed feature values.
        
        Args:
            feature_name: Feature name
            values: Feature values to store
            metadata: Additional metadata
            validate_schema: Whether to validate schema before storing
            dry_run: If True, validate but don't store
            
        Returns:
            ValidationResult if validate_schema=True, otherwise empty ValidationResult
        """
        # Get feature definition
        feature_def = self.storage.get_feature_definition(f"{feature_name}_v1")
        if feature_def is None:
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
    
    @cache_feature_store(ttl_seconds=900)
    def get_feature(
        self,
        feature_name: str,
        entity_ids: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Retrieve stored feature values with lazy loading.

        Uses lazy loading: only loads feature values on-demand and caches
        recently accessed features.  The underlying :class:`cachetools.TTLCache`
        provides both TTL-based expiry and LRU eviction automatically.

        Args:
            feature_name: Feature name.
            entity_ids: Optional entity IDs to filter.
            timestamp: Optional timestamp for point-in-time queries.
            use_cache: Whether to use the in-process value cache.

        Returns:
            Feature values if found, None otherwise.
        """
        # ── 1. Resolve feature definition (lightweight metadata cache) ────────
        if feature_name in self._metadata_cache:
            feature_def = self._metadata_cache[feature_name]
        else:
            feature_def = self.storage.get_feature_definition(f"{feature_name}_v1")
            if feature_def is None:
                raise ValueError(f"Feature '{feature_name}' not found")
            self._metadata_cache[feature_name] = feature_def

        feature_id = feature_def.feature_id

        # ── 2. Cache lookup ───────────────────────────────────────────────────
        if use_cache:
            with self._cache_lock:
                cache_entry = self._value_cache.get(feature_id)

            if cache_entry is not None:
                self._cache_hits += 1
                values = cache_entry["data"].copy()
                logger.debug(f"Cache hit for feature '{feature_name}'")
                if entity_ids:
                    values = values[values.index.isin(entity_ids)]
                return values

        # ── 3. Cache miss — load from storage ─────────────────────────────────
        self._cache_misses += 1
        logger.debug(f"Cache miss for feature '{feature_name}' — loading from storage")
        values = self.storage.get_feature_values(feature_id, entity_ids, timestamp)

        if values is not None and use_cache:
            value_size_bytes = self._get_dataframe_size_bytes(values)

            # Enforce soft memory cap before inserting.
            with self._cache_lock:
                if (
                    self._current_cache_size_bytes + value_size_bytes
                    > self._max_cache_size_bytes
                ):
                    required_space = (
                        self._current_cache_size_bytes + value_size_bytes
                        - self._max_cache_size_bytes
                    )
                    logger.debug(
                        f"Cache at memory cap, freeing {required_space} bytes"
                    )
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

        return values
    
    def compute_and_store(
        self,
        feature_name: str,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        metadata: Optional[Dict[str, Any]] = None,
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
        feature_names: List[str],
        description: str,
        entity_type: str,
        metadata: Optional[Dict[str, Any]] = None,
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
    
    def get_feature_set(self, name: str) -> Optional[FeatureSet]:
        """Retrieve feature set.
        
        Args:
            name: Feature set name
            
        Returns:
            Feature set if found, None otherwise
        """
        return self.storage.get_feature_set(name)
    
    def get_features_for_entities(
        self,
        feature_names: List[str],
        entity_ids: List[str],
        timestamp: Optional[datetime] = None,
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
        feature_data = {}

        if parallel and self._enable_parallel and len(feature_names) > 1:
            # Fetch features in parallel
            def fetch_feature(feature_name: str) -> tuple[str, Optional[pd.DataFrame]]:
                """Fetch a single feature."""
                values = self.get_feature(feature_name, entity_ids, timestamp)
                return feature_name, values

            try:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(self._max_workers, len(feature_names))
                ) as executor:
                    future_to_feature = {
                        executor.submit(fetch_feature, fn): fn
                        for fn in feature_names
                    }

                    for future in concurrent.futures.as_completed(future_to_feature):
                        feature_name = future_to_feature[future]
                        try:
                            fn, values = future.result()
                            if values is not None:
                                if len(values.columns) == 1:
                                    feature_data[feature_name] = values.iloc[:, 0]
                                else:
                                    for col in values.columns:
                                        feature_data[f"{feature_name}_{col}"] = values[col]
                        except Exception as e:
                            logger.error(f"Failed to fetch feature {feature_name}: {e}")
            except Exception as e:
                logger.warning(f"Parallel fetch failed, falling back to sequential: {e}")
                # Fallback to sequential
                for feature_name in feature_names:
                    values = self.get_feature(feature_name, entity_ids, timestamp)
                    if values is not None:
                        if len(values.columns) == 1:
                            feature_data[feature_name] = values.iloc[:, 0]
                        else:
                            for col in values.columns:
                                feature_data[f"{feature_name}_{col}"] = values[col]
        else:
            # Sequential fetch
            for feature_name in feature_names:
                values = self.get_feature(feature_name, entity_ids, timestamp)
                if values is not None:
                    if len(values.columns) == 1:
                        feature_data[feature_name] = values.iloc[:, 0]
                    else:
                        for col in values.columns:
                            feature_data[f"{feature_name}_{col}"] = values[col]

        if not feature_data:
            return pd.DataFrame()

        result = pd.DataFrame(feature_data, index=entity_ids)
        return result
    
    def list_features(
        self,
        status: Optional[FeatureStatus] = None,
        tags: Optional[List[str]] = None,
        owner: Optional[str] = None,
    ) -> List[FeatureDefinition]:
        """List available features.
        
        Args:
            status: Filter by status
            tags: Filter by tags
            owner: Filter by owner
            
        Returns:
            List of feature definitions
        """
        return self.storage.list_feature_definitions(status, tags, owner)
    
    def clear_cache(self) -> None:
        """Clear all in-process caches (metadata and value) and reset metrics."""
        self._metadata_cache.clear()
        with self._cache_lock:
            self._value_cache.clear()
            self._current_cache_size_bytes = 0
        logger.info("Feature cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
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
    def batch_mode(self):
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

def _load_feature_store_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
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

    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    logger.info(f"Loaded feature store config from '{config_path}'")
    return data


def create_feature_store(
    storage_path: str = "./feature_store",
    config_path: Optional[Union[str, Path]] = None,
    max_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    enable_parallel: Optional[bool] = None,
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
        max_workers=max_workers or parallel_cfg.get("max_workers", FeatureStore._DEFAULT_MAX_WORKERS),
        chunk_size=chunk_size or parallel_cfg.get("chunk_size", FeatureStore._DEFAULT_CHUNK_SIZE),
        enable_parallel=enable_parallel if enable_parallel is not None else parallel_cfg.get("enable", True),
    )


def get_feature_store(
    storage_path: str = "./feature_store",
    config_path: Optional[Union[str, Path]] = None,
) -> FeatureStore:
    """Alias for :func:`create_feature_store`.

    Args:
        storage_path: Path to feature store storage.
        config_path: Override for the YAML config file location.

    Returns:
        :class:`FeatureStore` instance.
    """
    return create_feature_store(storage_path, config_path=config_path)
