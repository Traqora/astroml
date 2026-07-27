"""Feature versioning and metadata management for the Feature Store.

This module provides comprehensive versioning, lineage tracking, and metadata
management capabilities for features in the Feature Store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from enum import Enum
import uuid
from contextlib import contextmanager

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """Feature version status."""
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ChangeType(Enum):
    """Types of changes in version history."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RENAME = "rename"
    PARAMETER_CHANGE = "parameter_change"
    DEPENDENCY_CHANGE = "dependency_change"
    CODE_CHANGE = "code_change"


@dataclass
class FeatureVersion:
    """Version information for a feature.
    
    Attributes:
        version_id: Unique version identifier
        feature_name: Feature name
        version: Version number
        status: Version status
        description: Version description
        code_hash: Hash of the computation code
        parameters_hash: Hash of parameters
        data_hash: Hash of input data schema
        created_at: Version creation time
        created_by: Creator
        approved_at: Approval time
        approved_by: Approver
        deployed_at: Deployment time
        metadata: Additional metadata
    """
    
    version_id: str
    feature_name: str
    version: int
    status: VersionStatus
    description: str
    code_hash: str
    parameters_hash: str
    data_hash: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    deployed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        if self.approved_at:
            data["approved_at"] = self.approved_at.isoformat()
        if self.deployed_at:
            data["deployed_at"] = self.deployed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureVersion:
        """Create from dictionary."""
        data = data.copy()
        data["status"] = VersionStatus(data["status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("approved_at"):
            data["approved_at"] = datetime.fromisoformat(data["approved_at"])
        if data.get("deployed_at"):
            data["deployed_at"] = datetime.fromisoformat(data["deployed_at"])
        return cls(**data)


@dataclass
class ChangeRecord:
    """Record of a change in version history.
    
    Attributes:
        change_id: Unique change identifier
        version_id: Version ID
        change_type: Type of change
        description: Change description
        old_value: Previous value (if applicable)
        new_value: New value (if applicable)
        changed_at: Change timestamp
        changed_by: Who made the change
        metadata: Additional metadata
    """
    
    change_id: str
    version_id: str
    change_type: ChangeType
    description: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    changed_at: datetime = field(default_factory=datetime.utcnow)
    changed_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["change_type"] = self.change_type.value
        data["changed_at"] = self.changed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChangeRecord:
        """Create from dictionary."""
        data = data.copy()
        data["change_type"] = ChangeType(data["change_type"])
        data["changed_at"] = datetime.fromisoformat(data["changed_at"])
        return cls(**data)


@dataclass
class FeatureLineage:
    """Lineage information for a feature.
    
    Attributes:
        lineage_id: Unique lineage identifier
        feature_name: Feature name
        upstream_features: List of upstream feature dependencies
        downstream_features: List of downstream dependent features
        data_sources: List of data sources
        transformation_steps: List of transformation steps
        created_at: Lineage creation time
        updated_at: Last update time
        metadata: Additional metadata
    """
    
    lineage_id: str
    feature_name: str
    upstream_features: List[str]
    downstream_features: List[str]
    data_sources: List[str]
    transformation_steps: List[Dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureLineage:
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class FeatureVersionManager:
    """Manages feature versioning and metadata."""
    
    def __init__(self, storage_path: Union[str, Path]):
        """Initialize version manager.
        
        Args:
            storage_path: Path to version storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.storage_path / "feature_versions.db"
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize version database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS feature_versions (
                    version_id TEXT PRIMARY KEY,
                    feature_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    description TEXT,
                    code_hash TEXT NOT NULL,
                    parameters_hash TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT,
                    approved_at TEXT,
                    approved_by TEXT,
                    deployed_at TEXT,
                    metadata TEXT,
                    UNIQUE(feature_name, version)
                );
                
                CREATE TABLE IF NOT EXISTS change_records (
                    change_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    description TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    changed_at TEXT NOT NULL,
                    changed_by TEXT,
                    metadata TEXT,
                    FOREIGN KEY (version_id) REFERENCES feature_versions(version_id)
                );
                
                CREATE TABLE IF NOT EXISTS feature_lineage (
                    lineage_id TEXT PRIMARY KEY,
                    feature_name TEXT NOT NULL,
                    upstream_features TEXT,
                    downstream_features TEXT,
                    data_sources TEXT,
                    transformation_steps TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_feature_versions_name 
                    ON feature_versions(feature_name);
                
                CREATE INDEX IF NOT EXISTS idx_feature_versions_status 
                    ON feature_versions(status);
                
                CREATE INDEX IF NOT EXISTS idx_change_records_version 
                    ON change_records(version_id);
                
                CREATE INDEX IF NOT EXISTS idx_feature_lineage_name 
                    ON feature_lineage(feature_name);
            """)
    
    def create_version(
        self,
        feature_name: str,
        code: str,
        parameters: Dict[str, Any],
        data_schema: Dict[str, Any],
        description: str = "",
        created_by: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeatureVersion:
        """Create a new feature version.
        
        Args:
            feature_name: Feature name
            code: Feature computation code
            parameters: Feature parameters
            data_schema: Input data schema
            description: Version description
            created_by: Creator
            metadata: Additional metadata
            
        Returns:
            Created feature version
        """
        # Get next version number
        latest_version = self.get_latest_version(feature_name)
        next_version = (latest_version.version if latest_version else 0) + 1
        
        # Generate hashes
        code_hash = self._compute_hash(code)
        parameters_hash = self._compute_hash(parameters)
        data_hash = self._compute_hash(data_schema)
        
        # Create version
        version = FeatureVersion(
            version_id=str(uuid.uuid4()),
            feature_name=feature_name,
            version=next_version,
            status=VersionStatus.DRAFT,
            description=description,
            code_hash=code_hash,
            parameters_hash=parameters_hash,
            data_hash=data_hash,
            created_by=created_by,
            metadata=metadata or {},
        )
        
        # Store version
        self._store_version(version)
        
        # Record creation change
        self._record_change(
            version_id=version.version_id,
            change_type=ChangeType.CREATE,
            description=f"Created version {next_version} of {feature_name}",
            changed_by=created_by,
        )
        
        logger.info(f"Created version {next_version} for feature {feature_name}")
        return version
    
    def get_latest_version(self, feature_name: str) -> Optional[FeatureVersion]:
        """Get latest version of a feature.
        
        Args:
            feature_name: Feature name
            
        Returns:
            Latest version if found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM feature_versions 
                WHERE feature_name = ? 
                ORDER BY version DESC 
                LIMIT 1
                """,
                (feature_name,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_version(row)
            return None
    
    def get_version(self, feature_name: str, version: int) -> Optional[FeatureVersion]:
        """Get specific version of a feature.
        
        Args:
            feature_name: Feature name
            version: Version number
            
        Returns:
            Feature version if found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM feature_versions 
                WHERE feature_name = ? AND version = ?
                """,
                (feature_name, version)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_version(row)
            return None
    
    def list_versions(
        self,
        feature_name: Optional[str] = None,
        status: Optional[VersionStatus] = None,
        limit: Optional[int] = None,
    ) -> List[FeatureVersion]:
        """List feature versions.
        
        Args:
            feature_name: Filter by feature name
            status: Filter by status
            limit: Limit number of results
            
        Returns:
            List of feature versions
        """
        query = "SELECT * FROM feature_versions WHERE 1=1"
        params = []
        
        if feature_name:
            query += " AND feature_name = ?"
            params.append(feature_name)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY feature_name, version DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_version(row) for row in rows]
    
    def update_version_status(
        self,
        version_id: str,
        status: VersionStatus,
        updated_by: str = "",
    ) -> bool:
        """Update version status.
        
        Args:
            version_id: Version ID
            status: New status
            updated_by: Who made the update
            
        Returns:
            True if updated successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get current version
            version = self._get_version_by_id(version_id)
            if not version:
                return False
            
            old_status = version.status
            
            # Update status
            updates = {"status": status.value}
            if status == VersionStatus.APPROVED:
                updates["approved_at"] = datetime.utcnow().isoformat()
                updates["approved_by"] = updated_by
            elif status == VersionStatus.DEPLOYED:
                updates["deployed_at"] = datetime.utcnow().isoformat()
            
            set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
            params = list(updates.values()) + [version_id]
            
            conn.execute(
                f"UPDATE feature_versions SET {set_clause} WHERE version_id = ?",
                params
            )
            
            # Record change
            self._record_change(
                version_id=version_id,
                change_type=ChangeType.UPDATE,
                description=f"Changed status from {old_status.value} to {status.value}",
                old_value=old_status.value,
                new_value=status.value,
                changed_by=updated_by,
            )
            
            logger.info(f"Updated version {version_id} status to {status.value}")
            return True
    
    def delete_version(self, version_id: str, deleted_by: str = "") -> bool:
        """Delete a feature version.
        
        Args:
            version_id: Version ID
            deleted_by: Who deleted the version
            
        Returns:
            True if deleted successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get version info before deletion
            version = self._get_version_by_id(version_id)
            if not version:
                return False
            
            # Delete version
            conn.execute("DELETE FROM feature_versions WHERE version_id = ?", (version_id,))
            
            # Record change
            self._record_change(
                version_id=version_id,
                change_type=ChangeType.DELETE,
                description=f"Deleted version {version.version} of {version.feature_name}",
                changed_by=deleted_by,
            )
            
            logger.info(f"Deleted version {version_id}")
            return True
    
    def get_change_history(
        self,
        feature_name: Optional[str] = None,
        version_id: Optional[str] = None,
        change_type: Optional[ChangeType] = None,
        limit: Optional[int] = None,
    ) -> List[ChangeRecord]:
        """Get change history.
        
        Args:
            feature_name: Filter by feature name
            version_id: Filter by version ID
            change_type: Filter by change type
            limit: Limit number of results
            
        Returns:
            List of change records
        """
        query = """
            SELECT cr.* FROM change_records cr
            JOIN feature_versions fv ON cr.version_id = fv.version_id
            WHERE 1=1
        """
        params = []
        
        if feature_name:
            query += " AND fv.feature_name = ?"
            params.append(feature_name)
        
        if version_id:
            query += " AND cr.version_id = ?"
            params.append(version_id)
        
        if change_type:
            query += " AND cr.change_type = ?"
            params.append(change_type.value)
        
        query += " ORDER BY cr.changed_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_change_record(row) for row in rows]
    
    def create_lineage(
        self,
        feature_name: str,
        upstream_features: List[str],
        downstream_features: List[str],
        data_sources: List[str],
        transformation_steps: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeatureLineage:
        """Create feature lineage.
        
        Args:
            feature_name: Feature name
            upstream_features: List of upstream features
            downstream_features: List of downstream features
            data_sources: List of data sources
            transformation_steps: List of transformation steps
            metadata: Additional metadata
            
        Returns:
            Created lineage
        """
        lineage = FeatureLineage(
            lineage_id=str(uuid.uuid4()),
            feature_name=feature_name,
            upstream_features=upstream_features,
            downstream_features=downstream_features,
            data_sources=data_sources,
            transformation_steps=transformation_steps,
            metadata=metadata or {},
        )
        
        self._store_lineage(lineage)
        return lineage
    
    def get_lineage(self, feature_name: str) -> Optional[FeatureLineage]:
        """Get feature lineage.
        
        Args:
            feature_name: Feature name
            
        Returns:
            Feature lineage if found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM feature_lineage WHERE feature_name = ?",
                (feature_name,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_lineage(row)
            return None
    
    def update_lineage(self, lineage: FeatureLineage) -> bool:
        """Update feature lineage.
        
        Args:
            lineage: Lineage to update
            
        Returns:
            True if updated successfully
        """
        lineage.updated_at = datetime.utcnow()
        return self._store_lineage(lineage)
    
    def _compute_hash(self, data: Any) -> str:
        """Compute hash of data.
        
        Args:
            data: Data to hash
            
        Returns:
            Hash string
        """
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _store_version(self, version: FeatureVersion) -> None:
        """Store feature version.
        
        Args:
            version: Version to store
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feature_versions 
                (version_id, feature_name, version, status, description, 
                 code_hash, parameters_hash, data_hash, created_at, created_by,
                 approved_at, approved_by, deployed_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.version_id,
                    version.feature_name,
                    version.version,
                    version.status.value,
                    version.description,
                    version.code_hash,
                    version.parameters_hash,
                    version.data_hash,
                    version.created_at.isoformat(),
                    version.created_by,
                    version.approved_at.isoformat() if version.approved_at else None,
                    version.approved_by,
                    version.deployed_at.isoformat() if version.deployed_at else None,
                    json.dumps(version.metadata),
                ),
            )
    
    def _get_version_by_id(self, version_id: str) -> Optional[FeatureVersion]:
        """Get version by ID.
        
        Args:
            version_id: Version ID
            
        Returns:
            Version if found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM feature_versions WHERE version_id = ?",
                (version_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_version(row)
            return None
    
    def _row_to_version(self, row: Tuple) -> FeatureVersion:
        """Convert database row to FeatureVersion.
        
        Args:
            row: Database row
            
        Returns:
            Feature version
        """
        columns = [
            "version_id", "feature_name", "version", "status", "description",
            "code_hash", "parameters_hash", "data_hash", "created_at", "created_by",
            "approved_at", "approved_by", "deployed_at", "metadata"
        ]
        data = dict(zip(columns, row))
        return FeatureVersion.from_dict(data)
    
    def _record_change(
        self,
        version_id: str,
        change_type: ChangeType,
        description: str,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        changed_by: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a change.
        
        Args:
            version_id: Version ID
            change_type: Type of change
            description: Change description
            old_value: Previous value
            new_value: New value
            changed_by: Who made the change
            metadata: Additional metadata
        """
        change = ChangeRecord(
            change_id=str(uuid.uuid4()),
            version_id=version_id,
            change_type=change_type,
            description=description,
            old_value=json.dumps(old_value) if old_value is not None else None,
            new_value=json.dumps(new_value) if new_value is not None else None,
            changed_by=changed_by,
            metadata=metadata or {},
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO change_records 
                (change_id, version_id, change_type, description, old_value, 
                 new_value, changed_at, changed_by, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    change.change_id,
                    change.version_id,
                    change.change_type.value,
                    change.description,
                    change.old_value,
                    change.new_value,
                    change.changed_at.isoformat(),
                    change.changed_by,
                    json.dumps(change.metadata),
                ),
            )
    
    def _row_to_change_record(self, row: Tuple) -> ChangeRecord:
        """Convert database row to ChangeRecord.
        
        Args:
            row: Database row
            
        Returns:
            Change record
        """
        columns = [
            "change_id", "version_id", "change_type", "description", "old_value",
            "new_value", "changed_at", "changed_by", "metadata"
        ]
        data = dict(zip(columns, row))
        
        # Parse JSON fields
        if data["old_value"]:
            data["old_value"] = json.loads(data["old_value"])
        if data["new_value"]:
            data["new_value"] = json.loads(data["new_value"])
        
        return ChangeRecord.from_dict(data)
    
    def _store_lineage(self, lineage: FeatureLineage) -> bool:
        """Store feature lineage.
        
        Args:
            lineage: Lineage to store
            
        Returns:
            True if stored successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feature_lineage 
                (lineage_id, feature_name, upstream_features, downstream_features,
                 data_sources, transformation_steps, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lineage.lineage_id,
                    lineage.feature_name,
                    json.dumps(lineage.upstream_features),
                    json.dumps(lineage.downstream_features),
                    json.dumps(lineage.data_sources),
                    json.dumps(lineage.transformation_steps),
                    lineage.created_at.isoformat(),
                    lineage.updated_at.isoformat(),
                    json.dumps(lineage.metadata),
                ),
            )
            return True
    
    def _row_to_lineage(self, row: Tuple) -> FeatureLineage:
        """Convert database row to FeatureLineage.
        
        Args:
            row: Database row
            
        Returns:
            Feature lineage
        """
        columns = [
            "lineage_id", "feature_name", "upstream_features", "downstream_features",
            "data_sources", "transformation_steps", "created_at", "updated_at", "metadata"
        ]
        data = dict(zip(columns, row))
        
        # Parse JSON fields
        data["upstream_features"] = json.loads(data["upstream_features"])
        data["downstream_features"] = json.loads(data["downstream_features"])
        data["data_sources"] = json.loads(data["data_sources"])
        data["transformation_steps"] = json.loads(data["transformation_steps"])
        
        return FeatureLineage.from_dict(data)
    
    @contextmanager
    def version_context(self, feature_name: str, created_by: str = ""):
        """Context manager for version operations.
        
        Args:
            feature_name: Feature name
            created_by: Creator
        """
        # Create initial version if needed
        if not self.get_latest_version(feature_name):
            self.create_version(
                feature_name=feature_name,
                code="",
                parameters={},
                data_schema={},
                description="Initial version",
                created_by=created_by,
            )
        
        try:
            yield self
        finally:
            # Cleanup if needed
            pass


# Convenience functions

def create_version_manager(storage_path: str = "./feature_versions") -> FeatureVersionManager:
    """Create a feature version manager.
    
    Args:
        storage_path: Path to version storage
        
    Returns:
        Version manager instance
    """
    return FeatureVersionManager(storage_path)


def compute_feature_hash(feature_def: Dict[str, Any]) -> str:
    """Compute hash of feature definition.
    
    Args:
        feature_def: Feature definition
        
    Returns:
        Feature hash
    """
    # Sort keys for consistent hashing
    sorted_def = json.dumps(feature_def, sort_keys=True)
    return hashlib.sha256(sorted_def.encode()).hexdigest()
