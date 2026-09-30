"""Schema registry for CDC data evolution (issue #626).

Manages schema versions, migrations, and compatibility checks for CDC
event data as the source database schema evolves over time.

Components:
- SchemaRegistry: Central schema version management
- SchemaVersion: Immutable schema snapshot at a point in time
- SchemaMigration: Forward-compatible schema migration rules
- CompatibilityMode: BACKWARD, FORWARD, FULL compatibility policies
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CompatibilityMode(str, Enum):
    """Schema compatibility policy."""

    BACKWARD = "backward"  # New schema can read old data
    FORWARD = "forward"  # Old schema can read new data
    FULL = "full"  # Both backward and forward compatible
    NONE = "none"  # No compatibility checks


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaVersion:
    """Immutable snapshot of a table schema at a specific version.

    Attributes:
        table_name: Fully qualified table name.
        version: Schema version number (monotonic).
        fields: Ordered list of field definitions ``{"name": str, "type": str}``.
        primary_key: Primary key column(s).
        created_at: When this schema version was registered.
        checksum: SHA-256 of serialized fields (for dedup).
    """

    table_name: str
    version: int
    fields: list[dict[str, str]]
    primary_key: list[str] = field(default_factory=lambda: ["id"])
    created_at: datetime = field(default_factory=datetime.utcnow)
    checksum: str = ""

    def __post_init__(self) -> Any:
        if not self.checksum:
            import hashlib

            raw = json.dumps(self.fields, sort_keys=True)
            object.__setattr__(self, "checksum", hashlib.sha256(raw.encode()).hexdigest()[:16])

    def get_field_names(self) -> set[str]:
        """Return the set of field names."""
        return {f["name"] for f in self.fields}

    def get_field_type(self, name: str) -> str | None:
        """Return the type of a field by name."""
        for f in self.fields:
            if f["name"] == name:
                return f["type"]
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the version, including the checksum it was minted with.

        Returns:
            The schema's fields in registry form. ``created_at`` is an
                ISO-8601 string and ``checksum`` is the SHA-256 prefix computed
                from ``fields`` at construction, so two versions that compare
                equal here describe the same shape. ``fields`` is shared, not
                copied — treat the result as read-only.
        """
        return {
            "table_name": self.table_name,
            "version": self.version,
            "fields": self.fields,
            "primary_key": self.primary_key,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
        }


@dataclass
class SchemaMigration:
    """A migration rule from one schema version to the next.

    Attributes:
        table_name: Target table.
        from_version: Source schema version.
        to_version: Target schema version.
        add_fields: Fields to add ``{"name": str, "type": str, "default": Any}``.
        drop_fields: Fields to drop.
        rename_fields: Field rename mapping ``{old: new}``.
        transform_fields: Field transformation callables ``{field: fn}``.
    """

    table_name: str
    from_version: int
    to_version: int
    add_fields: list[dict[str, Any]] = field(default_factory=list)
    drop_fields: list[str] = field(default_factory=list)
    rename_fields: dict[str, str] = field(default_factory=dict)
    transform_fields: dict[str, Any] = field(default_factory=dict)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply migration rules to a data row.

        Args:
            data: Data in *from_version* format.

        Returns:
            Data in *to_version* format.
        """
        result = dict(data)

        # Drop fields
        for col in self.drop_fields:
            result.pop(col, None)

        # Rename fields
        for old, new in self.rename_fields.items():
            if old in result:
                result[new] = result.pop(old)

        # Transform fields
        for col, fn in self.transform_fields.items():
            if col in result:
                result[col] = fn(result[col])

        # Add fields
        for col in self.add_fields:
            if col["name"] not in result:
                result[col["name"]] = col.get("default")

        return result


# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------


class SchemaRegistry:
    """Central registry for managing CDC schema versions and evolution.

    Supports registering schema versions, checking compatibility,
    and migrating data between versions.

    Args:
        compatibility_mode: Default compatibility checking policy.
        max_versions_per_table: Maximum schema versions retained.
    """

    def __init__(
        self,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARD,
        max_versions_per_table: int = 100,
    ) -> None:
        self.compatibility_mode = compatibility_mode
        self.max_versions_per_table = max_versions_per_table
        self._schemas: dict[str, list[SchemaVersion]] = {}
        self._migrations: dict[str, list[SchemaMigration]] = {}

    def register_schema(
        self,
        table_name: str,
        fields: list[dict[str, str]],
        primary_key: list[str] | None = None,
    ) -> SchemaVersion:
        """Register a new schema version for a table.

        Args:
            table_name: Qualified table name.
            fields: Field definitions.
            primary_key: Primary key columns (defaults to ``["id"]``).

        Returns:
            The registered :class:`SchemaVersion`.

        Raises:
            ValueError: If the new schema violates the compatibility policy.
        """
        pk = primary_key or ["id"]
        existing = self._schemas.get(table_name, [])
        version = len(existing) + 1

        new_schema = SchemaVersion(
            table_name=table_name,
            version=version,
            fields=fields,
            primary_key=pk,
        )

        if existing and self.compatibility_mode != CompatibilityMode.NONE:
            self._check_compatibility(existing[-1], new_schema)

        if table_name not in self._schemas:
            self._schemas[table_name] = []
        self._schemas[table_name].append(new_schema)

        if len(self._schemas[table_name]) > self.max_versions_per_table:
            self._schemas[table_name] = self._schemas[table_name][-self.max_versions_per_table:]

        logger.info(
            "Registered schema %s:v%d (%d fields, checksum=%s)",
            table_name,
            version,
            len(fields),
            new_schema.checksum,
        )
        return new_schema

    def register_migration(self, migration: SchemaMigration) -> None:
        """Register a migration rule between two schema versions.

        Args:
            migration: The migration definition.
        """
        if migration.table_name not in self._migrations:
            self._migrations[migration.table_name] = []
        self._migrations[migration.table_name].append(migration)
        logger.info(
            "Registered migration %s: v%d → v%d",
            migration.table_name,
            migration.from_version,
            migration.to_version,
        )

    def get_current_schema(self, table_name: str) -> SchemaVersion | None:
        """Return the latest schema version for a table."""
        versions = self._schemas.get(table_name, [])
        return versions[-1] if versions else None

    def get_schema_version(self, table_name: str, version: int) -> SchemaVersion | None:
        """Return a specific schema version."""
        versions = self._schemas.get(table_name, [])
        if 1 <= version <= len(versions):
            return versions[version - 1]
        return None

    def migrate_data(
        self,
        table_name: str,
        data: dict[str, Any],
        from_version: int,
        to_version: int,
    ) -> dict[str, Any]:
        """Migrate a data row between schema versions.

        Args:
            table_name: Table name.
            data: Data in *from_version* format.
            from_version: Source schema version.
            to_version: Target schema version.

        Returns:
            Data migrated to *to_version* format.

        Raises:
            ValueError: If migration path is impossible.
        """
        if from_version == to_version:
            return data

        migs = self._migrations.get(table_name, [])
        result = dict(data)

        if from_version < to_version:
            for v in range(from_version, to_version):
                mig = self._find_migration(migs, v, v + 1)
                if mig:
                    result = mig.apply(result)
        else:
            for v in range(from_version, to_version, -1):
                mig = self._find_migration(migs, v - 1, v)
                if mig:
                    result = self._reverse_apply(mig, result)

        return result

    def _find_migration(
        self,
        migrations: list[SchemaMigration],
        from_v: int,
        to_v: int,
    ) -> SchemaMigration | None:
        for m in migrations:
            if m.from_version == from_v and m.to_version == to_v:
                return m
        return None

    def _reverse_apply(self, migration: SchemaMigration, data: dict[str, Any]) -> dict[str, Any]:
        """Reverse-apply a migration (for downgrade)."""
        result = dict(data)
        for col in migration.add_fields:
            result.pop(col["name"], None)
        for old, new in migration.rename_fields.items():
            if new in result:
                result[old] = result.pop(new)
        return result

    def _check_compatibility(self, old: SchemaVersion, new: SchemaVersion) -> None:
        """Verify the new schema is compatible with the old one."""
        old_names = old.get_field_names()
        new_names = new.get_field_names()

        if self.compatibility_mode in (CompatibilityMode.BACKWARD, CompatibilityMode.FULL):
            # New schema can read old data: cannot remove required fields
            for name in old_names - new_names:
                logger.warning(
                    "BACKWARD compatibility: field %s removed from %s:v%d",
                    name,
                    new.table_name,
                    new.version,
                )

        if self.compatibility_mode in (CompatibilityMode.FORWARD, CompatibilityMode.FULL):
            # Old schema can read new data: cannot add required fields
            for name in new_names - old_names:
                logger.warning(
                    "FORWARD compatibility: field %s added in %s:v%d",
                    name,
                    new.table_name,
                    new.version,
                )

    def validate_data(
        self,
        table_name: str,
        data: dict[str, Any],
        version: int | None = None,
    ) -> list[str]:
        """Validate that data conforms to a schema version.

        Args:
            table_name: Table name.
            data: Data row to validate.
            version: Schema version (default: latest).

        Returns:
            List of validation error messages (empty = valid).
        """
        schema = (
            self.get_schema_version(table_name, version)
            if version is not None
            else self.get_current_schema(table_name)
        )
        if schema is None:
            return [f"No schema found for {table_name}"]

        errors: list[str] = []
        expected = schema.get_field_names()

        for field_def in schema.fields:
            name = field_def["name"]
            if name not in data and name not in schema.primary_key:
                pass  # field may be omitted in CDC partial updates
            elif name in data:
                expected_type = field_def.get("type", "string")
                actual = data[name]
                if not _type_check(actual, expected_type):
                    errors.append(
                        f"Field {name}: expected {expected_type}, got {type(actual).__name__}"
                    )

        return errors

    def get_all_schemas(self) -> dict[str, list[dict[str, Any]]]:
        """Return all registered schemas as dicts."""
        return {
            table: [s.to_dict() for s in versions]
            for table, versions in self._schemas.items()
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _type_check(value: Any, expected: str) -> bool:
    """Basic type check for schema validation."""
    type_map: dict[str, type] = {
        "string": str,
        "int": int,
        "integer": int,
        "bigint": int,
        "float": float,
        "double": float,
        "boolean": bool,
        "bool": bool,
    }
    py_type = type_map.get(expected.lower())
    if py_type is None:
        return True  # unknown type — skip strict check
    return isinstance(value, py_type)