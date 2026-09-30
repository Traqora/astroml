"""In-memory metadata store for lineage tracking.

Provides storage and retrieval of lineage metadata using Pydantic models
and dictionary-backed storage with an ORM-like pattern.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


def _sanitize_log(value: str) -> str:
    """Remove newline characters to prevent log injection.

    Args:
        value: Raw string from user input.

    Returns:
        String with newlines replaced by spaces.
    """
    return value.replace("\n", " ").replace("\r", " ")


class MetadataRecord(BaseModel):
    """A single lineage metadata record.

    Attributes:
        id: Unique identifier for the record.
        type: Type of record (dataset, transformation, model).
        timestamp: When the record was created.
        metadata: Arbitrary key-value metadata.
        parent_ids: IDs of parent/input entities.
        child_ids: IDs of child/output entities.
    """

    id: str
    type: str = Field(pattern=r"^(dataset|transformation|model)$")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_ids: list[str] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class MetadataStore:
    """Stores and queries lineage metadata records.

    Uses in-memory dictionaries keyed by record ID, entity type,
    and timestamp for efficient queries.
    """

    def __init__(self) -> None:
        self._records: dict[str, MetadataRecord] = {}
        self._by_type: dict[str, dict[str, MetadataRecord]] = {
            "dataset": {},
            "transformation": {},
            "model": {},
        }

    def store_run(self, run_id: str, metadata: dict[str, Any] | None = None) -> MetadataRecord:
        """Store a run-level metadata record.

        Args:
            run_id: Unique identifier for the run.
            metadata: Optional metadata to attach.

        Returns:
            The created MetadataRecord.
        """
        record = MetadataRecord(
            id=run_id,
            type="dataset",
            metadata=metadata or {},
        )
        self._records[run_id] = record
        self._by_type["dataset"][run_id] = record
        logger.debug("Stored run record: %s", _sanitize_log(run_id))
        return record

    def get_run(self, run_id: str) -> MetadataRecord | None:
        """Retrieve a run record by ID.

        Args:
            run_id: The run identifier.

        Returns:
            The MetadataRecord if found, None otherwise.
        """
        return self._records.get(run_id)

    def store_dataset(
        self,
        dataset_id: str,
        metadata: dict[str, Any] | None = None,
        parent_ids: list[str] | None = None,
    ) -> MetadataRecord:
        """Store a dataset metadata record.

        Args:
            dataset_id: Unique identifier for the dataset.
            metadata: Optional metadata.
            parent_ids: Optional list of parent entity IDs.

        Returns:
            The created MetadataRecord.
        """
        record = MetadataRecord(
            id=dataset_id,
            type="dataset",
            metadata=metadata or {},
            parent_ids=parent_ids or [],
        )
        self._records[dataset_id] = record
        self._by_type["dataset"][dataset_id] = record
        self._update_relationships(record)
        logger.debug("Stored dataset record: %s", _sanitize_log(dataset_id))
        return record

    def get_dataset(self, dataset_id: str) -> MetadataRecord | None:
        """Retrieve a dataset record by ID.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            The MetadataRecord if found, None otherwise.
        """
        return self._records.get(dataset_id)

    def store_transformation(
        self,
        transformation_id: str,
        metadata: dict[str, Any] | None = None,
        parent_ids: list[str] | None = None,
        child_ids: list[str] | None = None,
    ) -> MetadataRecord:
        """Store a transformation metadata record.

        Args:
            transformation_id: Unique identifier for the transformation.
            metadata: Optional metadata.
            parent_ids: Optional list of input entity IDs.
            child_ids: Optional list of output entity IDs.

        Returns:
            The created MetadataRecord.
        """
        record = MetadataRecord(
            id=transformation_id,
            type="transformation",
            metadata=metadata or {},
            parent_ids=parent_ids or [],
            child_ids=child_ids or [],
        )
        self._records[transformation_id] = record
        self._by_type["transformation"][transformation_id] = record
        self._update_relationships(record)
        logger.debug("Stored transformation record: %s", _sanitize_log(transformation_id))
        return record

    def get_transformations(
        self,
        entity_id: str | None = None,
    ) -> list[MetadataRecord]:
        """Retrieve transformation records, optionally filtered by entity.

        Args:
            entity_id: Optional entity ID to filter by. If provided, returns
                transformations that reference this entity as parent or child.

        Returns:
            List of matching MetadataRecords.
        """
        if entity_id is None:
            return list(self._by_type["transformation"].values())
        return [
            r
            for r in self._by_type["transformation"].values()
            if entity_id in r.parent_ids or entity_id in r.child_ids
        ]

    def store_model(
        self,
        model_id: str,
        metadata: dict[str, Any] | None = None,
        parent_ids: list[str] | None = None,
        child_ids: list[str] | None = None,
    ) -> MetadataRecord:
        """Store a model metadata record.

        Args:
            model_id: Unique identifier for the model.
            metadata: Optional metadata.
            parent_ids: Optional list of parent entity IDs.
            child_ids: Optional list of child entity IDs.

        Returns:
            The created MetadataRecord.
        """
        record = MetadataRecord(
            id=model_id,
            type="model",
            metadata=metadata or {},
            parent_ids=parent_ids or [],
            child_ids=child_ids or [],
        )
        self._records[model_id] = record
        self._by_type["model"][model_id] = record
        self._update_relationships(record)
        logger.debug("Stored model record: %s", _sanitize_log(model_id))
        return record

    def get_model(self, model_id: str) -> MetadataRecord | None:
        """Retrieve a model record by ID.

        Args:
            model_id: The model identifier.

        Returns:
            The MetadataRecord if found, None otherwise.
        """
        return self._records.get(model_id)

    def query_lineage(
        self,
        entity_id: str | None = None,
        entity_type: str | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
    ) -> list[MetadataRecord]:
        """Query lineage records with optional filters.

        Args:
            entity_id: Optional entity ID filter.
            entity_type: Optional entity type filter (dataset, transformation, model).
            time_start: Optional start of time range.
            time_end: Optional end of time range.

        Returns:
            List of matching MetadataRecords.
        """
        results: list[MetadataRecord] = []

        if entity_id:
            record = self._records.get(entity_id)
            if record:
                results = [record]
            return results

        if entity_type:
            records = list(self._by_type.get(entity_type, {}).values())
        else:
            records = list(self._records.values())

        for r in records:
            if time_start is not None and r.timestamp < time_start:
                continue
            if time_end is not None and r.timestamp > time_end:
                continue
            results.append(r)

        return results

    def _update_relationships(self, record: MetadataRecord) -> None:
        """Update parent/child cross-references when storing a record.

        Args:
            record: The record whose relationships to propagate.
        """
        for pid in record.parent_ids:
            parent = self._records.get(pid)
            if parent and record.id not in parent.child_ids:
                parent.child_ids.append(record.id)

        for cid in record.child_ids:
            child = self._records.get(cid)
            if child and record.id not in child.parent_ids:
                child.parent_ids.append(record.id)
