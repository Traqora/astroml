"""Data lineage tracker for recording provenance through pipeline stages.

Tracks how data flows through sources, transformations, model training,
and predictions, building a complete lineage DAG.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from datetime import datetime
from typing import Any

from astroml.tracking.lineage.metadata_store import MetadataRecord, MetadataStore

logger = logging.getLogger(__name__)


def _sanitize_log(value: str) -> str:
    """Remove newline characters to prevent log injection.

    Args:
        value: Raw string from user input.

    Returns:
        String with newlines replaced by spaces.
    """
    return value.replace("\n", " ").replace("\r", " ")


class DataLineageTracker:
    """Records data provenance through pipeline stages.

    Uses a MetadataStore internally to track data sources, transformations,
    model training, and predictions as nodes in a lineage DAG.
    """

    def __init__(self, metadata_store: MetadataStore | None = None) -> None:
        """Initialize the lineage tracker.

        Args:
            metadata_store: An optional MetadataStore instance. Creates a new
                one if not provided.
        """
        self._store = metadata_store or MetadataStore()

    def record_source(
        self,
        source_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> MetadataRecord:
        """Record a data source.

        Args:
            source_id: Unique identifier for the data source.
            metadata: Optional metadata about the source.

        Returns:
            The created MetadataRecord.
        """
        record = self._store.store_dataset(source_id, metadata=metadata)
        logger.info("Recorded data source: %s", _sanitize_log(source_id))
        return record

    def record_transformation(
        self,
        input_ids: list[str],
        output_id: str,
        transform_name: str,
        params: dict[str, Any] | None = None,
    ) -> MetadataRecord:
        """Record a transformation step.

        Args:
            input_ids: IDs of input entities (datasets or other transformations).
            output_id: ID of the output entity.
            transform_name: Name of the transformation applied.
            params: Optional parameters of the transformation.

        Returns:
            The created MetadataRecord.
        """
        metadata = {
            "transform_name": transform_name,
            "params": params or {},
        }
        record = self._store.store_transformation(
            transformation_id=output_id,
            metadata=metadata,
            parent_ids=input_ids,
        )
        logger.info(
            "Recorded transformation: %s -> %s",
            [_sanitize_log(i) for i in input_ids],
            _sanitize_log(output_id),
        )
        return record

    def record_model_training(
        self,
        model_id: str,
        dataset_id: str,
        model_metadata: dict[str, Any] | None = None,
    ) -> MetadataRecord:
        """Record model training lineage.

        Args:
            model_id: Unique identifier for the trained model.
            dataset_id: ID of the dataset used for training.
            model_metadata: Optional metadata about the training run.

        Returns:
            The created MetadataRecord.
        """
        metadata = dict(model_metadata or {})
        metadata["training_dataset_id"] = dataset_id
        record = self._store.store_model(
            model_id=model_id,
            metadata=metadata,
            parent_ids=[dataset_id],
        )
        logger.info(
            "Recorded model training: %s <- %s",
            _sanitize_log(model_id),
            _sanitize_log(dataset_id),
        )
        return record

    def record_prediction(
        self,
        model_id: str,
        input_id: str,
        prediction_id: str,
        prediction_metadata: dict[str, Any] | None = None,
    ) -> MetadataRecord:
        """Record a prediction lineage entry.

        Args:
            model_id: ID of the model that made the prediction.
            input_id: ID of the input data used for prediction.
            prediction_id: Unique identifier for the prediction.
            prediction_metadata: Optional metadata about the prediction.

        Returns:
            The created MetadataRecord.
        """
        metadata = {
            "prediction": True,
            "model_id": model_id,
            **(prediction_metadata or {}),
        }
        record = self._store.store_dataset(
            dataset_id=prediction_id,
            metadata=metadata,
            parent_ids=[input_id, model_id],
        )
        logger.info(
            "Recorded prediction: %s (model=%s, input=%s)",
            _sanitize_log(prediction_id),
            _sanitize_log(model_id),
            _sanitize_log(input_id),
        )
        return record

    def get_lineage(
        self,
        entity_id: str,
        entity_type: str | None = None,
    ) -> dict[str, Any]:
        """Return the full lineage DAG for an entity.

        Builds a dictionary containing the entity record, all upstream
        dependencies, and all downstream dependencies.

        Args:
            entity_id: ID of the entity to query.
            entity_type: Optional type hint for the entity.

        Returns:
            Dict with keys: entity, upstream, downstream, full_dag.
        """
        entity = (
            self._store.get_run(entity_id)
            or self._store.get_dataset(entity_id)
            or self._store.get_model(entity_id)
        )
        if entity is None:
            return {}

        upstream = self.get_upstream(entity_id, entity_type)
        downstream = self.get_downstream(entity_id, entity_type)

        full_dag = _build_dag(
            entity=entity,
            upstream=upstream,
            downstream=downstream,
        )

        return {
            "entity": _record_to_dict(entity),
            "upstream": [_record_to_dict(r) for r in upstream],
            "downstream": [_record_to_dict(r) for r in downstream],
            "full_dag": full_dag,
        }

    def get_upstream(
        self,
        entity_id: str,
        entity_type: str | None = None,
    ) -> list[MetadataRecord]:
        """Return all upstream dependencies for an entity.

        Traverses parent links to find all transitive dependencies.

        Args:
            entity_id: ID of the entity.
            entity_type: Optional type hint (unused, kept for API compat).

        Returns:
            List of upstream MetadataRecords (excluding the entity itself).
        """
        visited: set[str] = set()
        upstream: list[MetadataRecord] = []
        _traverse_upstream(self._store, entity_id, visited, upstream)
        return [r for r in upstream if r.id != entity_id]

    def get_downstream(
        self,
        entity_id: str,
        entity_type: str | None = None,
    ) -> list[MetadataRecord]:
        """Return all downstream dependencies for an entity.

        Traverses child links to find all transitive dependents.

        Args:
            entity_id: ID of the entity.
            entity_type: Optional type hint (unused, kept for API compat).

        Returns:
            List of downstream MetadataRecords (excluding the entity itself).
        """
        visited: set[str] = set()
        downstream: list[MetadataRecord] = []
        _traverse_downstream(self._store, entity_id, visited, downstream)
        return [r for r in downstream if r.id != entity_id]


def _record_to_dict(record: MetadataRecord) -> dict[str, Any]:
    """Convert a MetadataRecord to a dictionary.

    Args:
        record: The record to convert.

    Returns:
        Dictionary representation.
    """
    return {
        "id": record.id,
        "type": record.type,
        "timestamp": record.timestamp.isoformat(),
        "metadata": record.metadata,
        "parent_ids": record.parent_ids,
        "child_ids": record.child_ids,
    }


def _get_any_record(
    store: MetadataStore,
    entity_id: str,
) -> MetadataRecord | None:
    """Get a record by ID from any type category.

    Args:
        store: MetadataStore to query.
        entity_id: The entity identifier.

    Returns:
        The MetadataRecord if found, None otherwise.
    """
    return (
        store.get_dataset(entity_id)
        or store.get_model(entity_id)
        or _get_transformation_by_id(store, entity_id)
    )


def _get_transformation_by_id(
    store: MetadataStore,
    entity_id: str,
) -> MetadataRecord | None:
    """Get a transformation record by ID.

    Args:
        store: MetadataStore to query.
        entity_id: The transformation identifier.

    Returns:
        The MetadataRecord if found, None otherwise.
    """
    for r in store.get_transformations():
        if r.id == entity_id:
            return r
    return None


def _traverse_upstream(
    store: MetadataStore,
    entity_id: str,
    visited: set[str],
    results: list[MetadataRecord],
) -> None:
    """Recursively traverse upstream (parent) links.

    Args:
        store: MetadataStore to query.
        entity_id: Current entity ID.
        visited: Set of already-visited IDs to prevent cycles.
        results: Accumulator for found records.
    """
    if entity_id in visited:
        return
    visited.add(entity_id)

    record = _get_any_record(store, entity_id)
    if record is None:
        return

    results.append(record)
    for pid in record.parent_ids:
        _traverse_upstream(store, pid, visited, results)


def _traverse_downstream(
    store: MetadataStore,
    entity_id: str,
    visited: set[str],
    results: list[MetadataRecord],
) -> None:
    """Recursively traverse downstream (child) links.

    Args:
        store: MetadataStore to query.
        entity_id: Current entity ID.
        visited: Set of already-visited IDs to prevent cycles.
        results: Accumulator for found records.
    """
    if entity_id in visited:
        return
    visited.add(entity_id)

    record = _get_any_record(store, entity_id)
    if record is None:
        return

    results.append(record)
    for cid in record.child_ids:
        _traverse_downstream(store, cid, visited, results)


def _build_dag(
    entity: MetadataRecord,
    upstream: list[MetadataRecord],
    downstream: list[MetadataRecord],
) -> dict[str, Any]:
    """Build a DAG representation from lineage data.

    Args:
        entity: The root entity.
        upstream: List of upstream entities.
        downstream: List of downstream entities.

    Returns:
        Nested dict DAG representation.
    """
    dag: dict[str, Any] = {
        "nodes": {},
        "edges": [],
    }

    def _add_node(record: MetadataRecord) -> None:
        nid = record.id
        if nid not in dag["nodes"]:
            dag["nodes"][nid] = {
                "id": nid,
                "type": record.type,
                "label": nid,
            }

    def _add_edges(record: MetadataRecord) -> None:
        for pid in record.parent_ids:
            edge = {"source": pid, "target": record.id}
            if edge not in dag["edges"]:
                dag["edges"].append(edge)
        for cid in record.child_ids:
            edge = {"source": record.id, "target": cid}
            if edge not in dag["edges"]:
                dag["edges"].append(edge)

    _add_node(entity)
    _add_edges(entity)

    for r in upstream:
        _add_node(r)
        _add_edges(r)
    for r in downstream:
        _add_node(r)
        _add_edges(r)

    return dag


# ---------------------------------------------------------------------------
# Training Lineage Data Structures
# ---------------------------------------------------------------------------


@dataclass
class TrainingLineage:
    """Tracks end-to-end model training provenance."""

    dataset_id: str
    dataset_version: str = "latest"
    dataset_hash: str | None = None
    code_repository: str | None = None
    commit_hash: str | None = None
    branch: str | None = None
    pipeline_run_id: str | None = None
    parent_model_id: str | None = None
    parent_version: str | None = None
    hyperparameters: dict[str, Any] = None
    environment: dict[str, str] = None
    artifact_hashes: dict[str, str] = None
    created_at: str = None

    def __post_init__(self) -> None:
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.environment is None:
            self.environment = {}
        if self.artifact_hashes is None:
            self.artifact_hashes = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingLineage:
        known = {
            "dataset_id", "dataset_version", "dataset_hash", "code_repo", "code_repository",
            "commit_hash", "branch", "pipeline_run_id", "parent_model_id", "parent_version",
            "hyperparameters", "environment", "artifact_hashes", "created_at"
        }
        filtered = {k: v for k, v in data.items() if k in known}
        if "code_repo" in filtered and "code_repository" not in filtered:
            filtered["code_repository"] = filtered.pop("code_repo")
        return cls(**filtered)


@dataclass
class ModelLineage:
    """Full lineage record for a model version including upstream and downstream nodes."""

    model_name: str
    version: str
    training_lineage: TrainingLineage
    upstream_nodes: list[str] = None
    downstream_nodes: list[str] = None

    def __post_init__(self) -> None:
        if self.upstream_nodes is None:
            self.upstream_nodes = [self.training_lineage.dataset_id]
        if self.downstream_nodes is None:
            self.downstream_nodes = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "training_lineage": self.training_lineage.to_dict(),
            "upstream_nodes": self.upstream_nodes,
            "downstream_nodes": self.downstream_nodes,
        }
