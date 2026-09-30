from astroml.utils.exceptions import AstroMLError
"""Data lineage API router for AstroML.

Provides REST endpoints for querying and recording lineage and provenance
information about datasets, transformations, and models.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from astroml.tracking.lineage.data_lineage import DataLineageTracker
from astroml.tracking.lineage.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


def _sanitize_log(value: str) -> str:
    """Remove newlines from user-supplied strings before logging.

    Args:
        value: Raw string from user input.

    Returns:
        String with newlines replaced by spaces.
    """
    return value.replace("\n", " ").replace("\r", " ")


router = APIRouter(prefix="/api/v1", tags=["data-lineage"])

# ---------------------------------------------------------------------------
# Shared tracker instances
# ---------------------------------------------------------------------------

_lineage_tracker = DataLineageTracker()
_provenance_tracker = ProvenanceTracker()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RecordLineageRequest(BaseModel):
    """Request body for recording a lineage event."""

    event_type: str = Field(..., pattern=r"^(source|transformation|training|prediction)$")
    entity_id: str
    parent_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RecordLineageResponse(BaseModel):
    """Response after recording a lineage event."""

    success: bool
    entity_id: str
    message: str

    model_config = ConfigDict(extra="forbid")


class LineageEntity(BaseModel):
    """A single entity in a lineage response."""

    id: str
    type: str
    timestamp: str
    metadata: dict[str, Any]
    parent_ids: list[str]
    child_ids: list[str]

    model_config = ConfigDict(extra="forbid")


class LineageResponse(BaseModel):
    """Full lineage DAG response."""

    entity: LineageEntity | None = None
    upstream: list[LineageEntity]
    downstream: list[LineageEntity]
    full_dag: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class ProvenanceResponse(BaseModel):
    """Provenance details response."""

    run_id: str
    stages: list[dict[str, Any]]
    created_at: str

    model_config = ConfigDict(extra="forbid")


class ExportResponse(BaseModel):
    """Exported lineage data response."""

    data: dict[str, Any] | str

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/lineage/{entity_type}/{entity_id}",
    response_model=LineageResponse,
    summary="Get full lineage DAG for an entity",
)
async def get_lineage(
    entity_type: str,
    entity_id: str,
) -> LineageResponse:
    """Return the full lineage DAG including upstream and downstream dependencies."""
    try:
        data = _lineage_tracker.get_lineage(entity_id, entity_type)
        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Entity {entity_type}/{entity_id} not found",
            )
        return _build_lineage_response(data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Error fetching lineage for %s/%s", _sanitize_log(entity_type), _sanitize_log(entity_id)
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/lineage/{entity_type}/{entity_id}/upstream",
    response_model=list[LineageEntity],
    summary="Get upstream dependencies",
)
async def get_upstream(
    entity_type: str,
    entity_id: str,
) -> list[LineageEntity]:
    """Return all upstream dependencies for an entity."""
    try:
        records = _lineage_tracker.get_upstream(entity_id, entity_type)
        if not records:
            raise HTTPException(
                status_code=404,
                detail=f"No upstream dependencies found for {entity_type}/{entity_id}",
            )
        return [
            LineageEntity(
                id=r.id,
                type=r.type,
                timestamp=r.timestamp.isoformat(),
                metadata=r.metadata,
                parent_ids=r.parent_ids,
                child_ids=r.child_ids,
            )
            for r in records
        ]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Error fetching upstream for %s/%s",
            _sanitize_log(entity_type),
            _sanitize_log(entity_id),
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/lineage/{entity_type}/{entity_id}/downstream",
    response_model=list[LineageEntity],
    summary="Get downstream dependencies",
)
async def get_downstream(
    entity_type: str,
    entity_id: str,
) -> list[LineageEntity]:
    """Return all downstream dependencies for an entity."""
    try:
        records = _lineage_tracker.get_downstream(entity_id, entity_type)
        if not records:
            raise HTTPException(
                status_code=404,
                detail=f"No downstream dependencies found for {entity_type}/{entity_id}",
            )
        return [
            LineageEntity(
                id=r.id,
                type=r.type,
                timestamp=r.timestamp.isoformat(),
                metadata=r.metadata,
                parent_ids=r.parent_ids,
                child_ids=r.child_ids,
            )
            for r in records
        ]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Error fetching downstream for %s/%s",
            _sanitize_log(entity_type),
            _sanitize_log(entity_id),
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/lineage/{entity_type}/{entity_id}/provenance",
    response_model=ProvenanceResponse,
    summary="Get provenance details for a run",
)
async def get_provenance(
    entity_type: str,
    entity_id: str,
) -> ProvenanceResponse:
    """Return provenance details for a run entity."""
    try:
        chain = _provenance_tracker._chains.get(entity_id)
        if chain is None:
            raise HTTPException(
                status_code=404,
                detail=f"Provenance not found for {entity_id}",
            )
        return ProvenanceResponse(
            run_id=chain.run_id,
            stages=[s.to_dict() for s in chain.stages],
            created_at=chain.created_at.isoformat(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error fetching provenance for %s", _sanitize_log(entity_id))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/lineage/{entity_type}/{entity_id}/export",
    response_model=ExportResponse,
    summary="Export lineage data",
)
async def export_lineage(
    entity_type: str,
    entity_id: str,
    fmt: str = Query(default="json", pattern=r"^(json|dict)$"),
) -> ExportResponse:
    """Export lineage data in the requested format."""
    try:
        data = _lineage_tracker.get_lineage(entity_id, entity_type)
        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Entity {entity_type}/{entity_id} not found",
            )
        if fmt == "json":
            import json as json_mod

            return ExportResponse(data=json_mod.dumps(data, indent=2, default=str))
        return ExportResponse(data=data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Error exporting lineage for %s/%s",
            _sanitize_log(entity_type),
            _sanitize_log(entity_id),
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/lineage/record",
    response_model=RecordLineageResponse,
    status_code=201,
    summary="Record a lineage event",
)
async def record_lineage(
    request: RecordLineageRequest,
) -> RecordLineageResponse:
    """Record a lineage event (source, transformation, training, prediction)."""
    try:
        if request.event_type == "source":
            _lineage_tracker.record_source(
                source_id=request.entity_id,
                metadata=request.metadata,
            )
        elif request.event_type == "transformation":
            _lineage_tracker.record_transformation(
                input_ids=request.parent_ids,
                output_id=request.entity_id,
                transform_name=request.metadata.get("transform_name", "unknown"),
                params=request.metadata.get("params"),
            )
        elif request.event_type == "training":
            parent_ids = request.parent_ids
            dataset_id = parent_ids[0] if parent_ids else "unknown"
            _lineage_tracker.record_model_training(
                model_id=request.entity_id,
                dataset_id=dataset_id,
                model_metadata=request.metadata,
            )
        elif request.event_type == "prediction":
            parent_ids = request.parent_ids
            model_id = parent_ids[0] if parent_ids else "unknown"
            input_id = parent_ids[1] if len(parent_ids) > 1 else "unknown"
            _lineage_tracker.record_prediction(
                model_id=model_id,
                input_id=input_id,
                prediction_id=request.entity_id,
                prediction_metadata=request.metadata,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown event_type: {request.event_type}",
            )

        return RecordLineageResponse(
            success=True,
            entity_id=request.entity_id,
            message=f"Recorded {request.event_type} event for {request.entity_id}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error recording lineage event")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_lineage_response(data: dict[str, Any]) -> LineageResponse:
    """Build a LineageResponse from raw lineage data.

    Args:
        data: Raw lineage data dict.

    Returns:
        A populated LineageResponse.
    """
    entity_raw = data.get("entity")
    entity = None
    if entity_raw:
        entity = LineageEntity(
            id=entity_raw["id"],
            type=entity_raw["type"],
            timestamp=entity_raw["timestamp"],
            metadata=entity_raw["metadata"],
            parent_ids=entity_raw["parent_ids"],
            child_ids=entity_raw["child_ids"],
        )

    upstream = [
        LineageEntity(
            id=r["id"],
            type=r["type"],
            timestamp=r["timestamp"],
            metadata=r["metadata"],
            parent_ids=r["parent_ids"],
            child_ids=r["child_ids"],
        )
        for r in data.get("upstream", [])
    ]

    downstream = [
        LineageEntity(
            id=r["id"],
            type=r["type"],
            timestamp=r["timestamp"],
            metadata=r["metadata"],
            parent_ids=r["parent_ids"],
            child_ids=r["child_ids"],
        )
        for r in data.get("downstream", [])
    ]

    return LineageResponse(
        entity=entity,
        upstream=upstream,
        downstream=downstream,
        full_dag=data.get("full_dag", {}),
    )
