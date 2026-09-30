from astroml.utils.exceptions import AstroMLError
"""Data versioning API router for AstroML.

Provides REST endpoints for dataset versioning, tagging, comparison,
and DVC operations (push/pull/status).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from astroml.storage.data_versioning import DataVersionControl

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["data-versioning"])

_dvc = DataVersionControl()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class AddDatasetRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    path: str = Field(..., min_length=1)
    version: str = Field(default="latest")
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TagRequest(BaseModel):
    tags: list[str] = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class AnnotateRequest(BaseModel):
    annotations: dict[str, str] = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class DatasetVersionResponse(BaseModel):
    version_id: str
    name: str
    version: str
    path: str
    dvc_hash: str | None = None
    description: str
    tags: list[str]
    annotations: dict[str, str]
    size_bytes: int
    num_files: int
    created_at: str

    model_config = ConfigDict(extra="forbid")


class VersionDiffResponse(BaseModel):
    version_a: str
    version_b: str
    added_files: list[str]
    removed_files: list[str]
    modified_files: list[str]
    size_diff_bytes: int
    summary: str

    model_config = ConfigDict(extra="forbid")


class MessageResponse(BaseModel):
    message: str
    success: bool = True

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/data-versioning/datasets",
    response_model=DatasetVersionResponse,
    status_code=201,
    summary="Add and version a new dataset",
)
async def add_dataset(request: AddDatasetRequest) -> DatasetVersionResponse:
    try:
        ver = _dvc.add_dataset(
            name=request.name,
            path=request.path,
            version=request.version,
            description=request.description,
            tags=request.tags,
        )
        return _ver_to_response(ver)
    except AstroMLError as e:
        logger.exception("Error adding dataset")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/data-versioning/datasets",
    response_model=list[DatasetVersionResponse],
    summary="List dataset versions",
)
async def list_datasets(
    name: str | None = Query(default=None),
    tag: list[str] | None = Query(default=None),
) -> list[DatasetVersionResponse]:
    try:
        versions = _dvc.list_versions(name=name, tags=tag)
        return [_ver_to_response(v) for v in versions]
    except AstroMLError as e:
        logger.exception("Error listing datasets")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/data-versioning/datasets/{version_id}",
    response_model=DatasetVersionResponse,
    summary="Get a dataset version",
)
async def get_dataset(version_id: str) -> DatasetVersionResponse:
    ver = _dvc.get_version(version_id)
    if ver is None:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    return _ver_to_response(ver)


# ---------------------------------------------------------------------------
# Tagging and annotations
# ---------------------------------------------------------------------------


@router.post(
    "/data-versioning/datasets/{version_id}/tags",
    response_model=DatasetVersionResponse,
    summary="Add tags to a dataset version",
)
async def add_tags(
    version_id: str, request: TagRequest
) -> DatasetVersionResponse:
    try:
        ver = _dvc.tag_version(version_id, request.tags)
        return _ver_to_response(ver)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AstroMLError as e:
        logger.exception("Error adding tags")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/data-versioning/datasets/{version_id}/annotations",
    response_model=DatasetVersionResponse,
    summary="Add annotations to a dataset version",
)
async def add_annotations(
    version_id: str, request: AnnotateRequest
) -> DatasetVersionResponse:
    try:
        ver = _dvc.annotate(version_id, request.annotations)
        return _ver_to_response(ver)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AstroMLError as e:
        logger.exception("Error adding annotations")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@router.get(
    "/data-versioning/compare",
    response_model=VersionDiffResponse,
    summary="Compare two dataset versions",
)
async def compare_versions(
    version_id_a: str = Query(...),
    version_id_b: str = Query(...),
) -> VersionDiffResponse:
    try:
        diff = _dvc.compare_versions(version_id_a, version_id_b)
        return VersionDiffResponse(
            version_a=diff.version_a,
            version_b=diff.version_b,
            added_files=diff.added_files,
            removed_files=diff.removed_files,
            modified_files=diff.modified_files,
            size_diff_bytes=diff.size_diff_bytes,
            summary=diff.summary,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AstroMLError as e:
        logger.exception("Error comparing versions")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# DVC operations
# ---------------------------------------------------------------------------


@router.post(
    "/data-versioning/push",
    summary="Push data to DVC remote",
)
async def dvc_push() -> dict[str, Any]:
    return _dvc.push()


@router.post(
    "/data-versioning/pull",
    summary="Pull data from DVC remote",
)
async def dvc_pull() -> dict[str, Any]:
    return _dvc.pull()


@router.get(
    "/data-versioning/status",
    summary="Get DVC repository status",
)
async def dvc_status() -> dict[str, Any]:
    return _dvc.status()


@router.get(
    "/data-versioning/datasets/{version_id}/snapshot",
    summary="Get a version snapshot",
)
async def get_snapshot(version_id: str) -> dict[str, Any]:
    try:
        return _dvc.snapshot(version_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AstroMLError as e:
        logger.exception("Error getting snapshot")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/data-versioning/datasets/{version_id}/export",
    summary="Export version snapshot as JSON",
)
async def export_dataset(version_id: str) -> dict[str, Any]:
    try:
        import json as json_mod

        data = _dvc.snapshot(version_id)
        return {"data": data}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AstroMLError as e:
        logger.exception("Error exporting dataset")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ver_to_response(ver) -> DatasetVersionResponse:
    return DatasetVersionResponse(
        version_id=ver.version_id,
        name=ver.name,
        version=ver.version,
        path=ver.path,
        dvc_hash=ver.dvc_hash,
        description=ver.description,
        tags=ver.tags,
        annotations=ver.annotations,
        size_bytes=ver.size_bytes,
        num_files=ver.num_files,
        created_at=ver.created_at,
    )