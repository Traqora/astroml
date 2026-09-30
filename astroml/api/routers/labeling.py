"""Labeling API endpoints (issue #624).

Provides REST endpoints for the automated data labeling pipeline including
active learning queries, weak supervision, human review, and analytics.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from astroml.preprocessing.labeling import (
    ActiveLearner,
    BatchLabelingStrategy,
    ConflictResolver,
    LabelingFunction,
    LabelingPipeline,
    ReviewQueue,
    UncertaintySampling,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Shared in-memory state (replace with DB-backed store in production)
# ---------------------------------------------------------------------------

_lfs: list[LabelingFunction] = []
_pipeline: LabelingPipeline | None = None
_dashboard_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CreateLfRequest(BaseModel):
    """Request body for registering a labeling function."""

    name: str
    description: str = ""


class LfResponse(BaseModel):
    """Serialized labeling function metadata."""

    name: str
    coverage: float
    accuracy: float

    model_config = {"from_attributes": True}


class LabelRequest(BaseModel):
    """Request body for running the labeling pipeline."""

    samples: list[dict[str, Any]] = Field(
        ..., min_length=1, description="List of samples to label"
    )


class LabelResponse(BaseModel):
    """Response returned after labeling a batch of samples."""

    total: int
    auto_accepted: int
    queued_for_review: int
    quality_score: float


class DashboardResponse(BaseModel):
    """Analytics dashboard response."""

    pipeline_stats: dict[str, Any]
    quality_report: dict[str, Any]
    strategy_stats: dict[str, Any]
    auto_accept_rate: float


class ReviewItemResponse(BaseModel):
    """A single review queue item."""

    item_id: str
    conflict_score: float
    priority: float
    status: str
    labels: dict[str, Any]


class ReviewBatchResponse(BaseModel):
    """Batch of review items."""

    items: list[ReviewItemResponse]


class ResolveReviewRequest(BaseModel):
    """Request body for resolving a review item."""

    item_id: str
    resolution: Any


class QualityReportResponse(BaseModel):
    """Per-labeler quality report."""

    labeler_id: str
    total_labels: int
    agreement_with_consensus: float
    avg_confidence: float
    review_rejection_rate: float


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------


def _ensure_pipeline() -> LabelingPipeline:
    """Return (and lazily create) the global labeling pipeline."""
    global _pipeline
    if _pipeline is None:
        strategy = BatchLabelingStrategy(_lfs, majority_only=False)
        _pipeline = LabelingPipeline(
            strategy=strategy,
            review_queue=ReviewQueue(),
            conflict_resolver=ConflictResolver(strategy="majority"),
            auto_accept_threshold=0.85,
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/api/v1/labeling/lfs", response_model=LfResponse, tags=["labeling"])
def register_lf(req: CreateLfRequest) -> LfResponse:
    """Register a labeling function.

    Actual LF logic is placeholder; in production, users would upload
    or reference pre-defined LFs via a registry.
    """
    lf = LabelingFunction(
        name=req.name,
        fn=lambda x: 0,  # placeholder — replace with user-supplied logic
    )
    _lfs.append(lf)
    global _pipeline
    _pipeline = None  # invalidate pipeline
    return LfResponse(name=lf.name, coverage=lf.coverage, accuracy=lf.accuracy)


@router.get("/api/v1/labeling/lfs", response_model=list[LfResponse], tags=["labeling"])
def list_lfs() -> list[LfResponse]:
    """List all registered labeling functions."""
    return [
        LfResponse(name=lf.name, coverage=lf.coverage, accuracy=lf.accuracy)
        for lf in _lfs
    ]


@router.post("/api/v1/labeling/run", response_model=LabelResponse, tags=["labeling"])
def run_labeling(req: LabelRequest) -> LabelResponse:
    """Run the labeling pipeline on a batch of samples."""
    if not _lfs:
        raise HTTPException(status_code=400, detail="No labeling functions registered")

    pipeline = _ensure_pipeline()
    result = pipeline.run(req.samples)

    return LabelResponse(
        total=len(req.samples),
        auto_accepted=pipeline._pipeline_stats["auto_accepted"],
        queued_for_review=pipeline._pipeline_stats["queued_for_review"],
        quality_score=result.quality_score,
    )


@router.get("/api/v1/labeling/dashboard", response_model=DashboardResponse, tags=["labeling"])
def get_dashboard() -> DashboardResponse:
    """Return labeling analytics dashboard."""
    pipeline = _ensure_pipeline()
    data = pipeline.get_dashboard()
    return DashboardResponse(
        pipeline_stats=data["pipeline_stats"],
        quality_report=data["quality_report"],
        strategy_stats=data["strategy_stats"],
        auto_accept_rate=data["auto_accept_rate"],
    )


@router.get(
    "/api/v1/labeling/review-queue",
    response_model=ReviewBatchResponse,
    tags=["labeling"],
)
def get_review_queue(
    batch_size: int = Query(default=10, ge=1, le=100, description="Items per batch"),
) -> ReviewBatchResponse:
    """Return the highest-priority pending review items."""
    pipeline = _ensure_pipeline()
    items = pipeline.review_queue.dequeue_batch(batch_size)
    return ReviewBatchResponse(
        items=[
            ReviewItemResponse(
                item_id=it.item_id,
                conflict_score=it.conflict_score,
                priority=it.priority,
                status=it.status,
                labels=it.labels,
            )
            for it in items
        ]
    )


@router.post("/api/v1/labeling/review-queue/resolve", tags=["labeling"])
def resolve_review_item(req: ResolveReviewRequest) -> dict[str, str]:
    """Resolve a review item with a final label."""
    pipeline = _ensure_pipeline()
    ok = pipeline.review_queue.resolve(req.item_id, req.resolution)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Item {req.item_id!r} not found")
    return {"status": "resolved", "item_id": req.item_id}


@router.get(
    "/api/v1/labeling/quality",
    response_model=list[QualityReportResponse],
    tags=["labeling"],
)
def get_quality_report() -> list[QualityReportResponse]:
    """Return per-labeler quality metrics."""
    pipeline = _ensure_pipeline()
    report = pipeline.review_queue.get_quality_report()
    return [
        QualityReportResponse(
            labeler_id=lid,
            total_labels=data["total_labels"],
            agreement_with_consensus=data["agreement_with_consensus"],
            avg_confidence=data["avg_confidence"],
            review_rejection_rate=data["review_rejection_rate"],
        )
        for lid, data in report.items()
    ]