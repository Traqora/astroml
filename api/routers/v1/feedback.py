"""Feedback collection router (issue #308).

Collects in-app feedback (bug / feature / general), optionally opens a GitHub
issue, supports admin review (list + status updates), and exposes a public
roadmap derived from feedback status.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models.orm import Feedback
from api.schemas import (
    FeedbackIn,
    FeedbackListResponse,
    FeedbackOut,
    FeedbackStatusUpdate,
    RoadmapResponse,
    ROADMAP_STATUSES,
)
from api.services.github import create_feedback_issue

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackOut, status_code=201)
async def submit_feedback(
    payload: FeedbackIn,
    db: AsyncSession = Depends(get_db),
) -> Feedback:
    """Create a feedback item and (best-effort) open a GitHub issue."""
    issue_url = None
    try:
        issue_url = await create_feedback_issue(
            payload.category, payload.message, payload.email
        )
    except Exception:  # pragma: no cover - defensive; integration is best-effort
        issue_url = None

    feedback = Feedback(
        category=payload.category,
        message=payload.message,
        email=payload.email,
        screenshot=payload.screenshot,
        status="open",
        github_issue_url=issue_url,
    )
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    return feedback


@router.get("", response_model=FeedbackListResponse)
async def list_feedback(
    status: Optional[str] = None,
    category: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> FeedbackListResponse:
    """Admin: list feedback with optional status/category filters."""
    query = select(Feedback)
    count_query = select(func.count()).select_from(Feedback)
    if status:
        query = query.where(Feedback.status == status)
        count_query = count_query.where(Feedback.status == status)
    if category:
        query = query.where(Feedback.category == category)
        count_query = count_query.where(Feedback.category == category)

    total = (await db.execute(count_query)).scalar_one()
    query = query.order_by(Feedback.created_at.desc()).limit(page_size).offset(
        (page - 1) * page_size
    )
    rows = (await db.execute(query)).scalars().all()
    return FeedbackListResponse(
        data=[FeedbackOut.model_validate(r) for r in rows],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.get("/roadmap", response_model=RoadmapResponse)
async def roadmap(db: AsyncSession = Depends(get_db)) -> RoadmapResponse:
    """Public roadmap: feedback grouped by planned / in_progress / completed."""
    query = (
        select(Feedback)
        .where(Feedback.status.in_(ROADMAP_STATUSES))
        .order_by(Feedback.created_at.desc())
    )
    rows = (await db.execute(query)).scalars().all()
    grouped: dict[str, list] = {s: [] for s in ROADMAP_STATUSES}
    for r in rows:
        grouped[r.status].append(r)
    return RoadmapResponse(
        planned=grouped["planned"],
        in_progress=grouped["in_progress"],
        completed=grouped["completed"],
    )


@router.patch("/{feedback_id}", response_model=FeedbackOut)
async def update_feedback_status(
    feedback_id: int,
    payload: FeedbackStatusUpdate,
    db: AsyncSession = Depends(get_db),
) -> Feedback:
    """Admin: update a feedback item's status (drives the roadmap)."""
    feedback = (
        await db.execute(select(Feedback).where(Feedback.id == feedback_id))
    ).scalar_one_or_none()
    if feedback is None:
        raise HTTPException(status_code=404, detail="Feedback not found")
    feedback.status = payload.status
    await db.commit()
    await db.refresh(feedback)
    return feedback
