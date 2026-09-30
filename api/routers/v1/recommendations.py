"""Recommendations router for issue #474.

Provides endpoints for:
- Getting personalized recommendations
- Recording user activities
- Recording feedback on recommendations
- Getting feedback statistics
"""

from __future__ import annotations

from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from astroml.llm.recommendations import (
    recommendation_engine,
    UserProfile,
    UserRole,
    ActivityType,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


# ─── Request/Response Schemas ─────────────────────────────────────────────


class ActivityRequest(BaseModel):
    """Request schema for recording user activity."""

    user_id: str = Field(..., description="User identifier")
    activity_type: str = Field(..., description="Activity type")
    page: str = Field(..., description="Page where activity occurred")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class FeedbackRequest(BaseModel):
    """Request schema for recording feedback."""

    user_id: str = Field(..., description="User identifier")
    recommendation_id: str = Field(..., description="Recommendation identifier")
    accepted: bool = Field(..., description="Whether the recommendation was accepted")


class RecommendationsRequest(BaseModel):
    """Request schema for getting recommendations."""

    user_id: str = Field(..., description="User identifier")
    role: str = Field(default="viewer", description="User role")
    current_page: Optional[str] = Field(None, description="Current page")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results")


# ─── Recommendations Endpoints ─────────────────────────────────────────────


@router.post("/get", response_model=List[dict])
async def get_recommendations(request: RecommendationsRequest):
    """Get personalized recommendations for a user."""
    try:
        role = UserRole(request.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}")

    recommendations = recommendation_engine.get_recommendations(
        user_id=request.user_id,
        role=role,
        current_page=request.current_page,
        max_results=request.max_results,
    )

    return recommendations


@router.post("/activity")
async def record_activity(request: ActivityRequest):
    """Record a user activity for profiling."""
    try:
        activity_type = ActivityType(request.activity_type)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid activity type: {request.activity_type}"
        )

    recommendation_engine.record_activity(
        user_id=request.user_id,
        activity_type=activity_type,
        page=request.page,
        metadata=request.metadata,
    )

    return {"message": "Activity recorded successfully"}


@router.post("/feedback")
async def record_feedback(request: FeedbackRequest):
    """Record user feedback on a recommendation."""
    recommendation_engine.record_feedback(
        user_id=request.user_id,
        recommendation_id=request.recommendation_id,
        accepted=request.accepted,
    )

    return {"message": "Feedback recorded successfully"}


@router.get("/feedback/stats")
async def get_feedback_stats(user_id: Optional[str] = None):
    """Get feedback statistics.

    Args:
        user_id: Optional user ID for specific user stats
    """
    stats = recommendation_engine.get_feedback_stats(user_id)
    return stats


@router.post("/cleanup")
async def cleanup_old_data(days: int = 30):
    """Clean up old activities and feedback data.

    Args:
        days: Number of days to keep
    """
    recommendation_engine.cleanup_old_data(days)
    return {"message": f"Cleaned up data older than {days} days"}
