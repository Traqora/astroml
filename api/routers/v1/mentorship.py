"""Mentorship program API endpoints (Contributors).

Endpoints:
  GET /api/v1/mentorship/mentors                    — list mentors (paginated)
  POST /api/v1/mentorship/mentors                   — register as mentor
  GET /api/v1/mentorship/mentors/{id}               — mentor profile
  PUT /api/v1/mentorship/mentors/{id}               — update mentor profile
  GET /api/v1/mentorship/mentees                    — list mentees (paginated)
  POST /api/v1/mentorship/mentees                   — register as mentee
  GET /api/v1/mentorship/mentees/{id}               — mentee profile
  PUT /api/v1/mentorship/mentees/{id}               — update mentee profile
  GET /api/v1/mentorship/matches/{mentee_id}       — find mentor matches
  POST /api/v1/mentorship/relationships             — start mentorship
  GET /api/v1/mentorship/relationships              — list mentorships (paginated)
  GET /api/v1/mentorship/relationships/{id}        — get mentorship details
  PUT /api/v1/mentorship/relationships/{id}        — update mentorship status
  POST /api/v1/mentorship/sessions/{id}            — record session
  GET /api/v1/mentorship/sessions/{id}             — get session details
  POST /api/v1/mentorship/feedback                 — submit feedback
  GET /api/v1/mentorship/metrics/mentorship/{id}   — mentorship metrics
  GET /api/v1/mentorship/metrics/mentor/{id}       — mentor metrics
  GET /api/v1/mentorship/dashboard/mentor/{id}     — mentor dashboard
  GET /api/v1/mentorship/dashboard/mentee/{id}     — mentee dashboard
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import AuthContext, get_current_auth, require_scopes
from api.database import get_db
from api.models.orm import (
    Mentor,
    Mentee,
    Mentorship,
    MentorshipSession,
    MentorshipFeedback,
)
from api.schemas import (
    MentorProfileIn,
    MentorProfileOut,
    MenteeProfileIn,
    MenteeProfileOut,
    MentorMatchOut,
    MentorshipOut,
    MentorshipSessionIn,
    MentorshipSessionOut,
    MentorshipFeedbackIn,
    MentorshipFeedbackOut,
    MentorshipMetrics,
    MentorMetrics,
    MentorshipListResponse,
    MentorListResponse,
    MenteeListResponse,
)
from astroml.contributors.mentorship import MentorshipMatcher, MentorshipTracking

router = APIRouter(prefix="/api/v1/mentorship", tags=["mentorship"])


# ─── Mentors ──────────────────────────────────────────────────────────────

@router.get("/mentors", response_model=MentorListResponse)
async def list_mentors(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    skill: Optional[str] = None,
    available_only: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """List mentors with optional filtering."""
    q = select(Mentor)

    if available_only:
        q = q.where(Mentor.is_available.is_(True))

    if skill:
        # Note: This is a simplified filter; JSONB array contains would be better
        q = q.where(Mentor.skills.astext.like(f"%{skill}%"))

    # Count total
    count_result = await db.execute(select(func.count()).select_from(Mentor))
    total = count_result.scalar()

    # Paginate
    q = q.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(q)
    mentors = result.scalars().all()

    return MentorListResponse(
        data=[MentorProfileOut.from_orm(m) for m in mentors],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.post("/mentors", response_model=MentorProfileOut)
async def register_mentor(
    body: MentorProfileIn,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Register current user as a mentor."""
    # Check if mentor profile already exists
    result = await db.execute(
        select(Mentor).where(Mentor.user_id == auth.user_id)
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Mentor profile already exists")

    mentor = Mentor(
        user_id=auth.user_id,
        github_username=auth.subject,
        bio=body.bio,
        skills=body.skills,
        years_experience=body.years_experience,
        preferred_session_day=body.preferred_session_day,
        max_mentees=body.max_mentees,
    )
    db.add(mentor)
    await db.flush()

    return MentorProfileOut.from_orm(mentor)


@router.get("/mentors/{mentor_id}", response_model=MentorProfileOut)
async def get_mentor(mentor_id: int, db: AsyncSession = Depends(get_db)):
    """Get mentor profile by ID."""
    result = await db.execute(select(Mentor).where(Mentor.id == mentor_id))
    mentor = result.scalar_one_or_none()

    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")

    return MentorProfileOut.from_orm(mentor)


@router.put("/mentors/{mentor_id}", response_model=MentorProfileOut)
async def update_mentor(
    mentor_id: int,
    body: MentorProfileIn,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Update mentor profile (owner only)."""
    result = await db.execute(select(Mentor).where(Mentor.id == mentor_id))
    mentor = result.scalar_one_or_none()

    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")

    if mentor.user_id != auth.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this profile")

    mentor.bio = body.bio
    mentor.skills = body.skills
    mentor.years_experience = body.years_experience
    mentor.preferred_session_day = body.preferred_session_day
    mentor.max_mentees = body.max_mentees

    await db.flush()
    return MentorProfileOut.from_orm(mentor)


# ─── Mentees ──────────────────────────────────────────────────────────────

@router.get("/mentees", response_model=MenteeListResponse)
async def list_mentees(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    interest: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List mentees with optional filtering."""
    q = select(Mentee)

    if interest:
        q = q.where(Mentee.learning_interests.astext.like(f"%{interest}%"))

    # Count total
    count_result = await db.execute(select(func.count()).select_from(Mentee))
    total = count_result.scalar()

    # Paginate
    q = q.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(q)
    mentees = result.scalars().all()

    return MenteeListResponse(
        data=[MenteeProfileOut.from_orm(m) for m in mentees],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.post("/mentees", response_model=MenteeProfileOut)
async def register_mentee(
    body: MenteeProfileIn,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Register current user as a mentee."""
    # Check if mentee profile already exists
    result = await db.execute(
        select(Mentee).where(Mentee.user_id == auth.user_id)
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Mentee profile already exists")

    mentee = Mentee(
        user_id=auth.user_id,
        github_username=auth.subject,
        bio=body.bio,
        learning_interests=body.learning_interests,
        years_experience=body.years_experience,
        preferred_session_day=body.preferred_session_day,
        goals=body.goals,
    )
    db.add(mentee)
    await db.flush()

    return MenteeProfileOut.from_orm(mentee)


@router.get("/mentees/{mentee_id}", response_model=MenteeProfileOut)
async def get_mentee(mentee_id: int, db: AsyncSession = Depends(get_db)):
    """Get mentee profile by ID."""
    result = await db.execute(select(Mentee).where(Mentee.id == mentee_id))
    mentee = result.scalar_one_or_none()

    if not mentee:
        raise HTTPException(status_code=404, detail="Mentee not found")

    return MenteeProfileOut.from_orm(mentee)


@router.put("/mentees/{mentee_id}", response_model=MenteeProfileOut)
async def update_mentee(
    mentee_id: int,
    body: MenteeProfileIn,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Update mentee profile (owner only)."""
    result = await db.execute(select(Mentee).where(Mentee.id == mentee_id))
    mentee = result.scalar_one_or_none()

    if not mentee:
        raise HTTPException(status_code=404, detail="Mentee not found")

    if mentee.user_id != auth.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this profile")

    mentee.bio = body.bio
    mentee.learning_interests = body.learning_interests
    mentee.years_experience = body.years_experience
    mentee.preferred_session_day = body.preferred_session_day
    mentee.goals = body.goals

    await db.flush()
    return MenteeProfileOut.from_orm(mentee)


# ─── Matching ──────────────────────────────────────────────────────────────

@router.get("/matches/{mentee_id}", response_model=list[MentorMatchOut])
async def find_mentor_matches(
    mentee_id: int,
    limit: int = Query(5, ge=1, le=20),
    min_score: float = Query(0.6, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
):
    """Find best mentor matches for a mentee using skill-based algorithm."""
    # Verify mentee exists
    result = await db.execute(select(Mentee).where(Mentee.id == mentee_id))
    mentee = result.scalar_one_or_none()
    if not mentee:
        raise HTTPException(status_code=404, detail="Mentee not found")

    from sqlalchemy.orm import Session as SyncSession  # noqa: PLC0415

    matcher = MentorshipMatcher(db)  # type: ignore
    scores = matcher.find_matches(mentee_id, limit=limit, min_score=min_score)

    result_list = []
    for score in scores:
        mentor_result = await db.execute(
            select(Mentor).where(Mentor.id == score.mentor_id)
        )
        mentor = mentor_result.scalar_one()
        result_list.append(
            MentorMatchOut(
                mentor_id=mentor.id,
                mentor_username=mentor.github_username,
                skill_overlap=score.skill_overlap,
                experience_gap=score.experience_gap,
                availability_match=score.availability_match,
                total_score=score.total_score,
            )
        )

    return result_list


# ─── Relationships ───────────────────────────────────────────────────────────

@router.post("/relationships", response_model=MentorshipOut)
async def create_mentorship(
    mentor_id: int = Query(...),
    mentee_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Create a new mentorship relationship."""
    # Verify both mentor and mentee exist
    mentor_result = await db.execute(select(Mentor).where(Mentor.id == mentor_id))
    mentor = mentor_result.scalar_one_or_none()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")

    mentee_result = await db.execute(select(Mentee).where(Mentee.id == mentee_id))
    mentee = mentee_result.scalar_one_or_none()
    if not mentee:
        raise HTTPException(status_code=404, detail="Mentee not found")

    # Check if relationship already exists
    dup_result = await db.execute(
        select(Mentorship).where(
            Mentorship.mentor_id == mentor_id,
            Mentorship.mentee_id == mentee_id,
        )
    )
    if dup_result.scalar_one_or_none():
        raise HTTPException(
            status_code=409, detail="Mentorship relationship already exists"
        )

    # Calculate match score (simple version; use MentorshipMatcher for full logic)
    score = 0.7  # Placeholder; ideally from matcher

    mentorship = Mentorship(
        mentor_id=mentor_id,
        mentee_id=mentee_id,
        status="active",
        match_score=score,
    )
    db.add(mentorship)
    await db.flush()

    return MentorshipOut.from_orm(mentorship)


@router.get("/relationships", response_model=MentorshipListResponse)
async def list_mentorships(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    mentor_id: Optional[int] = None,
    mentee_id: Optional[int] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List mentorships with optional filtering."""
    q = select(Mentorship)

    if mentor_id:
        q = q.where(Mentorship.mentor_id == mentor_id)

    if mentee_id:
        q = q.where(Mentorship.mentee_id == mentee_id)

    if status:
        q = q.where(Mentorship.status == status)

    # Count total
    count_result = await db.execute(select(func.count()).select_from(Mentorship))
    total = count_result.scalar()

    # Paginate
    q = q.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(q)
    mentorships = result.scalars().all()

    # Enrich with usernames
    result_data = []
    for m in mentorships:
        mentor_result = await db.execute(
            select(Mentor).where(Mentor.id == m.mentor_id)
        )
        mentor = mentor_result.scalar_one()
        mentee_result = await db.execute(
            select(Mentee).where(Mentee.id == m.mentee_id)
        )
        mentee = mentee_result.scalar_one()

        result_data.append(
            MentorshipOut(
                id=m.id,
                mentor_id=m.mentor_id,
                mentor_username=mentor.github_username,
                mentee_id=m.mentee_id,
                mentee_username=mentee.github_username,
                status=m.status,
                match_score=m.match_score,
                started_at=m.started_at,
                ended_at=m.ended_at,
            )
        )

    return MentorshipListResponse(
        data=result_data,
        page=page,
        page_size=page_size,
        total=total,
    )


@router.get("/relationships/{mentorship_id}", response_model=MentorshipOut)
async def get_mentorship(mentorship_id: int, db: AsyncSession = Depends(get_db)):
    """Get mentorship relationship details."""
    result = await db.execute(
        select(Mentorship).where(Mentorship.id == mentorship_id)
    )
    mentorship = result.scalar_one_or_none()

    if not mentorship:
        raise HTTPException(status_code=404, detail="Mentorship not found")

    mentor_result = await db.execute(
        select(Mentor).where(Mentor.id == mentorship.mentor_id)
    )
    mentor = mentor_result.scalar_one()
    mentee_result = await db.execute(
        select(Mentee).where(Mentee.id == mentorship.mentee_id)
    )
    mentee = mentee_result.scalar_one()

    return MentorshipOut(
        id=mentorship.id,
        mentor_id=mentorship.mentor_id,
        mentor_username=mentor.github_username,
        mentee_id=mentorship.mentee_id,
        mentee_username=mentee.github_username,
        status=mentorship.status,
        match_score=mentorship.match_score,
        started_at=mentorship.started_at,
        ended_at=mentorship.ended_at,
    )


@router.put("/relationships/{mentorship_id}", response_model=MentorshipOut)
async def update_mentorship(
    mentorship_id: int,
    status: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Update mentorship status (active/paused/completed)."""
    result = await db.execute(
        select(Mentorship).where(Mentorship.id == mentorship_id)
    )
    mentorship = result.scalar_one_or_none()

    if not mentorship:
        raise HTTPException(status_code=404, detail="Mentorship not found")

    if status not in ("active", "paused", "completed"):
        raise HTTPException(
            status_code=400, detail="Invalid status; must be active/paused/completed"
        )

    mentorship.status = status
    if status == "completed":
        mentorship.ended_at = datetime.utcnow()

    await db.flush()
    return await get_mentorship(mentorship_id, db)


# ─── Sessions ──────────────────────────────────────────────────────────────

@router.post("/sessions", response_model=MentorshipSessionOut)
async def record_session(
    mentorship_id: int = Query(...),
    body: MentorshipSessionIn = ...,
    db: AsyncSession = Depends(get_db),
):
    """Record a mentorship session."""
    # Verify mentorship exists
    m_result = await db.execute(
        select(Mentorship).where(Mentorship.id == mentorship_id)
    )
    if not m_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Mentorship not found")

    session = MentorshipSession(
        mentorship_id=mentorship_id,
        session_date=datetime.utcnow(),
        duration_minutes=body.duration_minutes,
        topic=body.topic,
        notes=body.notes,
    )
    db.add(session)
    await db.flush()

    return MentorshipSessionOut.from_orm(session)


@router.get("/sessions/{session_id}", response_model=MentorshipSessionOut)
async def get_session(session_id: int, db: AsyncSession = Depends(get_db)):
    """Get session details."""
    result = await db.execute(
        select(MentorshipSession).where(MentorshipSession.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return MentorshipSessionOut.from_orm(session)


# ─── Feedback ──────────────────────────────────────────────────────────────

@router.post("/feedback", response_model=MentorshipFeedbackOut)
async def submit_feedback(
    session_id: int = Query(...),
    is_mentor: bool = Query(...),
    body: MentorshipFeedbackIn = ...,
    db: AsyncSession = Depends(get_db),
):
    """Submit feedback for a session."""
    # Verify session exists
    session_result = await db.execute(
        select(MentorshipSession).where(MentorshipSession.id == session_id)
    )
    session = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check for duplicate feedback
    dup_result = await db.execute(
        select(MentorshipFeedback).where(
            MentorshipFeedback.session_id == session_id,
            MentorshipFeedback.is_mentor_feedback.is_(is_mentor),
        )
    )
    if dup_result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Feedback already submitted")

    feedback = MentorshipFeedback(
        session_id=session_id,
        mentorship_id=session.mentorship_id,
        rating=body.rating,
        feedback_text=body.feedback_text,
        is_mentor_feedback=is_mentor,
    )
    db.add(feedback)
    await db.flush()

    return MentorshipFeedbackOut.from_orm(feedback)


# ─── Metrics & Dashboard ──────────────────────────────────────────────────────

@router.get("/metrics/mentorship/{mentorship_id}", response_model=MentorshipMetrics)
async def get_mentorship_metrics(
    mentorship_id: int, db: AsyncSession = Depends(get_db)
):
    """Get metrics for a specific mentorship relationship."""
    # Verify mentorship exists
    m_result = await db.execute(
        select(Mentorship).where(Mentorship.id == mentorship_id)
    )
    if not m_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Mentorship not found")

    sessions_result = await db.execute(
        select(MentorshipSession).where(
            MentorshipSession.mentorship_id == mentorship_id
        )
    )
    sessions = sessions_result.scalars().all()

    total_sessions = len(sessions)
    total_minutes = sum(s.duration_minutes for s in sessions)
    total_hours = total_minutes / 60.0

    topics = list(set(s.topic for s in sessions if s.topic))

    last_session = max(
        (s.session_date for s in sessions if s.session_date),
        default=None,
    )

    # Get feedback ratings
    feedback_result = await db.execute(
        select(MentorshipFeedback).where(
            MentorshipFeedback.mentorship_id == mentorship_id
        )
    )
    all_feedback = feedback_result.scalars().all()
    ratings = [f.rating for f in all_feedback if f.rating]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0.0

    return MentorshipMetrics(
        total_sessions=total_sessions,
        total_hours=round(total_hours, 2),
        avg_rating=round(avg_rating, 2),
        topics_covered=topics,
        last_session_date=last_session,
    )


@router.get("/metrics/mentor/{mentor_id}", response_model=MentorMetrics)
async def get_mentor_metrics(mentor_id: int, db: AsyncSession = Depends(get_db)):
    """Get aggregated metrics for a mentor."""
    # Verify mentor exists
    mentor_result = await db.execute(select(Mentor).where(Mentor.id == mentor_id))
    if not mentor_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Mentor not found")

    mentorships_result = await db.execute(
        select(Mentorship).where(Mentorship.mentor_id == mentor_id)
    )
    mentorships = mentorships_result.scalars().all()

    total_mentees = len(mentorships)
    total_sessions = 0
    total_hours = 0.0
    all_ratings = []

    for mentorship in mentorships:
        metrics = await get_mentorship_metrics(mentorship.id, db)
        total_sessions += metrics.total_sessions
        total_hours += metrics.total_hours
        if metrics.avg_rating > 0:
            all_ratings.append(metrics.avg_rating)

    avg_rating = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0

    return MentorMetrics(
        total_mentees=total_mentees,
        total_sessions=total_sessions,
        total_hours=round(total_hours, 2),
        avg_rating=round(avg_rating, 2),
    )


@router.get("/dashboard/mentor/{mentor_id}")
async def get_mentor_dashboard(mentor_id: int, db: AsyncSession = Depends(get_db)):
    """Get comprehensive mentor dashboard."""
    mentor_result = await db.execute(select(Mentor).where(Mentor.id == mentor_id))
    mentor = mentor_result.scalar_one_or_none()

    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")

    metrics = await get_mentor_metrics(mentor_id, db)

    # Get active mentorships
    mentorships_result = await db.execute(
        select(Mentorship).where(
            Mentorship.mentor_id == mentor_id,
            Mentorship.status == "active",
        )
    )
    active_mentorships = mentorships_result.scalars().all()

    return {
        "mentor": MentorProfileOut.from_orm(mentor),
        "metrics": metrics,
        "active_mentorships_count": len(active_mentorships),
    }


@router.get("/dashboard/mentee/{mentee_id}")
async def get_mentee_dashboard(mentee_id: int, db: AsyncSession = Depends(get_db)):
    """Get comprehensive mentee dashboard."""
    mentee_result = await db.execute(select(Mentee).where(Mentee.id == mentee_id))
    mentee = mentee_result.scalar_one_or_none()

    if not mentee:
        raise HTTPException(status_code=404, detail="Mentee not found")

    # Get active mentorships
    mentorships_result = await db.execute(
        select(Mentorship).where(
            Mentorship.mentee_id == mentee_id,
            Mentorship.status == "active",
        )
    )
    active_mentorships = mentorships_result.scalars().all()

    # Get metrics for each mentorship
    mentorship_metrics_list = []
    for mentorship in active_mentorships:
        metrics = await get_mentorship_metrics(mentorship.id, db)
        mentor_result = await db.execute(
            select(Mentor).where(Mentor.id == mentorship.mentor_id)
        )
        mentor = mentor_result.scalar_one()
        mentorship_metrics_list.append({
            "mentorship_id": mentorship.id,
            "mentor_name": mentor.github_username,
            "metrics": metrics,
        })

    return {
        "mentee": MenteeProfileOut.from_orm(mentee),
        "active_mentorships": mentorship_metrics_list,
    }
