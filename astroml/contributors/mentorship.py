"""Mentorship program core logic — matching, tracking, and metrics.

Features:
  - Mentor/mentee matching based on skills and interests
  - Session management and tracking
  - Feedback collection and storage
  - Progress metrics and KPIs
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session


@dataclass
class SkillScore:
    """Compatibility score for a mentor-mentee pair."""
    mentor_id: int
    mentee_id: int
    skill_overlap: float  # 0-1: matching skill interests
    experience_gap: float  # 0-1: mentee can learn from mentor
    availability_match: float  # 0-1: both available at similar times
    total_score: float  # weighted combination


class MentorshipMatcher:
    """Matches mentees with mentors using skill-based algorithm."""

    def __init__(self, db: Session):
        self.db = db

    def find_matches(
        self,
        mentee_id: int,
        limit: int = 5,
        min_score: float = 0.6,
    ) -> list[SkillScore]:
        """Find best mentor matches for a mentee.

        Args:
            mentee_id: ID of mentee seeking mentor
            limit: Max number of matches to return
            min_score: Minimum compatibility score (0-1)

        Returns:
            List of SkillScore objects, sorted by total_score descending
        """
        from api.models.orm import Mentee as MenteeModel  # noqa: PLC0415
        from api.models.orm import Mentor as MentorModel  # noqa: PLC0415

        mentee_result = self.db.execute(
            select(MenteeModel).where(MenteeModel.id == mentee_id)
        )
        mentee = mentee_result.scalar_one_or_none()
        if not mentee:
            return []

        mentors_result = self.db.execute(
            select(MentorModel).where(MentorModel.is_available.is_(True))
        )
        mentors = mentors_result.scalars().all()

        scores: list[SkillScore] = []
        for mentor in mentors:
            if mentor.id == mentee_id:  # Skip self-matching
                continue

            skill_overlap = self._calculate_skill_overlap(
                mentor.skills or [], mentee.learning_interests or []
            )
            experience_gap = self._calculate_experience_gap(
                mentor.years_experience or 0, mentee.years_experience or 0
            )
            availability_match = self._calculate_availability_match(
                mentor.preferred_session_day, mentee.preferred_session_day
            )

            total_score = (
                skill_overlap * 0.5 +
                experience_gap * 0.3 +
                availability_match * 0.2
            )

            if total_score >= min_score:
                scores.append(SkillScore(
                    mentor_id=mentor.id,
                    mentee_id=mentee_id,
                    skill_overlap=skill_overlap,
                    experience_gap=experience_gap,
                    availability_match=availability_match,
                    total_score=total_score,
                ))

        return sorted(scores, key=lambda s: s.total_score, reverse=True)[:limit]

    @staticmethod
    def _calculate_skill_overlap(mentor_skills: list[str], mentee_interests: list[str]) -> float:
        """Calculate overlap between mentor skills and mentee interests (0-1)."""
        if not mentor_skills or not mentee_interests:
            return 0.0

        mentor_set = set(s.lower().strip() for s in mentor_skills)
        interest_set = set(i.lower().strip() for i in mentee_interests)

        if not mentor_set or not interest_set:
            return 0.0

        overlap = len(mentor_set & interest_set)
        union = len(mentor_set | interest_set)
        return overlap / union if union > 0 else 0.0

    @staticmethod
    def _calculate_experience_gap(mentor_years: int, mentee_years: int) -> float:
        """Calculate if mentor has enough more experience than mentee (0-1).

        Higher score if mentor is significantly more experienced.
        """
        gap = mentor_years - mentee_years
        if gap < 0:
            return 0.0  # Mentor less experienced than mentee
        if gap > 10:
            return 1.0  # Plenty of experience to share
        return gap / 10.0  # Linear scale 0-10 years

    @staticmethod
    def _calculate_availability_match(mentor_day: Optional[str], mentee_day: Optional[str]) -> float:
        """Match preferred session days (0-1).

        Returns 1.0 if both prefer same day, 0.5 if both flexible, 0.0 if conflict.
        """
        if not mentor_day and not mentee_day:
            return 0.5  # Both flexible
        if mentor_day and mentee_day:
            return 1.0 if mentor_day.lower() == mentee_day.lower() else 0.0
        return 0.5  # One flexible, one has preference


class MentorshipTracking:
    """Tracks mentorship sessions and generates metrics."""

    def __init__(self, db: Session):
        self.db = db

    def record_session(
        self,
        mentorship_id: int,
        duration_minutes: int,
        topic: str,
        notes: Optional[str] = None,
    ) -> int:
        """Record a completed mentorship session.

        Args:
            mentorship_id: ID of the mentorship relationship
            duration_minutes: Session duration in minutes
            topic: Topic discussed
            notes: Optional session notes

        Returns:
            ID of created session record
        """
        from api.models.orm import MentorshipSession  # noqa: PLC0415

        session_record = MentorshipSession(
            mentorship_id=mentorship_id,
            duration_minutes=duration_minutes,
            topic=topic,
            notes=notes,
            session_date=datetime.now(timezone.utc),
        )
        self.db.add(session_record)
        self.db.flush()
        return session_record.id

    def collect_feedback(
        self,
        session_id: int,
        rating: int,
        feedback_text: Optional[str] = None,
        mentor_feedback: bool = False,
    ) -> int:
        """Collect feedback for a mentorship session.

        Args:
            session_id: ID of session
            rating: 1-5 star rating
            feedback_text: Optional written feedback
            mentor_feedback: True if from mentor, False if from mentee

        Returns:
            ID of created feedback record
        """
        from api.models.orm import MentorshipFeedback  # noqa: PLC0415

        feedback_record = MentorshipFeedback(
            session_id=session_id,
            rating=rating,
            feedback_text=feedback_text,
            is_mentor_feedback=mentor_feedback,
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(feedback_record)
        self.db.flush()
        return feedback_record.id

    def get_mentorship_metrics(self, mentorship_id: int) -> dict:
        """Calculate metrics for a mentorship relationship.

        Returns:
            Dict with keys: total_sessions, total_hours, avg_rating,
            topics_covered, last_session_date
        """
        from api.models.orm import (  # noqa: PLC0415
            MentorshipSession,
            MentorshipFeedback,
        )

        sessions_result = self.db.execute(
            select(MentorshipSession).where(
                MentorshipSession.mentorship_id == mentorship_id
            )
        )
        sessions = sessions_result.scalars().all()

        total_sessions = len(sessions)
        total_minutes = sum(s.duration_minutes for s in sessions)
        total_hours = total_minutes / 60.0

        topics = [s.topic for s in sessions if s.topic]

        last_session = max(
            (s.session_date for s in sessions if s.session_date),
            default=None,
        )

        # Get all feedback ratings for this mentorship
        feedback_result = self.db.execute(
            select(MentorshipFeedback).where(
                MentorshipFeedback.mentorship_id == mentorship_id
            )
        )
        all_feedback = feedback_result.scalars().all()
        ratings = [f.rating for f in all_feedback if f.rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0

        return {
            "total_sessions": total_sessions,
            "total_hours": round(total_hours, 2),
            "avg_rating": round(avg_rating, 2),
            "topics_covered": list(set(topics)),
            "last_session_date": last_session,
        }

    def get_mentor_metrics(self, mentor_id: int) -> dict:
        """Get aggregated metrics for a mentor.

        Returns:
            Dict with total_mentees, total_sessions, total_hours, avg_rating
        """
        from api.models.orm import Mentorship  # noqa: PLC0415

        mentorships_result = self.db.execute(
            select(Mentorship).where(Mentorship.mentor_id == mentor_id)
        )
        mentorships = mentorships_result.scalars().all()

        total_mentees = len(mentorships)
        total_sessions = 0
        total_hours = 0.0
        all_ratings = []

        for mentorship in mentorships:
            metrics = self.get_mentorship_metrics(mentorship.id)
            total_sessions += metrics["total_sessions"]
            total_hours += metrics["total_hours"]
            if metrics["avg_rating"] > 0:
                all_ratings.append(metrics["avg_rating"])

        avg_rating = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0

        return {
            "total_mentees": total_mentees,
            "total_sessions": total_sessions,
            "total_hours": round(total_hours, 2),
            "avg_rating": round(avg_rating, 2),
        }
