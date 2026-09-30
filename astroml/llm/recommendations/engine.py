from astroml.utils.exceptions import AstroMLError
"""Recommendation engine for LLM-based recommendations (issue #474)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .generators import (
    FeatureRecommendationGenerator,
    InsightGenerator,
    ModelRecommendationGenerator,
    QuerySuggestionGenerator,
    RecommendationGenerator,
)
from .profiler import ActivityType, UserProfiler, UserRole
from .ranker import RecommendationRanker

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Orchestrates recommendation generation and ranking."""

    def __init__(self) -> Any:
        """Initialize recommendation engine."""
        self.profiler = UserProfiler()
        self.ranker = RecommendationRanker()
        self.generators: List[RecommendationGenerator] = [
            FeatureRecommendationGenerator(),
            ModelRecommendationGenerator(),
            QuerySuggestionGenerator(),
            InsightGenerator(),
        ]
        self.feedback: Dict[str, Dict[str, bool]] = {}  # user_id -> {rec_id: accepted}

    def get_recommendations(
        self,
        user_id: str,
        role: UserRole = UserRole.VIEWER,
        current_page: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get personalized recommendations for a user.

        Args:
            user_id: User identifier
            role: User role
            current_page: Current page the user is on
            max_results: Maximum number of recommendations to return

        Returns:
            List of recommendation dictionaries
        """
        # Get user profile
        profile = self.profiler.get_or_create_profile(user_id, role)

        # Update skill level
        profile.skill_level = self.profiler.assess_skill_level(user_id)

        # Get context
        context = self.profiler.get_profile_context(user_id)
        if current_page:
            context["current_page"] = current_page

        # Generate recommendations from all generators
        all_recommendations = []
        for generator in self.generators:
            try:
                recommendations = generator.generate(context)
                all_recommendations.extend(recommendations)
            except AstroMLError as e:
                logger.error(f"Generator error: {e}")

        # Deduplicate
        all_recommendations = self.ranker.deduplicate(all_recommendations)

        # Diversify
        all_recommendations = self.ranker.diversify(all_recommendations, max_per_type=3)

        # Rank based on user context
        ranked = self.ranker.rank(all_recommendations, context, max_results)

        # Apply feedback filter (downvote previously rejected)
        user_feedback = self.feedback.get(user_id, {})
        filtered = [
            rec
            for rec in ranked
            if user_feedback.get(rec.id, True)  # Default to True if no feedback
        ]

        return [rec.to_dict() for rec in filtered]

    def record_activity(
        self,
        user_id: str,
        activity_type: ActivityType,
        page: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a user activity for profiling.

        Args:
            user_id: User identifier
            activity_type: Type of activity
            page: Page where activity occurred
            metadata: Additional metadata
        """
        self.profiler.record_activity(user_id, activity_type, page, metadata)

    def record_feedback(
        self,
        user_id: str,
        recommendation_id: str,
        accepted: bool,
    ) -> None:
        """Record user feedback on a recommendation.

        Args:
            user_id: User identifier
            recommendation_id: Recommendation identifier
            accepted: Whether the user accepted the recommendation
        """
        if user_id not in self.feedback:
            self.feedback[user_id] = {}
        self.feedback[user_id][recommendation_id] = accepted

        logger.info(
            f"Recorded feedback: user={user_id}, rec={recommendation_id}, accepted={accepted}"
        )

    def get_feedback_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get feedback statistics.

        Args:
            user_id: Optional user ID to get stats for specific user

        Returns:
            Dictionary with feedback statistics
        """
        if user_id:
            user_feedback = self.feedback.get(user_id, {})
            accepted = sum(1 for v in user_feedback.values() if v)
            total = len(user_feedback)
            return {
                "user_id": user_id,
                "total_feedback": total,
                "accepted_count": accepted,
                "rejected_count": total - accepted,
                "acceptance_rate": accepted / total if total > 0 else 0,
            }
        else:
            # Global stats
            total_feedback = sum(len(fb) for fb in self.feedback.values())
            total_accepted = sum(sum(1 for v in fb.values() if v) for fb in self.feedback.values())
            return {
                "total_users": len(self.feedback),
                "total_feedback": total_feedback,
                "total_accepted": total_accepted,
                "total_rejected": total_feedback - total_accepted,
                "global_acceptance_rate": (
                    total_accepted / total_feedback if total_feedback > 0 else 0
                ),
            }

    def cleanup_old_data(self, days: int = 30) -> None:
        """Clean up old activities and feedback data.

        Args:
            days: Number of days to keep
        """
        self.profiler.cleanup_old_activities(days)
        logger.info(f"Cleaned up data older than {days} days")


# Global recommendation engine instance
recommendation_engine = RecommendationEngine()
