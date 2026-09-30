"""User profiling for LLM-based recommendations (issue #474)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles in the system."""

    ADMIN = "admin"
    ANALYST = "analyst"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class ActivityType(Enum):
    """Types of user activities."""

    VIEW_DASHBOARD = "view_dashboard"
    RUN_MODEL = "run_model"
    VIEW_ALERTS = "view_alerts"
    QUERY_DATA = "query_data"
    EXPORT_DATA = "export_data"
    CONFIGURE_FEATURE = "configure_feature"
    VIEW_FEATURES = "view_features"


@dataclass
class UserActivity:
    """Record of a user activity.

    Attributes:
        activity_type: Type of activity
        timestamp: When the activity occurred
        page: Page where activity occurred
        metadata: Additional activity metadata
    """

    activity_type: ActivityType
    timestamp: datetime
    page: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """User profile for recommendation personalization.

    Attributes:
        user_id: User identifier
        role: User role
        activities: List of recent activities
        preferences: User preferences
        skill_level: Assessed skill level
        interests: Detected interests
        last_active: Last activity timestamp
    """

    user_id: str
    role: UserRole
    activities: List[UserActivity] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    skill_level: str = "intermediate"
    interests: Set[str] = field(default_factory=set)
    last_active: datetime = field(default_factory=datetime.utcnow)

    @property
    def recent_activities(self, hours: int = 24) -> List[UserActivity]:
        """Get activities from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [a for a in self.activities if a.timestamp >= cutoff]

    @property
    def most_common_pages(self, top_n: int = 5) -> List[str]:
        """Get most commonly visited pages."""
        from collections import Counter

        page_counts = Counter(a.page for a in self.activities)
        return [page for page, _ in page_counts.most_common(top_n)]

    @property
    def activity_summary(self) -> Dict[str, int]:
        """Get summary of activity types."""
        from collections import Counter

        activity_counts = Counter(a.activity_type for a in self.activities)
        return {activity.value: count for activity, count in activity_counts.items()}


class UserProfiler:
    """Profile users for personalized recommendations."""

    def __init__(self) -> Any:
        """Initialize user profiler."""
        self.profiles: Dict[str, UserProfile] = {}

    def get_or_create_profile(self, user_id: str, role: UserRole = UserRole.VIEWER) -> UserProfile:
        """Get or create a user profile.

        Args:
            user_id: User identifier
            role: User role

        Returns:
            UserProfile
        """
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id=user_id, role=role)
        return self.profiles[user_id]

    def record_activity(
        self,
        user_id: str,
        activity_type: ActivityType,
        page: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a user activity.

        Args:
            user_id: User identifier
            activity_type: Type of activity
            page: Page where activity occurred
            metadata: Additional metadata
        """
        profile = self.get_or_create_profile(user_id)
        activity = UserActivity(
            activity_type=activity_type,
            timestamp=datetime.utcnow(),
            page=page,
            metadata=metadata or {},
        )
        profile.activities.append(activity)
        profile.last_active = activity.timestamp

        # Update interests based on activity
        self._update_interests(profile, activity)

    def _update_interests(self, profile: UserProfile, activity: UserActivity) -> None:
        """Update user interests based on activity.

        Args:
            profile: User profile
            activity: User activity
        """
        # Extract interests from page and metadata
        if "fraud" in activity.page.lower():
            profile.interests.add("fraud_detection")
        if "model" in activity.page.lower():
            profile.interests.add("machine_learning")
        if "feature" in activity.page.lower():
            profile.interests.add("feature_engineering")
        if "graph" in activity.page.lower():
            profile.interests.add("graph_analysis")

        # Extract from metadata
        if "feature" in activity.metadata:
            profile.interests.add("feature_engineering")
        if "model" in activity.metadata:
            profile.interests.add("machine_learning")

    def assess_skill_level(self, user_id: str) -> str:
        """Assess user skill level based on activities.

        Args:
            user_id: User identifier

        Returns:
            Skill level: beginner, intermediate, or advanced
        """
        profile = self.get_or_create_profile(user_id)
        summary = profile.activity_summary

        # Count advanced activities
        advanced_activities = [
            ActivityType.RUN_MODEL,
            ActivityType.CONFIGURE_FEATURE,
            ActivityType.QUERY_DATA,
        ]
        advanced_count = sum(summary.get(activity.value, 0) for activity in advanced_activities)

        total_activities = sum(summary.values())

        if total_activities < 10:
            return "beginner"
        elif advanced_count > total_activities * 0.5:
            return "advanced"
        else:
            return "intermediate"

    def get_profile_context(self, user_id: str) -> Dict[str, Any]:
        """Get context for recommendation generation.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with profile context
        """
        profile = self.get_or_create_profile(user_id)
        return {
            "user_id": user_id,
            "role": profile.role.value,
            "skill_level": profile.skill_level,
            "interests": list(profile.interests),
            "recent_pages": profile.most_common_pages(3),
            "activity_summary": profile.activity_summary,
            "last_active": profile.last_active.isoformat(),
        }

    def cleanup_old_activities(self, days: int = 30) -> None:
        """Clean up activities older than N days.

        Args:
            days: Number of days to keep
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        for profile in self.profiles.values():
            profile.activities = [a for a in profile.activities if a.timestamp >= cutoff]
