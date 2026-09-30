"""Recommendation ranking for LLM-based recommendations (issue #474)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A single recommendation.

    Attributes:
        id: Unique identifier
        type: Recommendation type
        title: Recommendation title
        description: Detailed description
        explanation: Why this recommendation was made
        action_url: URL to take action
        confidence: Confidence score (0-1)
        priority: Priority level (low, medium, high)
        metadata: Additional metadata
        created_at: When recommendation was generated
    """

    id: str
    type: str
    title: str
    description: str
    explanation: str
    action_url: Optional[str] = None
    confidence: float = 0.5
    priority: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "explanation": self.explanation,
            "action_url": self.action_url,
            "confidence": self.confidence,
            "priority": self.priority,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class RecommendationRanker:
    """Rank recommendations based on relevance and user context."""

    def __init__(self) -> Any:
        """Initialize recommendation ranker."""
        self.priority_weights = {
            "high": 3.0,
            "medium": 2.0,
            "low": 1.0,
        }

    def rank(
        self,
        recommendations: List[Recommendation],
        user_context: Dict[str, Any],
        max_results: int = 10,
    ) -> List[Recommendation]:
        """Rank recommendations based on user context.

        Args:
            recommendations: List of recommendations to rank
            user_context: User profile and context
            max_results: Maximum number of results to return

        Returns:
            Ranked list of recommendations
        """
        scored = []

        for rec in recommendations:
            score = self._calculate_score(rec, user_context)
            scored.append((rec, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return [rec for rec, _ in scored[:max_results]]

    def _calculate_score(self, rec: Recommendation, user_context: Dict[str, Any]) -> float:
        """Calculate relevance score for a recommendation.

        Args:
            rec: Recommendation
            user_context: User context

        Returns:
            Relevance score
        """
        score = 0.0

        # Base confidence
        score += rec.confidence * 0.4

        # Priority weight
        priority_weight = self.priority_weights.get(rec.priority, 2.0)
        score += (priority_weight / 3.0) * 0.3

        # Interest alignment
        interests = user_context.get("interests", [])
        if interests:
            rec_interests = rec.metadata.get("interests", [])
            alignment = len(set(interests) & set(rec_interests)) / max(len(interests), 1)
            score += alignment * 0.2

        # Skill level match
        skill_level = user_context.get("skill_level", "intermediate")
        rec_skill_level = rec.metadata.get("skill_level", "intermediate")
        if skill_level == rec_skill_level:
            score += 0.1

        # Recency boost for recent page matches
        recent_pages = user_context.get("recent_pages", [])
        if recent_pages:
            rec_pages = rec.metadata.get("related_pages", [])
            if any(page in recent_pages for page in rec_pages):
                score += 0.1

        return min(score, 1.0)

    def deduplicate(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Remove duplicate recommendations.

        Args:
            recommendations: List of recommendations

        Returns:
            Deduplicated list
        """
        seen = set()
        deduped = []

        for rec in recommendations:
            # Use title + type as deduplication key
            key = (rec.title, rec.type)
            if key not in seen:
                seen.add(key)
                deduped.append(rec)

        return deduped

    def diversify(
        self,
        recommendations: List[Recommendation],
        max_per_type: int = 3,
    ) -> List[Recommendation]:
        """Ensure diversity in recommendation types.

        Args:
            recommendations: List of recommendations
            max_per_type: Maximum recommendations per type

        Returns:
            Diversified list
        """
        by_type: Dict[str, List[Recommendation]] = {}

        for rec in recommendations:
            if rec.type not in by_type:
                by_type[rec.type] = []
            by_type[rec.type].append(rec)

        # Take top N from each type
        diversified = []
        for rec_type, recs in by_type.items():
            diversified.extend(recs[:max_per_type])

        # Sort by original confidence
        diversified.sort(key=lambda r: r.confidence, reverse=True)

        return diversified
