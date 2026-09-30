"""Recommendation generators for LLM-based recommendations (issue #474)."""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .ranker import Recommendation

logger = logging.getLogger(__name__)


class RecommendationGenerator(ABC):
    """Abstract base class for recommendation generators."""

    @abstractmethod
    def generate(self, context: Dict[str, Any]) -> List[Recommendation]:
        """Generate recommendations based on context.

        Args:
            context: User and system context

        Returns:
            List of recommendations
        """
        pass


class FeatureRecommendationGenerator(RecommendationGenerator):
    """Generate feature-related recommendations."""

    def generate(self, context: Dict[str, Any]) -> List[Recommendation]:
        """Generate feature recommendations.

        Args:
            context: User and system context

        Returns:
            List of feature recommendations
        """
        recommendations = []
        interests = context.get("interests", [])
        recent_pages = context.get("recent_pages", [])

        # Fraud explanation feature
        if "fraud_detection" in interests or any("fraud" in p for p in recent_pages):
            recommendations.append(
                Recommendation(
                    id=str(uuid.uuid4()),
                    type="feature",
                    title="Try the fraud explanation feature",
                    description="Get detailed explanations for fraud alerts using LLM-powered analysis",
                    explanation="Based on your interest in fraud detection, this feature can help you understand alert patterns",
                    action_url="/features/fraud-explanation",
                    confidence=0.8,
                    priority="high",
                    metadata={
                        "interests": ["fraud_detection"],
                        "skill_level": "intermediate",
                        "related_pages": ["/alerts", "/fraud"],
                    },
                )
            )

        # Batch scoring feature
        if "machine_learning" in interests or any("model" in p for p in recent_pages):
            recommendations.append(
                Recommendation(
                    id=str(uuid.uuid4()),
                    type="feature",
                    title="Set up batch scoring for these accounts",
                    description="Configure automated batch scoring for multiple accounts at once",
                    explanation="You've been working with models - batch scoring can improve your workflow efficiency",
                    action_url="/models/batch-scoring",
                    confidence=0.7,
                    priority="medium",
                    metadata={
                        "interests": ["machine_learning"],
                        "skill_level": "advanced",
                        "related_pages": ["/models", "/scoring"],
                    },
                )
            )

        # Feature engineering suggestions
        if "feature_engineering" in interests:
            recommendations.append(
                Recommendation(
                    id=str(uuid.uuid4()),
                    type="feature",
                    title="Explore graph-based features",
                    description="Use transaction graph features to improve model performance",
                    explanation="Graph-based features can capture structural patterns in transaction networks",
                    action_url="/features/graph",
                    confidence=0.6,
                    priority="medium",
                    metadata={
                        "interests": ["feature_engineering", "graph_analysis"],
                        "skill_level": "intermediate",
                        "related_pages": ["/features"],
                    },
                )
            )

        return recommendations


class ModelRecommendationGenerator(RecommendationGenerator):
    """Generate model-related recommendations."""

    def generate(self, context: Dict[str, Any]) -> List[Recommendation]:
        """Generate model recommendations.

        Args:
            context: User and system context

        Returns:
            List of model recommendations
        """
        recommendations = []
        skill_level = context.get("skill_level", "intermediate")

        # Model version recommendation
        if skill_level in ["intermediate", "advanced"]:
            recommendations.append(
                Recommendation(
                    id=str(uuid.uuid4()),
                    type="model",
                    title="Model v2.1 shows 15% improvement",
                    description="The latest model version shows improved performance on recent data",
                    explanation="Based on evaluation metrics, upgrading to v2.1 could improve your fraud detection accuracy",
                    action_url="/models/versions",
                    confidence=0.75,
                    priority="high",
                    metadata={
                        "interests": ["machine_learning"],
                        "skill_level": "advanced",
                        "related_pages": ["/models"],
                    },
                )
            )

        # Retrain recommendation
        recommendations.append(
            Recommendation(
                id=str(uuid.uuid4()),
                type="model",
                title="Retrain model with recent data",
                description="Your model training data is 30 days old - consider retraining with recent transactions",
                explanation="Models can drift over time - retraining with fresh data maintains performance",
                action_url="/models/retrain",
                confidence=0.65,
                priority="medium",
                metadata={
                    "interests": ["machine_learning"],
                    "skill_level": "advanced",
                    "related_pages": ["/models"],
                },
            )
        )

        return recommendations


class QuerySuggestionGenerator(RecommendationGenerator):
    """Generate query suggestions."""

    def generate(self, context: Dict[str, Any]) -> List[Recommendation]:
        """Generate query suggestions.

        Args:
            context: User and system context

        Returns:
            List of query suggestions
        """
        recommendations = []
        recent_pages = context.get("recent_pages", [])

        # Similar queries
        if any("query" in p for p in recent_pages):
            recommendations.append(
                Recommendation(
                    id=str(uuid.uuid4()),
                    type="query",
                    title="Similar queries: high-value transactions",
                    description="Try querying for transactions above $10,000 for fraud analysis",
                    explanation="Users with similar interests often query high-value transactions",
                    action_url="/query?template=high-value",
                    confidence=0.6,
                    priority="low",
                    metadata={
                        "interests": ["fraud_detection"],
                        "skill_level": "beginner",
                        "related_pages": ["/query"],
                    },
                )
            )

        # Related metrics
        recommendations.append(
            Recommendation(
                id=str(uuid.uuid4()),
                type="query",
                title="Related metrics: transaction frequency",
                description="Explore transaction frequency metrics for anomaly detection",
                explanation="Frequency analysis can reveal unusual patterns in transaction behavior",
                action_url="/metrics/frequency",
                confidence=0.55,
                priority="low",
                metadata={
                    "interests": ["fraud_detection", "feature_engineering"],
                    "skill_level": "intermediate",
                    "related_pages": ["/metrics"],
                },
            )
        )

        return recommendations


class InsightGenerator(RecommendationGenerator):
    """Generate insight-based recommendations."""

    def generate(self, context: Dict[str, Any]) -> List[Recommendation]:
        """Generate insight recommendations.

        Args:
            context: User and system context

        Returns:
            List of insight recommendations
        """
        recommendations = []
        interests = context.get("interests", [])

        # Pattern detection insight
        if "fraud_detection" in interests:
            recommendations.append(
                Recommendation(
                    id=str(uuid.uuid4()),
                    type="insight",
                    title="Unusual pattern detected in transactions",
                    description="We've detected a spike in circular transactions in the last 24 hours",
                    explanation="Anomaly detection identified a 3x increase in circular transaction patterns",
                    action_url="/insights/circular-transactions",
                    confidence=0.85,
                    priority="high",
                    metadata={
                        "interests": ["fraud_detection"],
                        "skill_level": "intermediate",
                        "related_pages": ["/insights", "/alerts"],
                    },
                )
            )

        # Threshold adjustment insight
        recommendations.append(
            Recommendation(
                id=str(uuid.uuid4()),
                type="insight",
                title="Consider adjusting fraud threshold",
                description="Current false positive rate is 12% - adjusting threshold could improve precision",
                explanation="Based on recent alert performance, a threshold adjustment may reduce false positives",
                action_url="/models/thresholds",
                confidence=0.7,
                priority="medium",
                metadata={
                    "interests": ["machine_learning", "fraud_detection"],
                    "skill_level": "advanced",
                    "related_pages": ["/models", "/alerts"],
                },
            )
        )

        return recommendations
