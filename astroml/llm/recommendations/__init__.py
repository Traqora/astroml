"""LLM-based recommendation system for issue #474.

Provides intelligent suggestions for platform features, actions, and insights
based on user activity and context.

Components:
- Engine: Recommendation orchestrator
- Profiler: User profiling
- Ranker: Result ranking
- Generators: Suggestion generators
"""

from __future__ import annotations

from .engine import RecommendationEngine, recommendation_engine
from .generators import (
    FeatureRecommendationGenerator,
    InsightGenerator,
    ModelRecommendationGenerator,
    QuerySuggestionGenerator,
    RecommendationGenerator,
)
from .profiler import ActivityType, UserProfile, UserProfiler, UserRole
from .ranker import RecommendationRanker

__all__ = [
    "RecommendationEngine",
    "recommendation_engine",
    "UserProfile",
    "UserProfiler",
    "UserRole",
    "ActivityType",
    "RecommendationRanker",
    "RecommendationGenerator",
    "FeatureRecommendationGenerator",
    "ModelRecommendationGenerator",
    "QuerySuggestionGenerator",
    "InsightGenerator",
]
