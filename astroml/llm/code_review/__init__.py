"""
LLM-powered code review system for astroml.

This module provides intelligent code review capabilities including:
- Security vulnerability detection
- Performance analysis
- Style and best practices checking
- Correctness verification
- Testing coverage analysis
- Documentation completeness
- Complexity assessment
"""

from astroml.llm.code_review.reviewer import CodeReviewer, ReviewResult
from astroml.llm.code_review.suggestions import (
    Suggestion,
    SuggestionCategory,
    SuggestionSeverity,
)

__all__ = [
    "CodeReviewer",
    "ReviewResult",
    "Suggestion",
    "SuggestionCategory",
    "SuggestionSeverity",
]

__version__ = "0.1.0"
