"""
Improvement suggestions for code review.

This module defines the data structures for code review suggestions,
including categories, severity levels, and the suggestion format.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class SuggestionCategory(Enum):
    """Categories of code review suggestions."""

    SECURITY = "Security"
    PERFORMANCE = "Performance"
    STYLE = "Style"
    CORRECTNESS = "Correctness"
    TESTING = "Testing"
    DOCUMENTATION = "Documentation"
    COMPLEXITY = "Complexity"


class SuggestionSeverity(Enum):
    """Severity levels for code review suggestions."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Suggestion:
    """
    A code review suggestion.

    Attributes:
        category: The category of the suggestion (e.g., SECURITY, PERFORMANCE)
        severity: The severity level (HIGH, MEDIUM, LOW)
        message: The main issue description
        file_path: Path to the file where the issue was found
        line_number: Line number where the issue occurs
        suggested_fix: Suggested fix with code snippet
        context: Additional context about the issue
        rule_id: Identifier for the rule that triggered this suggestion
    """

    category: SuggestionCategory
    severity: SuggestionSeverity
    message: str
    file_path: str
    line_number: int
    suggested_fix: Optional[str] = None
    context: Optional[str] = None
    rule_id: Optional[str] = None

    def format_markdown(self) -> str:
        """
        Format the suggestion as a markdown comment.

        Returns:
            Formatted markdown string suitable for PR comments
        """
        fix_text = f"\n  - {self.suggested_fix}" if self.suggested_fix else ""
        context_text = f"\n  - Context: {self.context}" if self.context else ""

        return (
            f"- **[{self.severity.value}]** {self.message} in `{self.file_path}:{self.line_number}`"
            f"{fix_text}{context_text}"
        )

    def to_dict(self) -> dict:
        """Convert suggestion to dictionary representation."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "suggested_fix": self.suggested_fix,
            "context": self.context,
            "rule_id": self.rule_id,
        }


@dataclass
class SuggestionGroup:
    """
    A group of suggestions organized by category.

    Attributes:
        category: The category for this group
        suggestions: List of suggestions in this category
    """

    category: SuggestionCategory
    suggestions: List[Suggestion] = field(default_factory=list)

    def add_suggestion(self, suggestion: Suggestion) -> None:
        """Add a suggestion to this group."""
        if suggestion.category != self.category:
            raise ValueError(
                f"Suggestion category {suggestion.category} does not match "
                f"group category {self.category}"
            )
        self.suggestions.append(suggestion)

    def format_markdown(self) -> str:
        """
        Format the suggestion group as markdown.

        Returns:
            Formatted markdown string
        """
        if not self.suggestions:
            return ""

        lines = [f"## {self.category.value}"]
        for suggestion in self.suggestions:
            lines.append(suggestion.format_markdown())
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert suggestion group to dictionary representation."""
        return {
            "category": self.category.value,
            "suggestions": [s.to_dict() for s in self.suggestions],
        }
