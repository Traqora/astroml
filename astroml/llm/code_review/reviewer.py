"""
Main code review logic.

This module implements the core code review functionality,
coordinating analyzers and checks to provide comprehensive code review.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set
from enum import Enum

from astroml.llm.code_review.suggestions import (
    Suggestion,
    SuggestionCategory,
    SuggestionGroup,
    SuggestionSeverity,
)
from astroml.llm.code_review.analyzers.python_analyzer import PythonAnalyzer
from astroml.llm.code_review.analyzers.sql_analyzer import SQLAnalyzer
from astroml.llm.code_review.analyzers.yaml_analyzer import YAMLAnalyzer
from astroml.llm.code_review.checks.security import SecurityCheck
from astroml.llm.code_review.checks.performance import PerformanceCheck
from astroml.llm.code_review.checks.style import StyleCheck
from astroml.llm.code_review.checks.correctness import CorrectnessCheck
from astroml.llm.code_review.checks.testing import TestingCheck
from astroml.llm.code_review.checks.documentation import DocumentationCheck
from astroml.llm.code_review.checks.complexity import ComplexityCheck


class ReviewStatus(Enum):
    """Status of a code review."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReviewResult:
    """
    Result of a code review.

    Attributes:
        status: The review status
        suggestions: List of all suggestions found
        suggestion_groups: Suggestions grouped by category
        files_reviewed: Number of files reviewed
        duration_seconds: Time taken for the review
        error: Optional error message if review failed
    """

    status: ReviewStatus
    suggestions: List[Suggestion] = field(default_factory=list)
    suggestion_groups: List[SuggestionGroup] = field(default_factory=list)
    files_reviewed: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None

    def format_markdown(self) -> str:
        """
        Format the review result as markdown.

        Returns:
            Formatted markdown string suitable for PR comments
        """
        if self.status == ReviewStatus.FAILED:
            return f"## Code Review Failed\n\nError: {self.error}"

        if not self.suggestions:
            return "## Code Review\n\nNo issues found! 🎉"

        lines = ["## Code Review Results"]
        lines.append(f"\nReviewed {self.files_reviewed} file(s) in {self.duration_seconds:.2f}s")
        lines.append(f"\nFound {len(self.suggestions)} issue(s):\n")

        for group in self.suggestion_groups:
            lines.append(group.format_markdown())
            lines.append("")

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, int]:
        """
        Get a summary of suggestions by category and severity.

        Returns:
            Dictionary with counts by category and severity
        """
        summary = {}

        for category in SuggestionCategory:
            category_key = category.value.lower()
            summary[category_key] = 0

        for severity in SuggestionSeverity:
            severity_key = f"{severity.value.lower()}_count"
            summary[severity_key] = 0

        for suggestion in self.suggestions:
            category_key = suggestion.category.value.lower()
            summary[category_key] = summary.get(category_key, 0) + 1

            severity_key = f"{suggestion.severity.value.lower()}_count"
            summary[severity_key] = summary.get(severity_key, 0) + 1

        return summary


class CodeReviewer:
    """
    Main code review engine.

    Coordinates language-specific analyzers and category-specific checks
    to provide comprehensive code review capabilities.
    """

    def __init__(
        self,
        enable_llm: bool = False,
        max_review_time: int = 120,
        ignored_rules: Optional[Set[str]] = None,
    ):
        """
        Initialize the code reviewer.

        Args:
            enable_llm: Whether to enable LLM-powered analysis
            max_review_time: Maximum time in seconds for review
            ignored_rules: Set of rule IDs to ignore
        """
        self.enable_llm = enable_llm
        self.max_review_time = max_review_time
        self.ignored_rules = ignored_rules or set()

        # Initialize analyzers
        self.analyzers = {
            ".py": PythonAnalyzer(),
            ".sql": SQLAnalyzer(),
            ".yaml": YAMLAnalyzer(),
            ".yml": YAMLAnalyzer(),
        }

        # Initialize checks
        self.checks = [
            SecurityCheck(),
            PerformanceCheck(),
            StyleCheck(),
            CorrectnessCheck(),
            TestingCheck(),
            DocumentationCheck(),
            ComplexityCheck(),
        ]

        # Learning from human decisions (simple storage)
        self.accepted_suggestions: Dict[str, int] = {}
        self.rejected_suggestions: Dict[str, int] = {}

    def review_diff(
        self, diff_content: str, file_path: str
    ) -> ReviewResult:
        """
        Review a git diff for code issues.

        Args:
            diff_content: The git diff content
            file_path: Path to the file being reviewed

        Returns:
            ReviewResult with suggestions found
        """
        start_time = time.time()
        suggestions = []

        try:
            # Get file extension
            file_ext = Path(file_path).suffix

            # Use language-specific analyzer if available
            if file_ext in self.analyzers:
                analyzer = self.analyzers[file_ext]
                analyzer_suggestions = analyzer.analyze_diff(diff_content, file_path)
                suggestions.extend(analyzer_suggestions)

            # Apply all checks to the diff content
            for check in self.checks:
                check_suggestions = check.check(diff_content, file_path)
                suggestions.extend(check_suggestions)

            # Filter out ignored rules
            suggestions = [
                s for s in suggestions if s.rule_id not in self.ignored_rules
            ]

            # Group suggestions by category
            suggestion_groups = self._group_suggestions(suggestions)

            duration = time.time() - start_time

            return ReviewResult(
                status=ReviewStatus.COMPLETED,
                suggestions=suggestions,
                suggestion_groups=suggestion_groups,
                files_reviewed=1,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ReviewResult(
                status=ReviewStatus.FAILED,
                error=str(e),
                duration_seconds=duration,
            )

    def review_file(self, file_path: str) -> ReviewResult:
        """
        Review a single file for code issues.

        Args:
            file_path: Path to the file to review

        Returns:
            ReviewResult with suggestions found
        """
        start_time = time.time()
        suggestions = []

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Get file extension
            file_ext = Path(file_path).suffix

            # Use language-specific analyzer if available
            if file_ext in self.analyzers:
                analyzer = self.analyzers[file_ext]
                analyzer_suggestions = analyzer.analyze_code(content, file_path)
                suggestions.extend(analyzer_suggestions)

            # Apply all checks to the content
            for check in self.checks:
                check_suggestions = check.check(content, file_path)
                suggestions.extend(check_suggestions)

            # Filter out ignored rules
            suggestions = [
                s for s in suggestions if s.rule_id not in self.ignored_rules
            ]

            # Group suggestions by category
            suggestion_groups = self._group_suggestions(suggestions)

            duration = time.time() - start_time

            return ReviewResult(
                status=ReviewStatus.COMPLETED,
                suggestions=suggestions,
                suggestion_groups=suggestion_groups,
                files_reviewed=1,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ReviewResult(
                status=ReviewStatus.FAILED,
                error=str(e),
                duration_seconds=duration,
            )

    def review_directory(
        self, directory_path: str, file_patterns: Optional[List[str]] = None
    ) -> ReviewResult:
        """
        Review all files in a directory.

        Args:
            directory_path: Path to the directory to review
            file_patterns: Optional list of file patterns to include

        Returns:
            ReviewResult with aggregated suggestions
        """
        start_time = time.time()
        all_suggestions = []
        files_reviewed = 0

        try:
            directory = Path(directory_path)

            # Default file patterns if none provided
            if file_patterns is None:
                file_patterns = ["*.py", "*.sql", "*.yaml", "*.yml"]

            # Find matching files
            for pattern in file_patterns:
                for file_path in directory.rglob(pattern):
                    if file_path.is_file():
                        result = self.review_file(str(file_path))
                        if result.status == ReviewStatus.COMPLETED:
                            all_suggestions.extend(result.suggestions)
                            files_reviewed += 1

            # Group suggestions by category
            suggestion_groups = self._group_suggestions(all_suggestions)

            duration = time.time() - start_time

            return ReviewResult(
                status=ReviewStatus.COMPLETED,
                suggestions=all_suggestions,
                suggestion_groups=suggestion_groups,
                files_reviewed=files_reviewed,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ReviewResult(
                status=ReviewStatus.FAILED,
                error=str(e),
                duration_seconds=duration,
            )

    def review_pr(
        self, pr_diff: Dict[str, str]
    ) -> ReviewResult:
        """
        Review a pull request by analyzing its diff.

        Args:
            pr_diff: Dictionary mapping file paths to their diff content

        Returns:
            ReviewResult with aggregated suggestions
        """
        start_time = time.time()
        all_suggestions = []
        files_reviewed = 0

        try:
            for file_path, diff_content in pr_diff.items():
                result = self.review_diff(diff_content, file_path)
                if result.status == ReviewStatus.COMPLETED:
                    all_suggestions.extend(result.suggestions)
                    files_reviewed += 1

            # Group suggestions by category
            suggestion_groups = self._group_suggestions(all_suggestions)

            duration = time.time() - start_time

            return ReviewResult(
                status=ReviewStatus.COMPLETED,
                suggestions=all_suggestions,
                suggestion_groups=suggestion_groups,
                files_reviewed=files_reviewed,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ReviewResult(
                status=ReviewStatus.FAILED,
                error=str(e),
                duration_seconds=duration,
            )

    def _group_suggestions(self, suggestions: List[Suggestion]) -> List[SuggestionGroup]:
        """
        Group suggestions by category.

        Args:
            suggestions: List of suggestions to group

        Returns:
            List of SuggestionGroup objects
        """
        groups = {}

        for category in SuggestionCategory:
            groups[category] = SuggestionGroup(category)

        for suggestion in suggestions:
            if suggestion.category in groups:
                groups[suggestion.category].add_suggestion(suggestion)

        # Return only non-empty groups
        return [group for group in groups.values() if group.suggestions]

    def record_feedback(self, rule_id: str, accepted: bool) -> None:
        """
        Record human feedback on a suggestion.

        Args:
            rule_id: The rule ID that generated the suggestion
            accepted: Whether the suggestion was accepted
        """
        if accepted:
            self.accepted_suggestions[rule_id] = (
                self.accepted_suggestions.get(rule_id, 0) + 1
            )
        else:
            self.rejected_suggestions[rule_id] = (
                self.rejected_suggestions.get(rule_id, 0) + 1
            )

    def get_rule_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics on rule performance.

        Returns:
            Dictionary mapping rule IDs to acceptance/rejection counts
        """
        stats = {}

        for rule_id in set(list(self.accepted_suggestions.keys()) + list(self.rejected_suggestions.keys())):
            stats[rule_id] = {
                "accepted": self.accepted_suggestions.get(rule_id, 0),
                "rejected": self.rejected_suggestions.get(rule_id, 0),
            }

        return stats

    def calculate_accuracy(self) -> float:
        """
        Calculate the accuracy of suggestions based on feedback.

        Returns:
            Accuracy percentage (0-100)
        """
        total_accepted = sum(self.accepted_suggestions.values())
        total_rejected = sum(self.rejected_suggestions.values())
        total = total_accepted + total_rejected

        if total == 0:
            return 0.0

        return (total_accepted / total) * 100
