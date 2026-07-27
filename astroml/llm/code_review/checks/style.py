"""
Style checks for code review.

This module implements style-focused checks for code review,
including PEP8 compliance, naming conventions, and best practices.
"""

import re
from typing import List

from astroml.llm.code_review.checks.security import BaseCheck
from astroml.llm.code_review.suggestions import (
    Suggestion,
    SuggestionCategory,
    SuggestionSeverity,
)


class StyleCheck(BaseCheck):
    """
    Style-focused code review checks.

    Checks for style issues including:
    - PEP8 violations
    - Naming conventions
    - Code formatting
    - Best practices
    """

    def __init__(self):
        """Initialize the style check."""
        self.patterns = self._init_patterns()

    def _init_patterns(self) -> dict:
        """Initialize style issue patterns."""
        return {
            "line_too_long": {
                "pattern": r'.{120,}',
                "severity": SuggestionSeverity.LOW,
                "message": "Line exceeds 120 characters",
                "suggested_fix": "Break the line into multiple lines",
            },
            "trailing_whitespace": {
                "pattern": r'\s+$',
                "severity": SuggestionSeverity.LOW,
                "message": "Trailing whitespace",
                "suggested_fix": "Remove trailing whitespace",
            },
            "magic_number": {
                "pattern": r'\b\d{2,}\b',
                "severity": SuggestionSeverity.LOW,
                "message": "Magic number detected",
                "suggested_fix": "Extract to a named constant",
            },
            "camel_case_variable": {
                "pattern": r'[a-z][A-Z]',
                "severity": SuggestionSeverity.LOW,
                "message": "Variable name uses camelCase instead of snake_case",
                "suggested_fix": "Use snake_case for variable names",
            },
            "unused_import": {
                "pattern": r'^import\s+\w+.*$',
                "severity": SuggestionSeverity.LOW,
                "message": "Import may be unused (heuristic)",
                "suggested_fix": "Remove unused imports",
            },
            "commented_code": {
                "pattern": r'^\s*#.*[=;(){}\[\]]',
                "severity": SuggestionSeverity.LOW,
                "message": "Commented-out code detected",
                "suggested_fix": "Remove commented code or add TODO comment",
            },
        }

    def check(self, content: str, file_path: str) -> List[Suggestion]:
        """
        Perform style checks on the given content.

        Args:
            content: The code content to check
            file_path: Path to the file being checked

        Returns:
            List of style suggestions found
        """
        suggestions = []

        lines = content.split("\n")
        for line_num, line_content in enumerate(lines, start=1):
            for rule_id, pattern_info in self.patterns.items():
                if re.search(pattern_info["pattern"], line_content):
                    suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.STYLE,
                            severity=pattern_info["severity"],
                            message=pattern_info["message"],
                            file_path=file_path,
                            line_number=line_num,
                            suggested_fix=pattern_info["suggested_fix"],
                            rule_id=f"style_{rule_id}",
                        )
                    )

        return suggestions
