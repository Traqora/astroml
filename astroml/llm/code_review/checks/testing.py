"""
Testing checks for code review.

This module implements testing-focused checks for code review,
including missing tests, weak assertions, and test coverage.
"""

import re
from typing import List

from astroml.llm.code_review.checks.security import BaseCheck
from astroml.llm.code_review.suggestions import (
    Suggestion,
    SuggestionCategory,
    SuggestionSeverity,
)


class TestingCheck(BaseCheck):
    """
    Testing-focused code review checks.

    Checks for testing issues including:
    - Missing tests
    - Weak assertions
    - Test coverage gaps
    - Test quality issues
    """

    def __init__(self):
        """Initialize the testing check."""
        self.patterns = self._init_patterns()

    def _init_patterns(self) -> dict:
        """Initialize testing issue patterns."""
        return {
            "assert_without_message": {
                "pattern": r'assert\s+\w+',
                "severity": SuggestionSeverity.LOW,
                "message": "Assert without message makes debugging harder",
                "suggested_fix": "Add a message to assert for better debugging",
            },
            "pass_in_test": {
                "pattern": r'def\s+test_\w+.*:\s*pass',
                "severity": SuggestionSeverity.MEDIUM,
                "message": "Test function with only 'pass' statement",
                "suggested_fix": "Implement the test or remove the function",
            },
            "print_in_test": {
                "pattern": r'print\s*\(',
                "severity": SuggestionSeverity.LOW,
                "message": "Print statement in test function",
                "suggested_fix": "Use assertions instead of print statements",
            },
            "no_assertions": {
                "pattern": r'def\s+test_\w+.*:(?!.*assert)',
                "severity": SuggestionSeverity.MEDIUM,
                "message": "Test function may lack assertions",
                "suggested_fix": "Add assertions to verify expected behavior",
            },
            "mock_unused": {
                "pattern": r'@patch.*\ndef\s+test_\w+',
                "severity": SuggestionSeverity.LOW,
                "message": "Mock decorator used but may not be utilized",
                "suggested_fix": "Ensure mock is properly used in the test",
            },
        }

    def check(self, content: str, file_path: str) -> List[Suggestion]:
        """
        Perform testing checks on the given content.

        Args:
            content: The code content to check
            file_path: Path to the file being checked

        Returns:
            List of testing suggestions found
        """
        suggestions = []

        lines = content.split("\n")
        for line_num, line_content in enumerate(lines, start=1):
            for rule_id, pattern_info in self.patterns.items():
                if re.search(pattern_info["pattern"], line_content):
                    suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.TESTING,
                            severity=pattern_info["severity"],
                            message=pattern_info["message"],
                            file_path=file_path,
                            line_number=line_num,
                            suggested_fix=pattern_info["suggested_fix"],
                            rule_id=f"testing_{rule_id}",
                        )
                    )

        return suggestions
