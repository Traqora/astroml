"""
Correctness checks for code review.

This module implements correctness-focused checks for code review,
including logic errors, edge cases, and potential bugs.
"""

import re
from typing import List

from astroml.llm.code_review.checks.security import BaseCheck
from astroml.llm.code_review.suggestions import (
    Suggestion,
    SuggestionCategory,
    SuggestionSeverity,
)


class CorrectnessCheck(BaseCheck):
    """
    Correctness-focused code review checks.

    Checks for correctness issues including:
    - Logic errors
    - Edge cases
    - Type errors
    - Null pointer risks
    - Off-by-one errors
    """

    def __init__(self):
        """Initialize the correctness check."""
        self.patterns = self._init_patterns()

    def _init_patterns(self) -> dict:
        """Initialize correctness issue patterns."""
        return {
            "none_comparison": {
                "pattern": r'==\s*None|!=\s*None',
                "severity": SuggestionSeverity.MEDIUM,
                "message": "Use 'is None' or 'is not None' instead of == or !=",
                "suggested_fix": "Use 'is None' or 'is not None' for None comparisons",
            },
            "mutable_default_arg": {
                "pattern": r'def\s+\w+\([^)]*=\s*\[|\{',
                "severity": SuggestionSeverity.HIGH,
                "message": "Mutable default argument detected",
                "suggested_fix": "Use None as default and initialize inside function",
            },
            "except_bare": {
                "pattern": r'except\s*:',
                "severity": SuggestionSeverity.HIGH,
                "message": "Bare except clause catches all exceptions",
                "suggested_fix": "Specify the exception type to catch",
            },
            "return_in_finally": {
                "pattern": r'finally:\s*return',
                "severity": SuggestionSeverity.HIGH,
                "message": "Return in finally block suppresses exceptions",
                "suggested_fix": "Move return outside finally block",
            },
            "unused_variable": {
                "pattern": r'_\s*=',
                "severity": SuggestionSeverity.LOW,
                "message": "Variable assigned but not used",
                "suggested_fix": "Remove unused variable or use proper naming",
            },
            "comparison_literal": {
                "pattern": r'(True|False)\s*==\s*\w+|\w+\s*==\s*(True|False)',
                "severity": SuggestionSeverity.LOW,
                "message": "Comparison with boolean literal is redundant",
                "suggested_fix": "Use the boolean directly or 'if not x'",
            },
        }

    def check(self, content: str, file_path: str) -> List[Suggestion]:
        """
        Perform correctness checks on the given content.

        Args:
            content: The code content to check
            file_path: Path to the file being checked

        Returns:
            List of correctness suggestions found
        """
        suggestions = []

        lines = content.split("\n")
        for line_num, line_content in enumerate(lines, start=1):
            for rule_id, pattern_info in self.patterns.items():
                if re.search(pattern_info["pattern"], line_content):
                    suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.CORRECTNESS,
                            severity=pattern_info["severity"],
                            message=pattern_info["message"],
                            file_path=file_path,
                            line_number=line_num,
                            suggested_fix=pattern_info["suggested_fix"],
                            rule_id=f"correctness_{rule_id}",
                        )
                    )

        return suggestions
