"""
Performance checks for code review.

This module implements performance-focused checks for code review,
including inefficient algorithms, memory leaks, and N+1 queries.
"""

import re
from typing import List

from astroml.llm.code_review.checks.security import BaseCheck
from astroml.llm.code_review.suggestions import (
    Suggestion,
    SuggestionCategory,
    SuggestionSeverity,
)


class PerformanceCheck(BaseCheck):
    """
    Performance-focused code review checks.

    Checks for common performance issues including:
    - Inefficient algorithms
    - Memory leaks
    - N+1 queries
    - Unnecessary computations
    - Poor database usage
    """

    def __init__(self):
        """Initialize the performance check."""
        self.patterns = self._init_patterns()

    def _init_patterns(self) -> dict:
        """Initialize performance issue patterns."""
        return {
            "nested_loop_o_n2": {
                "pattern": r'for\s+\w+\s+in\s+.*:\s*for\s+\w+\s+in\s+.*:',
                "severity": SuggestionSeverity.MEDIUM,
                "message": "Nested loops may indicate O(n²) complexity",
                "suggested_fix": "Consider using sets, dictionaries, or more efficient algorithms",
            },
            "list_append_in_loop": {
                "pattern": r'for\s+\w+\s+in\s+.*:\s*.*\.append\(',
                "severity": SuggestionSeverity.LOW,
                "message": "List append in loop - consider list comprehension",
                "suggested_fix": "Use list comprehension for better performance",
            },
            "string_concat_loop": {
                "pattern": r'\+=\s*["\']',
                "severity": SuggestionSeverity.MEDIUM,
                "message": "String concatenation in loop is inefficient",
                "suggested_fix": "Use list and join() for better performance",
            },
            "global_variable_mutation": {
                "pattern": r'global\s+\w+',
                "severity": SuggestionSeverity.LOW,
                "message": "Global variable mutation can impact performance",
                "suggested_fix": "Consider using function arguments or class attributes",
            },
            "database_query_in_loop": {
                "pattern": r'(execute|query|select)\s*\(',
                "severity": SuggestionSeverity.HIGH,
                "message": "Database query inside loop - potential N+1 issue",
                "suggested_fix": "Use batch operations or eager loading",
            },
            "synchronous_io_in_async": {
                "pattern": r'(time\.sleep|requests\.get|urllib\.request)',
                "severity": SuggestionSeverity.HIGH,
                "message": "Synchronous I/O in async context",
                "suggested_fix": "Use async alternatives (aiohttp, asyncio.sleep)",
            },
        }

    def check(self, content: str, file_path: str) -> List[Suggestion]:
        """
        Perform performance checks on the given content.

        Args:
            content: The code content to check
            file_path: Path to the file being checked

        Returns:
            List of performance suggestions found
        """
        suggestions = []

        lines = content.split("\n")
        for line_num, line_content in enumerate(lines, start=1):
            for rule_id, pattern_info in self.patterns.items():
                if re.search(pattern_info["pattern"], line_content, re.IGNORECASE):
                    suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.PERFORMANCE,
                            severity=pattern_info["severity"],
                            message=pattern_info["message"],
                            file_path=file_path,
                            line_number=line_num,
                            suggested_fix=pattern_info["suggested_fix"],
                            rule_id=f"performance_{rule_id}",
                        )
                    )

        return suggestions
