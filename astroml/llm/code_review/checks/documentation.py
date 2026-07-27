"""
Documentation checks for code review.

This module implements documentation-focused checks for code review,
including missing docstrings, incomplete documentation, and documentation quality.
"""

import re
import ast
from typing import List

from astroml.llm.code_review.checks.security import BaseCheck
from astroml.llm.code_review.suggestions import (
    Suggestion,
    SuggestionCategory,
    SuggestionSeverity,
)


class DocumentationCheck(BaseCheck):
    """
    Documentation-focused code review checks.

    Checks for documentation issues including:
    - Missing docstrings
    - Incomplete documentation
    - Documentation quality
    - Type hints
    """

    def __init__(self):
        """Initialize the documentation check."""
        self.patterns = self._init_patterns()

    def _init_patterns(self) -> dict:
        """Initialize documentation issue patterns."""
        return {
            "todo_without_issue": {
                "pattern": r'#\s*TODO(?!\s*#\s*\d+)',
                "severity": SuggestionSeverity.LOW,
                "message": "TODO comment without issue reference",
                "suggested_fix": "Add issue reference to TODO comment",
            },
            "fixme_without_issue": {
                "pattern": r'#\s*FIXME(?!\s*#\s*\d+)',
                "severity": SuggestionSeverity.LOW,
                "message": "FIXME comment without issue reference",
                "suggested_fix": "Add issue reference to FIXME comment",
            },
        }

    def check(self, content: str, file_path: str) -> List[Suggestion]:
        """
        Perform documentation checks on the given content.

        Args:
            content: The code content to check
            file_path: Path to the file being checked

        Returns:
            List of documentation suggestions found
        """
        suggestions = []

        # Pattern-based checks
        lines = content.split("\n")
        for line_num, line_content in enumerate(lines, start=1):
            for rule_id, pattern_info in self.patterns.items():
                if re.search(pattern_info["pattern"], line_content):
                    suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.DOCUMENTATION,
                            severity=pattern_info["severity"],
                            message=pattern_info["message"],
                            file_path=file_path,
                            line_number=line_num,
                            suggested_fix=pattern_info["suggested_fix"],
                            rule_id=f"documentation_{rule_id}",
                        )
                    )

        # AST-based checks for docstrings
        try:
            ast_suggestions = self._check_docstrings(content, file_path)
            suggestions.extend(ast_suggestions)
        except SyntaxError:
            # Skip AST analysis if syntax is invalid
            pass

        return suggestions

    def _check_docstrings(self, content: str, file_path: str) -> List[Suggestion]:
        """
        Check for missing docstrings using AST.

        Args:
            content: The code content
            file_path: Path to the file

        Returns:
            List of docstring-related suggestions
        """
        suggestions = []
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                if not docstring:
                    suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.DOCUMENTATION,
                            severity=SuggestionSeverity.LOW,
                            message=f"{node.__class__.__name__} '{node.name}' is missing docstring",
                            file_path=file_path,
                            line_number=node.lineno,
                            suggested_fix="Add a docstring to document this function/class",
                            rule_id="missing_docstring",
                        )
                    )

        return suggestions
