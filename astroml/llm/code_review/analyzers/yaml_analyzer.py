"""
YAML-specific code analyzer.

This module provides analysis capabilities for YAML code,
including pattern matching for common YAML issues.
"""

import re
from typing import List, Dict, Any

from astroml.llm.code_review.suggestions import (
    Suggestion,
    SuggestionCategory,
    SuggestionSeverity,
)


class YAMLAnalyzer:
    """
    Analyzer for YAML code.

    Performs static analysis on YAML code to identify potential issues
    related to security, correctness, and best practices.
    """

    def __init__(self):
        """Initialize the YAML analyzer."""
        self.patterns = self._init_patterns()

    def _init_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize YAML analysis patterns."""
        return {
            "hardcoded_secret": {
                "pattern": re.compile(
                    r'(password|secret|api_key|token):\s*["\']?[^\s"\']{8,}["\']?',
                    re.IGNORECASE,
                ),
                "severity": SuggestionSeverity.HIGH,
                "category": SuggestionCategory.SECURITY,
                "message": "Potentially hardcoded secret in YAML",
                "suggested_fix": "Use environment variable references or secret management",
            },
            "debug_enabled": {
                "pattern": re.compile(r'debug:\s*true', re.IGNORECASE),
                "severity": SuggestionSeverity.MEDIUM,
                "category": SuggestionCategory.SECURITY,
                "message": "Debug mode enabled in configuration",
                "suggested_fix": "Disable debug mode in production configurations",
            },
            "insecure_port": {
                "pattern": re.compile(r'port:\s*(80|8080|5000)\s*$', re.MULTILINE),
                "severity": SuggestionSeverity.LOW,
                "category": SuggestionCategory.SECURITY,
                "message": "Using non-HTTPS port",
                "suggested_fix": "Use HTTPS ports (443, 8443) in production",
            },
            "missing_quotes": {
                "pattern": re.compile(r'^\s*\w+:\s*[^"\'].*$', re.MULTILINE),
                "severity": SuggestionSeverity.LOW,
                "category": SuggestionCategory.STYLE,
                "message": "Unquoted value may cause type ambiguity",
                "suggested_fix": "Quote string values to avoid type ambiguity",
            },
        }

    def analyze_diff(
        self, diff_content: str, file_path: str
    ) -> List[Suggestion]:
        """
        Analyze a git diff for YAML code issues.

        Args:
            diff_content: The git diff content
            file_path: Path to the file being analyzed

        Returns:
            List of suggestions found in the diff
        """
        suggestions = []
        added_lines = self._extract_added_lines(diff_content)

        for line_num, line_content in added_lines:
            suggestions.extend(self._analyze_line(line_content, file_path, line_num))

        return suggestions

    def analyze_code(
        self, content: str, file_path: str
    ) -> List[Suggestion]:
        """
        Analyze YAML code content for issues.

        Args:
            content: The YAML code content
            file_path: Path to the file being analyzed

        Returns:
            List of suggestions found in the code
        """
        suggestions = []
        lines = content.split("\n")

        for line_num, line_content in enumerate(lines, start=1):
            suggestions.extend(self._analyze_line(line_content, file_path, line_num))

        return suggestions

    def _extract_added_lines(self, diff_content: str) -> List[tuple]:
        """
        Extract added lines from a git diff.

        Args:
            diff_content: The git diff content

        Returns:
            List of (line_number, line_content) tuples for added lines
        """
        added_lines = []
        current_line_num = 0

        for line in diff_content.split("\n"):
            if line.startswith("@@"):
                match = re.search(r'\+(\d+)', line)
                if match:
                    current_line_num = int(match.group(1))
            elif line.startswith("+") and not line.startswith("+++"):
                added_lines.append((current_line_num, line[1:]))
                current_line_num += 1
            elif not line.startswith("-") and not line.startswith("\\"):
                current_line_num += 1

        return added_lines

    def _analyze_line(
        self, line_content: str, file_path: str, line_number: int
    ) -> List[Suggestion]:
        """
        Analyze a single line of YAML code.

        Args:
            line_content: The line content to analyze
            file_path: Path to the file
            line_number: Line number

        Returns:
            List of suggestions found in this line
        """
        suggestions = []

        for rule_id, pattern_info in self.patterns.items():
            if pattern_info["pattern"].search(line_content):
                suggestions.append(
                    Suggestion(
                        category=pattern_info["category"],
                        severity=pattern_info["severity"],
                        message=pattern_info["message"],
                        file_path=file_path,
                        line_number=line_number,
                        suggested_fix=pattern_info["suggested_fix"],
                        rule_id=rule_id,
                    )
                )

        return suggestions
