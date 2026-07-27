"""
Security checks for code review.

This module implements security-focused checks for code review,
including SQL injection, XSS, authentication issues, and more.
"""

from abc import ABC, abstractmethod
from typing import List

from astroml.llm.code_review.suggestions import (
    Suggestion,
    SuggestionCategory,
    SuggestionSeverity,
)


class BaseCheck(ABC):
    """Base class for code review checks."""

    @abstractmethod
    def check(self, content: str, file_path: str) -> List[Suggestion]:
        """
        Perform the check on the given content.

        Args:
            content: The code content to check
            file_path: Path to the file being checked

        Returns:
            List of suggestions found
        """
        pass


class SecurityCheck(BaseCheck):
    """
    Security-focused code review checks.

    Checks for common security vulnerabilities including:
    - SQL injection
    - XSS vulnerabilities
    - Authentication issues
    - Hardcoded secrets
    - Insecure dependencies
    """

    def __init__(self):
        """Initialize the security check."""
        self.vulnerability_patterns = self._init_patterns()

    def _init_patterns(self) -> dict:
        """Initialize security vulnerability patterns."""
        return {
            "sql_injection_fstring": {
                "pattern": r'execute\s*\(\s*f["\'].*\{.*\}.*["\']',
                "severity": SuggestionSeverity.HIGH,
                "message": "SQL injection risk via f-string",
                "suggested_fix": "Use parameterized queries with ? or %s placeholders",
            },
            "xss_render": {
                "pattern": r'render\s*\(\s*.*\|\s*safe',
                "severity": SuggestionSeverity.HIGH,
                "message": "XSS risk: using | safe filter on user input",
                "suggested_fix": "Avoid using | safe on untrusted user input",
            },
            "hardcoded_password": {
                "pattern": r'(password|passwd|secret)\s*=\s*["\'][^"\']{8,}["\']',
                "severity": SuggestionSeverity.HIGH,
                "message": "Hardcoded password detected",
                "suggested_fix": "Use environment variables or secret management",
            },
            "weak_hash": {
                "pattern": r'(md5|sha1)\s*\(',
                "severity": SuggestionSeverity.MEDIUM,
                "message": "Weak cryptographic hash algorithm",
                "suggested_fix": "Use stronger algorithms like SHA-256 or SHA-512",
            },
            "random_not_crypto": {
                "pattern": r'import\s+random\s*$',
                "severity": SuggestionSeverity.MEDIUM,
                "message": "Using random module for security-sensitive operations",
                "suggested_fix": "Use secrets module for cryptographic operations",
            },
            "verify_disabled_ssl": {
                "pattern": r'verify\s*=\s*False',
                "severity": SuggestionSeverity.HIGH,
                "message": "SSL verification disabled",
                "suggested_fix": "Enable SSL verification for secure connections",
            },
        }

    def check(self, content: str, file_path: str) -> List[Suggestion]:
        """
        Perform security checks on the given content.

        Args:
            content: The code content to check
            file_path: Path to the file being checked

        Returns:
            List of security suggestions found
        """
        suggestions = []
        import re

        lines = content.split("\n")
        for line_num, line_content in enumerate(lines, start=1):
            for rule_id, pattern_info in self.vulnerability_patterns.items():
                if re.search(pattern_info["pattern"], line_content, re.IGNORECASE):
                    suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.SECURITY,
                            severity=pattern_info["severity"],
                            message=pattern_info["message"],
                            file_path=file_path,
                            line_number=line_num,
                            suggested_fix=pattern_info["suggested_fix"],
                            rule_id=f"security_{rule_id}",
                        )
                    )

        return suggestions
