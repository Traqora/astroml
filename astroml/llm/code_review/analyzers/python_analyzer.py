"""
Python-specific code analyzer.

This module provides analysis capabilities for Python code,
including AST parsing and pattern matching for common issues.
"""

import ast
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from astroml.llm.code_review.suggestions import (
    Suggestion,
    SuggestionCategory,
    SuggestionSeverity,
)


@dataclass
class CodeContext:
    """Context information for code analysis."""

    file_path: str
    content: str
    line_offset: int = 0


class PythonAnalyzer:
    """
    Analyzer for Python code.

    Performs static analysis on Python code to identify potential issues
    related to security, performance, style, and correctness.
    """

    def __init__(self):
        """Initialize the Python analyzer."""
        self.security_patterns = self._init_security_patterns()
        self.performance_patterns = self._init_performance_patterns()

    def _init_security_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize security vulnerability patterns."""
        return {
            "sql_injection": {
                "pattern": re.compile(
                    r'(execute|executemany)\s*\(\s*[f"\'].*\{.*\}.*[f"\']\s*\)',
                    re.IGNORECASE,
                ),
                "severity": SuggestionSeverity.HIGH,
                "message": "SQL injection risk - use parameterized queries",
                "suggested_fix": "Use parameterized queries with ? or %s placeholders",
            },
            "eval_usage": {
                "pattern": re.compile(r'\beval\s*\(', re.IGNORECASE),
                "severity": SuggestionSeverity.HIGH,
                "message": "Use of eval() is dangerous",
                "suggested_fix": "Replace with safer alternatives like ast.literal_eval",
            },
            "exec_usage": {
                "pattern": re.compile(r'\bexec\s*\(', re.IGNORECASE),
                "severity": SuggestionSeverity.HIGH,
                "message": "Use of exec() is dangerous",
                "suggested_fix": "Remove exec() or use safer alternatives",
            },
            "shell_injection": {
                "pattern": re.compile(
                    r'(os\.system|subprocess\.(call|run|Popen))\s*\(\s*[f"\'].*\{.*\}.*[f"\']',
                    re.IGNORECASE,
                ),
                "severity": SuggestionSeverity.HIGH,
                "message": "Shell injection risk",
                "suggested_fix": "Use subprocess with shell=False and parameterized arguments",
            },
            "hardcoded_secrets": {
                "pattern": re.compile(
                    r'(password|secret|api_key|token)\s*=\s*["\'][^"\']{8,}["\']',
                    re.IGNORECASE,
                ),
                "severity": SuggestionSeverity.HIGH,
                "message": "Potentially hardcoded secret detected",
                "suggested_fix": "Use environment variables or secret management",
            },
        }

    def _init_performance_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize performance issue patterns."""
        return {
            "string_concatenation": {
                "pattern": re.compile(r'\+\s*=\s*["\']', re.IGNORECASE),
                "severity": SuggestionSeverity.MEDIUM,
                "message": "Inefficient string concatenation in loop",
                "suggested_fix": "Use list comprehension and join() for better performance",
            },
            "global_import": {
                "pattern": re.compile(r'^import\s+.*\s*$', re.MULTILINE),
                "severity": SuggestionSeverity.LOW,
                "message": "Consider moving imports to module level",
                "suggested_fix": "Move imports to top of file for better performance",
            },
        }

    def analyze_diff(
        self, diff_content: str, file_path: str
    ) -> List[Suggestion]:
        """
        Analyze a git diff for Python code issues.

        Args:
            diff_content: The git diff content
            file_path: Path to the file being analyzed

        Returns:
            List of suggestions found in the diff
        """
        suggestions = []

        # Extract added lines from diff
        added_lines = self._extract_added_lines(diff_content)

        for line_num, line_content in added_lines:
            suggestions.extend(self._analyze_line(line_content, file_path, line_num))

        return suggestions

    def analyze_code(
        self, content: str, file_path: str
    ) -> List[Suggestion]:
        """
        Analyze Python code content for issues.

        Args:
            content: The Python code content
            file_path: Path to the file being analyzed

        Returns:
            List of suggestions found in the code
        """
        suggestions = []

        # Pattern-based analysis
        lines = content.split("\n")
        for line_num, line_content in enumerate(lines, start=1):
            suggestions.extend(self._analyze_line(line_content, file_path, line_num))

        # AST-based analysis
        try:
            ast_suggestions = self._analyze_ast(content, file_path)
            suggestions.extend(ast_suggestions)
        except SyntaxError:
            # Skip AST analysis if syntax is invalid
            pass

        return suggestions

    def _extract_added_lines(
        self, diff_content: str
    ) -> List[tuple]:
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
                # Extract line number from hunk header
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
        Analyze a single line of code.

        Args:
            line_content: The line content to analyze
            file_path: Path to the file
            line_number: Line number

        Returns:
            List of suggestions found in this line
        """
        suggestions = []

        # Check security patterns
        for rule_id, pattern_info in self.security_patterns.items():
            if pattern_info["pattern"].search(line_content):
                suggestions.append(
                    Suggestion(
                        category=SuggestionCategory.SECURITY,
                        severity=pattern_info["severity"],
                        message=pattern_info["message"],
                        file_path=file_path,
                        line_number=line_number,
                        suggested_fix=pattern_info["suggested_fix"],
                        rule_id=rule_id,
                    )
                )

        # Check performance patterns
        for rule_id, pattern_info in self.performance_patterns.items():
            if pattern_info["pattern"].search(line_content):
                suggestions.append(
                    Suggestion(
                        category=SuggestionCategory.PERFORMANCE,
                        severity=pattern_info["severity"],
                        message=pattern_info["message"],
                        file_path=file_path,
                        line_number=line_number,
                        suggested_fix=pattern_info["suggested_fix"],
                        rule_id=rule_id,
                    )
                )

        return suggestions

    def _analyze_ast(self, content: str, file_path: str) -> List[Suggestion]:
        """
        Analyze Python code using AST.

        Args:
            content: The Python code content
            file_path: Path to the file

        Returns:
            List of suggestions found via AST analysis
        """
        suggestions = []
        tree = ast.parse(content)

        class ComplexityVisitor(ast.NodeVisitor):
            """AST visitor to check complexity issues."""

            def __init__(self, suggestions: List[Suggestion], file_path: str):
                self.suggestions = suggestions
                self.file_path = file_path

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                """Check function complexity."""
                # Count cyclomatic complexity
                complexity = 1  # Base complexity
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1

                if complexity > 10:
                    self.suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.COMPLEXITY,
                            severity=SuggestionSeverity.MEDIUM,
                            message=f"Function '{node.name}' has high complexity ({complexity})",
                            file_path=self.file_path,
                            line_number=node.lineno,
                            suggested_fix="Consider breaking this function into smaller functions",
                            rule_id="high_complexity",
                        )
                    )

                # Check for missing docstring
                docstring = ast.get_docstring(node)
                if not docstring:
                    self.suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.DOCUMENTATION,
                            severity=SuggestionSeverity.LOW,
                            message=f"Function '{node.name}' is missing docstring",
                            file_path=self.file_path,
                            line_number=node.lineno,
                            suggested_fix="Add a docstring to document the function",
                            rule_id="missing_docstring",
                        )
                    )

                self.generic_visit(node)

        visitor = ComplexityVisitor(suggestions, file_path)
        visitor.visit(tree)

        return suggestions
