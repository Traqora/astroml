"""
Complexity checks for code review.

This module implements complexity-focused checks for code review,
including cyclomatic complexity, function length, and nesting depth.
"""

import ast
from typing import List

from astroml.llm.code_review.checks.security import BaseCheck
from astroml.llm.code_review.suggestions import (
    Suggestion,
    SuggestionCategory,
    SuggestionSeverity,
)


class ComplexityCheck(BaseCheck):
    """
    Complexity-focused code review checks.

    Checks for complexity issues including:
    - Cyclomatic complexity
    - Function length
    - Nesting depth
    - Parameter count
    """

    def __init__(self):
        """Initialize the complexity check."""
        self.max_complexity = 10
        self.max_function_length = 50
        self.max_nesting_depth = 4
        self.max_parameters = 7

    def check(self, content: str, file_path: str) -> List[Suggestion]:
        """
        Perform complexity checks on the given content.

        Args:
            content: The code content to check
            file_path: Path to the file being checked

        Returns:
            List of complexity suggestions found
        """
        suggestions = []

        try:
            tree = ast.parse(content)
            suggestions.extend(self._check_complexity(tree, file_path))
            suggestions.extend(self._check_function_length(tree, file_path))
            suggestions.extend(self._check_nesting_depth(tree, file_path))
            suggestions.extend(self._check_parameter_count(tree, file_path))
        except SyntaxError:
            # Skip AST analysis if syntax is invalid
            pass

        return suggestions

    def _check_complexity(self, tree: ast.AST, file_path: str) -> List[Suggestion]:
        """Check cyclomatic complexity of functions."""
        suggestions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_complexity(node)
                if complexity > self.max_complexity:
                    suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.COMPLEXITY,
                            severity=SuggestionSeverity.MEDIUM,
                            message=f"Function '{node.name}' has high cyclomatic complexity ({complexity})",
                            file_path=file_path,
                            line_number=node.lineno,
                            suggested_fix="Consider breaking this function into smaller functions",
                            rule_id="high_complexity",
                        )
                    )

        return suggestions

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(
                child,
                (
                    ast.If,
                    ast.While,
                    ast.For,
                    ast.AsyncFor,
                    ast.ExceptHandler,
                    ast.With,
                    ast.AsyncWith,
                ),
            ):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _check_function_length(self, tree: ast.AST, file_path: str) -> List[Suggestion]:
        """Check function length in lines."""
        suggestions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1 if node.end_lineno else 0
                if length > self.max_function_length:
                    suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.COMPLEXITY,
                            severity=SuggestionSeverity.MEDIUM,
                            message=f"Function '{node.name}' is too long ({length} lines)",
                            file_path=file_path,
                            line_number=node.lineno,
                            suggested_fix="Consider breaking this function into smaller functions",
                            rule_id="long_function",
                        )
                    )

        return suggestions

    def _check_nesting_depth(self, tree: ast.AST, file_path: str) -> List[Suggestion]:
        """Check nesting depth of code blocks."""
        suggestions = []

        class NestingDepthVisitor(ast.NodeVisitor):
            def __init__(self, max_depth: int):
                self.max_depth = max_depth
                self.suggestions = []
                self.current_depth = 0

            def visit_If(self, node: ast.If) -> None:
                self.current_depth += 1
                if self.current_depth > self.max_depth:
                    self.suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.COMPLEXITY,
                            severity=SuggestionSeverity.MEDIUM,
                            message=f"Nesting depth {self.current_depth} exceeds maximum",
                            file_path=file_path,
                            line_number=node.lineno,
                            suggested_fix="Consider extracting nested logic into separate functions",
                            rule_id="deep_nesting",
                        )
                    )
                self.generic_visit(node)
                self.current_depth -= 1

            def visit_For(self, node: ast.For) -> None:
                self.current_depth += 1
                if self.current_depth > self.max_depth:
                    self.suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.COMPLEXITY,
                            severity=SuggestionSeverity.MEDIUM,
                            message=f"Nesting depth {self.current_depth} exceeds maximum",
                            file_path=file_path,
                            line_number=node.lineno,
                            suggested_fix="Consider extracting nested logic into separate functions",
                            rule_id="deep_nesting",
                        )
                    )
                self.generic_visit(node)
                self.current_depth -= 1

            def visit_While(self, node: ast.While) -> None:
                self.current_depth += 1
                if self.current_depth > self.max_depth:
                    self.suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.COMPLEXITY,
                            severity=SuggestionSeverity.MEDIUM,
                            message=f"Nesting depth {self.current_depth} exceeds maximum",
                            file_path=file_path,
                            line_number=node.lineno,
                            suggested_fix="Consider extracting nested logic into separate functions",
                            rule_id="deep_nesting",
                        )
                    )
                self.generic_visit(node)
                self.current_depth -= 1

        visitor = NestingDepthVisitor(self.max_nesting_depth)
        visitor.visit(tree)

        return visitor.suggestions

    def _check_parameter_count(self, tree: ast.AST, file_path: str) -> List[Suggestion]:
        """Check parameter count of functions."""
        suggestions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                param_count = len(node.args.args)
                if param_count > self.max_parameters:
                    suggestions.append(
                        Suggestion(
                            category=SuggestionCategory.COMPLEXITY,
                            severity=SuggestionSeverity.LOW,
                            message=f"Function '{node.name}' has many parameters ({param_count})",
                            file_path=file_path,
                            line_number=node.lineno,
                            suggested_fix="Consider using a dataclass or configuration object",
                            rule_id="many_parameters",
                        )
                    )

        return suggestions
