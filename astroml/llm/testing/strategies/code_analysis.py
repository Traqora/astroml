"""Code analysis based test generation strategy.

Analyzes function signatures, control flow, and error paths
to generate comprehensive test cases.
"""

from __future__ import annotations

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CodeAnalysisStrategy:
    """Strategy that generates tests by analyzing source code structure."""

    def __init__(self, source_code -> Any: str):
        self.source_code = source_code
        self.tree = ast.parse(source_code)

    def extract_functions(self) -> list[dict[str, Any]]:
        """Extract function signatures and docstrings from source."""
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "returns": (ast.dump(node.returns) if node.returns else None),
                    "docstring": ast.get_docstring(node) or "",
                    "lineno": node.lineno,
                    "decorators": [ast.dump(d) for d in node.decorator_list],
                }
                functions.append(func_info)
        return functions

    def identify_branches(self, function_name: str) -> list[dict[str, Any]]:
        """Identify conditional branches for edge case discovery."""
        branches = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                for child in ast.walk(node):
                    if isinstance(child, ast.If):
                        branches.append(
                            {
                                "test": ast.dump(child.test),
                                "lineno": child.lineno,
                            }
                        )
                    elif isinstance(child, (ast.Try, ast.ExceptHandler)):
                        branches.append(
                            {
                                "type": "exception",
                                "lineno": child.lineno,
                            }
                        )
        return branches

    def get_return_paths(self, function_name: str) -> list[str]:
        """Identify all possible return paths in a function."""
        paths = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        paths.append(ast.dump(child.value) if child.value else "None")
                    elif isinstance(child, ast.Raise):
                        paths.append(f"raises: {ast.dump(child.exc)}")
        return paths if paths else ["None"]

    def get_type_hints(self, function_name: str) -> dict[str, str]:
        """Extract type hints from function signature."""
        hints = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                for arg in node.args.args:
                    if arg.annotation:
                        hints[arg.arg] = ast.dump(arg.annotation)
                if node.returns:
                    hints["return"] = ast.dump(node.returns)
        return hints
