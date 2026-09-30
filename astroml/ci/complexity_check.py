"""Cyclomatic complexity checking for CI (issue #507).

This module provides tools to:
1. Check that no function exceeds the max complexity threshold (15 for CI fail).
2. Report functions with complexity > 10 (warning threshold for new functions).
3. Exit with non-zero status if any function exceeds the hard limit.
"""

from __future__ import annotations

import ast
import collections
import sys
from pathlib import Path
from typing import List

#: Hard limit — CI will fail if any function exceeds this.
MAX_COMPLEXITY_HARD: int = 15
#: Soft limit — new functions should stay under this to comply with the
#: project complexity budget (documented in CONTRIBUTING.md).
MAX_COMPLEXITY_SOFT: int = 10


_ComplexityRecord = collections.namedtuple(
    "_ComplexityRecord", ["path", "name", "lineno", "complexity"]
)


def _count_branches(node: ast.AST) -> int:
    """Recursively count branching points in an AST node."""
    count = 1  # Base: one path through this node
    if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
        count += 1
    elif isinstance(node, ast.BoolOp):
        # Each 'and'/'or' adds an alternative path
        count += len(node.values) - 1
    elif isinstance(node, ast.Try):
        count += len(node.handlers) + len(node.finalbody)
    elif isinstance(node, (ast.ExceptHandler, ast.With, ast.AsyncWith)):
        count += 1
    for child in ast.iter_child_nodes(node):
        count += _count_branches(child)
    return count


def cyclomatic_complexity(node: ast.AST) -> int:
    """Compute McCabe cyclomatic complexity for a function/class body."""
    # M = 1 (base) + number of decision points
    branches = _count_branches(node) - 1  # subtract the base 1
    return max(1, branches)


def _check_file(path: Path) -> List[_ComplexityRecord]:
    """Analyse a single .py file and return records above thresholds."""
    records: List[_ComplexityRecord] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        # Silently skip files that can't be parsed (e.g. generated code).
        return records

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cc = cyclomatic_complexity(node)
            if cc > MAX_COMPLEXITY_SOFT:
                records.append(
                    _ComplexityRecord(
                        path=str(path),
                        name=node.name,
                        lineno=node.lineno,
                        complexity=cc,
                    )
                )
    return records


def check_complexity(
    paths: List[Path],
    hard_limit: int = MAX_COMPLEXITY_HARD,
    soft_limit: int = MAX_COMPLEXITY_SOFT,
) -> int:
    """Check complexity of all Python files under *paths*.

    Args:
        paths: List of directories/files to scan.
        hard_limit: Functions above this cause a non-zero exit.
        soft_limit: Functions above this are reported as warnings.

    Returns:
        Exit code (0 = pass, 1 = hard limit violations found).
    """
    all_records: List[_ComplexityRecord] = []
    for p in paths:
        if p.is_dir():
            for py_file in sorted(p.rglob("*.py")):
                all_records.extend(_check_file(py_file))
        elif p.suffix == ".py":
            all_records.extend(_check_file(p))

    hard_violations = [r for r in all_records if r.complexity > hard_limit]
    soft_violations = [r for r in all_records if hard_limit >= r.complexity > soft_limit]

    if soft_violations:
        print("=== Functions exceeding soft limit (%d) ===" % soft_limit)
        for r in sorted(soft_violations, key=lambda x: -x.complexity):
            print(f"  {r.path}:{r.lineno}  {r.name}  (CC={r.complexity})")

    if hard_violations:
        print("\n=== Functions exceeding HARD limit (%d) — FAIL ===" % hard_limit)
        for r in sorted(hard_violations, key=lambda x: -x.complexity):
            print(f"  {r.path}:{r.lineno}  {r.name}  (CC={r.complexity})")
        return 1

    if not soft_violations and not hard_violations:
        print("All functions within complexity budget.")
    return 0


def main() -> int:
    """CLI entry point."""
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else [Path("astroml")]
    return check_complexity(paths)


if __name__ == "__main__":
    sys.exit(main())
