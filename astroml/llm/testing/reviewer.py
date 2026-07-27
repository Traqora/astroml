"""Test quality reviewer.

Reviews generated tests for correctness, coverage, and best practices.
Provides feedback for iterative improvement.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .generator import GeneratedTest, TestType

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Result of a test quality review."""
    score: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    coverage_tags: List[str] = field(default_factory=list)
    missing_coverage: List[str] = field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0


class TestReviewer:
    """Reviews generated tests for quality and completeness.

    Checks include:
    - Syntax validity
    - Assertion presence
    - Naming conventions
    - Coverage of edge cases
    - Use of fixtures
    - Mock usage
    """

    def __init__(self, source_code: Optional[str] = None):
        self.source_code = source_code

    def review_test(self, test: GeneratedTest) -> ReviewResult:
        """Review a single generated test case."""
        issues: List[str] = []
        suggestions: List[str] = []
        score = 100.0

        syntax_issues = self._check_syntax(test)
        issues.extend(syntax_issues)
        score -= len(syntax_issues) * 15

        assertion_issues = self._check_assertions(test)
        issues.extend(assertion_issues)
        if assertion_issues:
            score -= 20

        naming_issues = self._check_naming(test)
        issues.extend(naming_issues)
        score -= len(naming_issues) * 5

        coverage_tags = self._identify_coverage(test)

        return ReviewResult(
            score=max(0, score),
            issues=issues,
            suggestions=suggestions,
            coverage_tags=coverage_tags,
            passed_checks=0,
            failed_checks=len(issues),
        )

    def review_all(self, tests: List[GeneratedTest]) -> List[ReviewResult]:
        """Review all generated tests."""
        return [self.review_test(test) for test in tests]

    def _check_syntax(self, test: GeneratedTest) -> List[str]:
        issues = []
        full_code = f"\n{test.imports}\ndef {test.name}():\n{test.body}"
        try:
            ast.parse(full_code)
        except SyntaxError as e:
            issues.append(f"Syntax error in {test.name}: {e}")
        except Exception as e:
            issues.append(f"Parse error in {test.name}: {e}")
        return issues

    def _check_assertions(self, test: GeneratedTest) -> List[str]:
        issues = []
        if "assert" not in test.body and "pytest.raises" not in test.body:
            issues.append(f"No assertions found in {test.name}")
        return issues

    def _check_naming(self, test: GeneratedTest) -> List[str]:
        issues = []
        if not test.name.startswith("test_"):
            issues.append(f"Test name '{test.name}' should start with 'test_'")
        if "_" not in test.name:
            issues.append(f"Test name '{test.name}' should be descriptive with underscores")
        return issues

    def _identify_coverage(self, test: GeneratedTest) -> List[str]:
        tags = []
        body_lower = test.body.lower()

        if "empty" in body_lower or "none" in body_lower:
            tags.append("edge_case:null_input")
        if "empty" in body_lower:
            tags.append("edge_case:empty_input")
        if "negative" in body_lower or "invalid" in body_lower:
            tags.append("edge_case:invalid_input")
        if "large" in body_lower or "many" in body_lower or "long" in body_lower:
            tags.append("edge_case:large_input")
        if "raises" in body_lower or "exception" in body_lower or "error" in body_lower:
            tags.append("error_handling")
        if "boundary" in body_lower or "limit" in body_lower:
            tags.append("boundary_condition")
        if "mock" in body_lower or "patch" in body_lower:
            tags.append("mocking")

        return tags

    def generate_summary_report(
        self,
        results: List[ReviewResult],
    ) -> str:
        """Generate a human-readable summary report."""
        avg_score = sum(r.score for r in results) / max(len(results), 1)
        all_issues = [i for r in results for i in r.issues]
        all_suggestions = [s for r in results for s in r.suggestions]
        all_tags = set(t for r in results for t in r.coverage_tags)

        report = [
            f"Test Review Summary",
            f"{'=' * 40}",
            f"Tests reviewed: {len(results)}",
            f"Average score: {avg_score:.1f}/100",
            f"Issues found: {len(all_issues)}",
            f"",
            f"Coverage Tags:",
        ]
        for tag in sorted(all_tags):
            report.append(f"  - {tag}")
        report.append("")

        if all_issues:
            report.append("Issues:")
            for issue in all_issues[:10]:
                report.append(f"  - {issue}")

        return "\n".join(report)

    def get_improvement_suggestions(
        self,
        results: List[ReviewResult],
    ) -> List[str]:
        """Generate actionable improvement suggestions."""
        suggestions = []
        missing_assertions = sum(
            1 for r in results if "No assertions" in str(r.issues)
        )
        if missing_assertions > len(results) / 2:
            suggestions.append(
                "Most tests are missing assertions. Consider adding "
                "assert statements or pytest.raises context managers."
            )

        all_tags = set(t for r in results for t in r.coverage_tags)
        if "edge_case:null_input" not in all_tags:
            suggestions.append(
                "No null input tests detected. Add edge cases for None/null values."
            )
        if "edge_case:empty_input" not in all_tags:
            suggestions.append(
                "No empty input tests detected. Add edge cases for empty inputs."
            )

        return suggestions
