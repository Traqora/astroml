from astroml.utils.exceptions import AstroMLError
"""Pipeline test fixtures and runner for data pipeline testing.

Issue #638 Step 4 & 5: Implements reusable test fixtures for pipeline stages
and a comprehensive pipeline test runner with reporting.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline fixture
# ---------------------------------------------------------------------------


@dataclass
class PipelineFixture:
    """Reusable test fixture for a pipeline stage.

    Provides input data, expected output, and configuration for testing
    a specific pipeline stage.

    Example:
        fixture = PipelineFixture(
            name="feature_computation",
            input_data=pd.DataFrame({"tx_id": ["1", "2"], "amount": [100.0, 200.0]}),
            expected_output_columns={"tx_id", "in_degree", "out_degree"},
            config={"window_days": 7},
        )
    """

    name: str
    input_data: pd.DataFrame
    fixture_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    expected_output_columns: set[str] = field(default_factory=set)
    expected_row_count: int | None = None
    expected_row_count_range: tuple[int, int] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_csv(
        cls,
        name: str,
        input_path: str | Path,
        expected_output_path: str | Path | None = None,
        **kwargs: Any,
    ) -> PipelineFixture:
        """Create a fixture from CSV files.

        Args:
            name: Fixture name.
            input_path: Path to input CSV.
            expected_output_path: Optional path to expected output CSV.
            **kwargs: Additional fixture parameters.

        Returns:
            PipelineFixture instance.
        """
        input_data = pd.read_csv(input_path)
        fixture = cls(name=name, input_data=input_data, **kwargs)

        if expected_output_path:
            expected = pd.read_csv(expected_output_path)
            fixture.expected_output_columns = set(expected.columns)
            fixture.expected_row_count = len(expected)

        return fixture

    @classmethod
    def from_dicts(
        cls,
        name: str,
        input_dicts: list[dict[str, Any]],
        expected_output_dicts: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> PipelineFixture:
        """Create a fixture from lists of dictionaries.

        Args:
            name: Fixture name.
            input_dicts: List of input row dicts.
            expected_output_dicts: Optional list of expected output row dicts.
            **kwargs: Additional fixture parameters.

        Returns:
            PipelineFixture instance.
        """
        input_data = pd.DataFrame(input_dicts)
        fixture = cls(name=name, input_data=input_data, **kwargs)

        if expected_output_dicts:
            expected = pd.DataFrame(expected_output_dicts)
            fixture.expected_output_columns = set(expected.columns)
            fixture.expected_row_count = len(expected)

        return fixture

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "name": self.name,
            "input_rows": len(self.input_data),
            "input_columns": list(self.input_data.columns),
            "expected_output_columns": sorted(self.expected_output_columns),
            "expected_row_count": self.expected_row_count,
            "config": self.config,
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# Pipeline test runner
# ---------------------------------------------------------------------------


@dataclass
class PipelineTestRun:
    """Result of running a single pipeline fixture test."""

    fixture_name: str
    passed: bool
    error_message: str = ""
    output_row_count: int = 0
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_name": self.fixture_name,
            "passed": self.passed,
            "error_message": self.error_message,
            "output_row_count": self.output_row_count,
            "duration_ms": round(self.duration_ms, 2),
            "details": self.details,
        }


@dataclass
class PipelineTestReport:
    """Comprehensive pipeline test report from running multiple fixtures."""

    report_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    pipeline_name: str = ""
    runs: list[PipelineTestRun] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.runs)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.runs if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.runs if not r.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "pipeline_name": self.pipeline_name,
            "passed": self.passed,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "total_runs": len(self.runs),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "generated_at": self.generated_at.isoformat(),
            "runs": [r.to_dict() for r in self.runs],
        }

    def to_json(self, path: str | Path) -> None:
        """Write report to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Pipeline test report written to {path}")


class PipelineTestRunner:
    """Runs pipeline fixture tests and produces reports.

    Orchestrates the execution of pipeline fixtures, collecting results
    and generating comprehensive test reports.

    Example:
        runner = PipelineTestRunner(pipeline_name="feature_pipeline")

        runner.add_fixture(fixture_1)
        runner.add_fixture(fixture_2)

        runner.run(stage_fn=train_model)
        report = runner.generate_report()
        print(f"Tests passed: {report.pass_count}/{report.total_runs}")
    """

    def __init__(self, pipeline_name: str = "default") -> None:
        self.pipeline_name = pipeline_name
        self._fixtures: list[PipelineFixture] = []
        self._runs: list[PipelineTestRun] = []
        self._hooks: dict[str, list[Callable[[Any, Any], None]]] = {
            "before_run": [],
            "after_run": [],
            "on_pass": [],
            "on_fail": [],
        }

    def add_fixture(self, fixture: PipelineFixture) -> None:
        """Add a test fixture."""
        self._fixtures.append(fixture)

    def add_fixtures(self, fixtures: Sequence[PipelineFixture]) -> None:
        """Add multiple test fixtures."""
        self._fixtures.extend(fixtures)

    def add_hook(
        self,
        hook_name: str,
        callback: Callable[[Any, Any], None],
    ) -> None:
        """Add a lifecycle hook.

        Supported hooks: 'before_run', 'after_run', 'on_pass', 'on_fail'.

        Args:
            hook_name: Name of the hook.
            callback: Function to call (receives fixture, run_result).
        """
        if hook_name in self._hooks:
            self._hooks[hook_name].append(callback)

    def run(
        self,
        stage_fn: Callable[[pd.DataFrame, dict[str, Any]], Any],
        validate_fn: Callable[[Any, PipelineFixture], tuple[bool, str]] | None = None,
    ) -> list[PipelineTestRun]:
        """Run all fixtures against a pipeline stage.

        Args:
            stage_fn: The pipeline stage function. Receives (input_data, config).
            validate_fn: Optional custom validator. Receives (output, fixture),
                         returns (passed, error_message).

        Returns:
            List of PipelineTestRun results.
        """
        runs: list[PipelineTestRun] = []

        for fixture in self._fixtures:
            # Before hook
            for hook in self._hooks["before_run"]:
                try:
                    hook(fixture, None)
                except AstroMLError:
                    pass

            start = time.monotonic()

            try:
                output = stage_fn(fixture.input_data, fixture.config)
                elapsed = (time.monotonic() - start) * 1000

                if isinstance(output, pd.DataFrame):
                    output_count = len(output)
                elif isinstance(output, (list, tuple)):
                    output_count = len(output)
                else:
                    output_count = 1

                # Validate output
                passed, error = self._default_validate(output, fixture)

                if passed and validate_fn:
                    passed, error = validate_fn(output, fixture)

                run = PipelineTestRun(
                    fixture_name=fixture.name,
                    passed=passed,
                    error_message=error,
                    output_row_count=output_count,
                    duration_ms=elapsed,
                    details={
                        "fixture": fixture.to_dict(),
                    },
                )

            except AstroMLError as e:
                elapsed = (time.monotonic() - start) * 1000
                run = PipelineTestRun(
                    fixture_name=fixture.name,
                    passed=False,
                    error_message=f"Exception: {type(e).__name__}: {e}",
                    duration_ms=elapsed,
                )

            runs.append(run)

            # After hook
            for hook in self._hooks["after_run"]:
                try:
                    hook(fixture, run)
                except AstroMLError:
                    pass

            # Pass/fail hooks
            if run.passed:
                for hook in self._hooks["on_pass"]:
                    try:
                        hook(fixture, run)
                    except AstroMLError:
                        pass
            else:
                for hook in self._hooks["on_fail"]:
                    try:
                        hook(fixture, run)
                    except AstroMLError:
                        pass

        self._runs.extend(runs)
        return runs

    @staticmethod
    def _default_validate(output: Any, fixture: PipelineFixture) -> tuple[bool, str]:
        """Default validation logic for pipeline outputs.

        Args:
            output: Pipeline stage output.
            fixture: The test fixture.

        Returns:
            (passed, error_message).
        """
        import pandas as pd

        if isinstance(output, pd.DataFrame):
            # Check expected columns
            if fixture.expected_output_columns:
                actual_cols = set(output.columns)
                missing = fixture.expected_output_columns - actual_cols
                if missing:
                    return False, f"Missing expected columns: {sorted(missing)}"

            # Check row count
            if fixture.expected_row_count is not None:
                if len(output) != fixture.expected_row_count:
                    return False, (
                        f"Expected {fixture.expected_row_count} rows, got {len(output)}"
                    )

            if fixture.expected_row_count_range is not None:
                lo, hi = fixture.expected_row_count_range
                if not (lo <= len(output) <= hi):
                    return False, (
                        f"Expected row count in [{lo}, {hi}], got {len(output)}"
                    )

        return True, ""

    def generate_report(self) -> PipelineTestReport:
        """Generate a comprehensive test report of all runs."""
        report = PipelineTestReport(pipeline_name=self.pipeline_name)
        report.runs = list(self._runs)
        report.total_duration_ms = sum(r.duration_ms for r in self._runs)
        return report

    def clear(self) -> None:
        """Clear all fixtures and runs."""
        self._fixtures.clear()
        self._runs.clear()

    @property
    def fixtures(self) -> list[PipelineFixture]:
        return list(self._fixtures)

    @property
    def runs(self) -> list[PipelineTestRun]:
        return list(self._runs)


__all__ = [
    "PipelineFixture",
    "PipelineTestRun",
    "PipelineTestReport",
    "PipelineTestRunner",
]