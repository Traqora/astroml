"""Pipeline integrity checks for data pipeline testing.

Issue #638 Step 3: Implements pipeline integrity verification including
stage ordering, data contract enforcement, and pipeline graph validation.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class IntegrityCheckSeverity(Enum):
    """Severity of integrity check failures."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IntegrityCheckType(Enum):
    """Types of pipeline integrity checks."""

    INPUT_OUTPUT_MATCH = "input_output_match"
    STAGE_ORDERING = "stage_ordering"
    DATA_CONTRACT = "data_contract"
    IDEMPOTENCY = "idempotency"
    DETERMINISM = "determinism"
    RESOURCE_BOUNDS = "resource_bounds"
    NO_DATA_LOSS = "no_data_loss"


@dataclass
class IntegrityCheckResult:
    """Result of a single integrity check."""

    check_name: str
    check_type: IntegrityCheckType
    passed: bool
    severity: IntegrityCheckSeverity = IntegrityCheckSeverity.ERROR
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "check_type": self.check_type.name,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class IntegrityReport:
    """Comprehensive pipeline integrity report."""

    pipeline_name: str
    report_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    checks: list[IntegrityCheckResult] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if no CRITICAL or ERROR checks failed."""
        return all(
            c.passed or c.severity in (IntegrityCheckSeverity.INFO, IntegrityCheckSeverity.WARNING)
            for c in self.checks
        )

    @property
    def critical_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == IntegrityCheckSeverity.CRITICAL)

    @property
    def error_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == IntegrityCheckSeverity.ERROR)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "pipeline_name": self.pipeline_name,
            "passed": self.passed,
            "checks_total": len(self.checks),
            "checks_passed": sum(1 for c in self.checks if c.passed),
            "checks_failed": sum(1 for c in self.checks if not c.passed),
            "critical_count": self.critical_count,
            "error_count": self.error_count,
            "generated_at": self.generated_at.isoformat(),
            "checks": [c.to_dict() for c in self.checks],
        }


class IntegrityChecker:
    """Runs integrity checks on data pipeline stages.

    Provides checks for pipeline correctness: input/output matching,
    stage ordering, idempotency, determinism, and data loss detection.

    Example:
        checker = IntegrityChecker(pipeline_name="feature_pipeline")

        # Check that stage outputs match expected schemas
        checker.check_input_output_match(stage_name="ingestion", input_count=1000, output_count=1000)

        # Check that the pipeline is deterministic
        result = checker.check_determinism(
            stage_fn=my_stage,
            input_data=sample_data,
            num_runs=3,
        )

        report = checker.generate_report()
    """

    def __init__(self, pipeline_name: str = "default") -> None:
        self.pipeline_name = pipeline_name
        self._checks: list[IntegrityCheckResult] = []

    def _record(
        self,
        check_name: str,
        check_type: IntegrityCheckType,
        passed: bool,
        severity: IntegrityCheckSeverity = IntegrityCheckSeverity.ERROR,
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> IntegrityCheckResult:
        result = IntegrityCheckResult(
            check_name=check_name,
            check_type=check_type,
            passed=passed,
            severity=severity,
            message=message,
            details=details or {},
        )
        self._checks.append(result)
        if not passed and severity.value in ("error", "critical"):
            logger.error(f"INTEGRITY FAIL [{severity.value}]: {check_name} - {message}")
        elif not passed:
            logger.warning(f"INTEGRITY WARN: {check_name} - {message}")
        return result

    # ── Input/output match ──────────────────────────────────────────────

    def check_input_output_match(
        self,
        stage_name: str,
        input_count: int,
        output_count: int,
        expected_ratio: float = 1.0,
        tolerance: float = 0.05,
        severity: IntegrityCheckSeverity = IntegrityCheckSeverity.ERROR,
    ) -> IntegrityCheckResult:
        """Verify that the ratio of output to input is within tolerance of expected.

        Args:
            stage_name: Name of the pipeline stage.
            input_count: Number of input records.
            output_count: Number of output records.
            expected_ratio: Expected output/input ratio.
            tolerance: Allowed deviation from expected ratio.
            severity: Severity on failure.
        """
        if input_count == 0:
            return self._record(
                f"io_match:{stage_name}",
                IntegrityCheckType.INPUT_OUTPUT_MATCH,
                True,
                severity,
                "No input records, skipping check",
            )

        actual_ratio = output_count / input_count
        passed = abs(actual_ratio - expected_ratio) <= tolerance

        return self._record(
            f"io_match:{stage_name}",
            IntegrityCheckType.INPUT_OUTPUT_MATCH,
            passed,
            severity,
            f"Ratio {actual_ratio:.4f} vs expected {expected_ratio:.4f}" if not passed else "",
            {"input_count": input_count, "output_count": output_count, "ratio": actual_ratio, "expected": expected_ratio},
        )

    # ── Stage ordering ──────────────────────────────────────────────────

    def check_stage_ordering(
        self,
        stages: Sequence[str],
        expected_order: Sequence[str],
        severity: IntegrityCheckSeverity = IntegrityCheckSeverity.ERROR,
    ) -> IntegrityCheckResult:
        """Verify that pipeline stages are in the expected order.

        Args:
            stages: Names of stages in their actual order.
            expected_order: Expected order of stage names.
            severity: Severity on failure.
        """
        passed = list(stages) == list(expected_order)
        return self._record(
            "stage_ordering",
            IntegrityCheckType.STAGE_ORDERING,
            passed,
            severity,
            f"Expected {list(expected_order)}, got {list(stages)}" if not passed else "",
            {"actual": list(stages), "expected": list(expected_order)},
        )

    def check_stage_present(
        self,
        stages: Sequence[str],
        required_stages: set[str],
        severity: IntegrityCheckSeverity = IntegrityCheckSeverity.CRITICAL,
    ) -> IntegrityCheckResult:
        """Verify that all required stages are present in the pipeline.

        Args:
            stages: Actual stage names.
            required_stages: Stages that must be present.
            severity: Severity on failure.
        """
        stages_set = set(stages)
        missing = required_stages - stages_set
        passed = len(missing) == 0
        return self._record(
            "stage_present",
            IntegrityCheckType.STAGE_ORDERING,
            passed,
            severity,
            f"Missing stages: {sorted(missing)}" if not passed else "",
            {"present": sorted(stages_set), "required": sorted(required_stages), "missing": sorted(missing)},
        )

    # ── Data contract ───────────────────────────────────────────────────

    def check_data_contract(
        self,
        stage_name: str,
        input_schema: dict[str, str],
        output_schema: dict[str, str],
        actual_output_columns: set[str],
        severity: IntegrityCheckSeverity = IntegrityCheckSeverity.ERROR,
    ) -> IntegrityCheckResult:
        """Verify that stage output matches the expected schema.

        Args:
            stage_name: Name of the pipeline stage.
            input_schema: Expected input column name -> type mapping.
            output_schema: Expected output column name -> type mapping.
            actual_output_columns: Actual columns in the output.
            severity: Severity on failure.
        """
        expected_cols = set(output_schema.keys())
        missing = expected_cols - actual_output_columns
        extra = actual_output_columns - expected_cols

        passed = len(missing) == 0
        msg_parts = []
        if missing:
            msg_parts.append(f"Missing columns: {sorted(missing)}")
        if extra:
            msg_parts.append(f"Extra columns: {sorted(extra)}")

        return self._record(
            f"data_contract:{stage_name}",
            IntegrityCheckType.DATA_CONTRACT,
            passed,
            severity,
            "; ".join(msg_parts),
            {
                "missing": sorted(missing),
                "extra": sorted(extra),
                "expected": sorted(expected_cols),
                "actual": sorted(actual_output_columns),
            },
        )

    # ── Idempotency ─────────────────────────────────────────────────────

    def check_idempotency(
        self,
        stage_name: str,
        stage_fn: Callable[[Any], Any],
        input_data: Any,
        num_runs: int = 2,
        severity: IntegrityCheckSeverity = IntegrityCheckSeverity.WARNING,
    ) -> IntegrityCheckResult:
        """Verify that a pipeline stage is idempotent (same input → same output).

        Args:
            stage_name: Name of the pipeline stage.
            stage_fn: The stage function to test.
            input_data: Input data.
            num_runs: Number of times to run.
            severity: Severity on failure.
        """
        results = []
        for _ in range(num_runs):
            results.append(stage_fn(input_data))

        # Compare all results to the first
        import pandas as pd

        for i in range(1, len(results)):
            r1, ri = results[0], results[i]
            if isinstance(r1, pd.DataFrame) and isinstance(ri, pd.DataFrame):
                if not r1.equals(ri):
                    return self._record(
                        f"idempotency:{stage_name}",
                        IntegrityCheckType.IDEMPOTENCY,
                        False,
                        severity,
                        f"Run {i + 1} differs from run 1",
                    )
            elif r1 != ri:
                return self._record(
                    f"idempotency:{stage_name}",
                    IntegrityCheckType.IDEMPOTENCY,
                    False,
                    severity,
                    f"Run {i + 1} differs from run 1",
                )

        return self._record(
            f"idempotency:{stage_name}",
            IntegrityCheckType.IDEMPOTENCY,
            True,
            severity,
            details={"num_runs": num_runs},
        )

    # ── Determinism ─────────────────────────────────────────────────────

    def check_determinism(
        self,
        stage_name: str,
        stage_fn: Callable[[Any], Any],
        input_data: Any,
        num_runs: int = 3,
        tolerance: float = 1e-6,
        severity: IntegrityCheckSeverity = IntegrityCheckSeverity.WARNING,
    ) -> IntegrityCheckResult:
        """Verify that a pipeline stage is deterministic.

        For numeric outputs, checks values within tolerance.
        For categorical outputs, checks exact equality.

        Args:
            stage_name: Name of the pipeline stage.
            stage_fn: The stage function.
            input_data: Input data.
            num_runs: Number of runs.
            tolerance: Numeric tolerance.
            severity: Severity on failure.
        """
        import numpy as np
        import pandas as pd

        results = [stage_fn(input_data) for _ in range(num_runs)]
        reference = results[0]

        for i in range(1, len(results)):
            current = results[i]

            if isinstance(reference, pd.DataFrame) and isinstance(current, pd.DataFrame):
                # Compare numeric columns with tolerance
                num_cols = reference.select_dtypes(include=[np.number]).columns
                for col in num_cols:
                    diff = np.max(np.abs(reference[col].values - current[col].values))
                    if diff > tolerance:
                        return self._record(
                            f"determinism:{stage_name}",
                            IntegrityCheckType.DETERMINISM,
                            False,
                            severity,
                            f"Run {i + 1}: column '{col}' max diff {diff:.6e} > {tolerance}",
                        )
                # Compare non-numeric columns exactly
                non_num = reference.select_dtypes(exclude=[np.number]).columns
                for col in non_num:
                    if not reference[col].equals(current[col]):
                        return self._record(
                            f"determinism:{stage_name}",
                            IntegrityCheckType.DETERMINISM,
                            False,
                            severity,
                            f"Run {i + 1}: column '{col}' differs",
                        )

            elif isinstance(reference, np.ndarray) and isinstance(current, np.ndarray):
                diff = np.max(np.abs(reference - current))
                if diff > tolerance:
                    return self._record(
                        f"determinism:{stage_name}",
                        IntegrityCheckType.DETERMINISM,
                        False,
                        severity,
                        f"Run {i + 1}: max diff {diff:.6e} > {tolerance}",
                    )

            elif reference != current:
                return self._record(
                    f"determinism:{stage_name}",
                    IntegrityCheckType.DETERMINISM,
                    False,
                    severity,
                    f"Run {i + 1} differs from run 1",
                )

        return self._record(
            f"determinism:{stage_name}",
            IntegrityCheckType.DETERMINISM,
            True,
            severity,
            details={"num_runs": num_runs, "tolerance": tolerance},
        )

    # ── No data loss ────────────────────────────────────────────────────

    def check_no_data_loss(
        self,
        stage_name: str,
        input_keys: set[str],
        output_keys: set[str],
        severity: IntegrityCheckSeverity = IntegrityCheckSeverity.CRITICAL,
    ) -> IntegrityCheckResult:
        """Verify no data loss: all input keys should appear in output.

        Args:
            stage_name: Name of the pipeline stage.
            input_keys: Set of keys from input.
            output_keys: Set of keys from output.
            severity: Severity on failure.
        """
        lost = input_keys - output_keys
        passed = len(lost) == 0
        return self._record(
            f"no_data_loss:{stage_name}",
            IntegrityCheckType.NO_DATA_LOSS,
            passed,
            severity,
            f"Lost {len(lost)} keys: {sorted(lost)[:10]}" if not passed else "",
            {
                "input_count": len(input_keys),
                "output_count": len(output_keys),
                "lost_count": len(lost),
                "lost_keys": sorted(lost)[:50],
            },
        )

    def check_row_preservation(
        self,
        stage_name: str,
        input_count: int,
        output_count: int,
        max_loss_ratio: float = 0.0,
        severity: IntegrityCheckSeverity = IntegrityCheckSeverity.ERROR,
    ) -> IntegrityCheckResult:
        """Verify that no more than max_loss_ratio fraction of rows are lost.

        Args:
            stage_name: Pipeline stage name.
            input_count: Input row count.
            output_count: Output row count.
            max_loss_ratio: Maximum allowed row loss (0 = no loss).
            severity: Severity on failure.
        """
        if input_count == 0:
            return self._record(
                f"row_preservation:{stage_name}",
                IntegrityCheckType.NO_DATA_LOSS,
                True,
                severity,
                details={"input": 0, "output": 0},
            )

        loss_ratio = (input_count - output_count) / input_count
        passed = loss_ratio <= max_loss_ratio
        return self._record(
            f"row_preservation:{stage_name}",
            IntegrityCheckType.NO_DATA_LOSS,
            passed,
            severity,
            f"Lost {loss_ratio:.2%} rows (max {max_loss_ratio:.2%})" if not passed else "",
            {"input_count": input_count, "output_count": output_count, "loss_ratio": loss_ratio, "max_allowed": max_loss_ratio},
        )

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self) -> IntegrityReport:
        """Generate a comprehensive integrity report."""
        report = IntegrityReport(pipeline_name=self.pipeline_name)
        report.checks = list(self._checks)
        report.generated_at = datetime.now(timezone.utc)

        total = len(self._checks)
        passed = sum(1 for c in self._checks if c.passed)
        logger.info(f"Integrity report for '{self.pipeline_name}': {passed}/{total} passed")
        return report

    def clear(self) -> None:
        """Clear all recorded checks."""
        self._checks.clear()


__all__ = [
    "IntegrityCheckSeverity",
    "IntegrityCheckType",
    "IntegrityCheckResult",
    "IntegrityReport",
    "IntegrityChecker",
]