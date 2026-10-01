from astroml.utils.exceptions import AstroMLError
"""Expectation suite execution and validation-result storage.

Resolves part of #644.

:class:`DataValidator` evaluates an
:class:`~astroml.validation.great_expectations.suite_builder.ExpectationSuite`
against a dataset and returns a :class:`ValidationResult` whose shape mirrors
the Great Expectations validation result, so downstream consumers (data docs,
CI gates, dashboards) work identically with or without GE installed.

:class:`ValidationStore` persists results to disk as JSON, keyed by suite and
run identifier, which gives the validation dashboard and the CI gate a shared
source of truth.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from astroml.validation.great_expectations.suite_builder import (
    Expectation,
    ExpectationSuite,
    ExpectationType,
    _as_column_mapping,
    _is_null,
)

__all__ = [
    "DataValidationError",
    "DataValidator",
    "ExpectationResult",
    "ValidationResult",
    "ValidationStore",
]


class DataValidationError(RuntimeError):
    """Raised by :meth:`DataValidator.validate_or_raise` on failure."""


def _utcnow_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExpectationResult:
    """Outcome of evaluating one expectation."""

    expectation: Expectation
    success: bool
    observed_value: Any = None
    unexpected_count: int = 0
    unexpected_percent: float = 0.0
    element_count: int = 0
    exception_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a GE-shaped dictionary representation."""
        return {
            "success": self.success,
            "expectation_config": self.expectation.to_dict(),
            "result": {
                "observed_value": self.observed_value,
                "element_count": self.element_count,
                "unexpected_count": self.unexpected_count,
                "unexpected_percent": self.unexpected_percent,
            },
            "exception_info": {
                "raised_exception": self.exception_message is not None,
                "exception_message": self.exception_message,
            },
        }


@dataclass(frozen=True)
class ValidationResult:
    """Aggregate outcome of validating a dataset against a suite."""

    suite_name: str
    success: bool
    results: tuple[ExpectationResult, ...]
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    validated_at: str = field(default_factory=_utcnow_iso)
    dataset_name: str = "dataset"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def evaluated_expectations(self) -> int:
        """Number of expectations evaluated."""
        return len(self.results)

    @property
    def successful_expectations(self) -> int:
        """Number of expectations that passed."""
        return sum(1 for result in self.results if result.success)

    @property
    def failed_expectations(self) -> tuple[ExpectationResult, ...]:
        """The expectations that failed."""
        return tuple(result for result in self.results if not result.success)

    @property
    def success_percent(self) -> float:
        """Share of expectations that passed, in ``[0, 100]``."""
        if not self.results:
            return 100.0
        return 100.0 * self.successful_expectations / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        """Return a GE-shaped dictionary representation."""
        return {
            "success": self.success,
            "run_id": self.run_id,
            "validated_at": self.validated_at,
            "dataset_name": self.dataset_name,
            "meta": dict(self.meta),
            "statistics": {
                "evaluated_expectations": self.evaluated_expectations,
                "successful_expectations": self.successful_expectations,
                "unsuccessful_expectations": len(self.failed_expectations),
                "success_percent": self.success_percent,
            },
            "expectation_suite_name": self.suite_name,
            "results": [result.to_dict() for result in self.results],
        }

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        state = "PASSED" if self.success else "FAILED"
        return (
            f"{self.suite_name}: {state} — {self.successful_expectations}/"
            f"{self.evaluated_expectations} expectations met "
            f"({self.success_percent:.1f}%)"
        )


class DataValidator:
    """Evaluates expectation suites against tabular data.

    Accepts a pandas ``DataFrame`` or a ``dict[str, list]`` column mapping, so
    validation runs in environments without pandas or Great Expectations.
    """

    def __init__(self, suite: ExpectationSuite) -> None:
        self.suite = suite

    def validate(
        self,
        data: Any,
        *,
        dataset_name: str = "dataset",
        meta: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Evaluate every expectation in the suite against ``data``."""
        columns = _as_column_mapping(data)
        row_count = max((len(values) for values in columns.values()), default=0)

        results: list[ExpectationResult] = []
        for expectation in self.suite.expectations:
            handler = _HANDLERS.get(expectation.expectation_type)
            if handler is None:  # pragma: no cover - guarded by the enum
                results.append(
                    ExpectationResult(
                        expectation=expectation,
                        success=False,
                        exception_message=(f"no handler for {expectation.expectation_type.value}"),
                    )
                )
                continue
            try:
                results.append(handler(expectation, columns, row_count))
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                results.append(
                    ExpectationResult(
                        expectation=expectation, success=False, exception_message=str(exc)
                    )
                )

        return ValidationResult(
            suite_name=self.suite.name,
            success=all(result.success for result in results),
            results=tuple(results),
            dataset_name=dataset_name,
            meta=dict(meta or {}),
        )

    def validate_or_raise(self, data: Any, **kwargs: Any) -> ValidationResult:
        """Validate ``data`` and raise :class:`DataValidationError` on failure.

        This is the form CI pipelines want: a non-zero exit on bad data, with
        the failing expectations named in the message.
        """
        result = self.validate(data, **kwargs)
        if not result.success:
            failures = "; ".join(
                f"{r.expectation.expectation_type.value}"
                f"({r.expectation.kwargs.get('column', 'table')})"
                for r in result.failed_expectations
            )
            raise DataValidationError(f"{result.summary()} — failed: {failures}")
        return result


class ValidationStore:
    """Persists validation results as JSON under a root directory.

    Layout::

        <root>/<suite-name>/<run-id>.json
        <root>/<suite-name>/history.json
    """

    def __init__(self, root: str | Path = "validation_results") -> None:
        self.root = Path(root)

    def save(self, result: ValidationResult) -> Path:
        """Write ``result`` to disk and append it to the suite history."""
        directory = self.root / _slugify(result.suite_name)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{result.run_id}.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")

        history_file = directory / "history.json"
        history: list[dict[str, Any]] = []
        if history_file.is_file():
            history = json.loads(history_file.read_text(encoding="utf-8"))
        history.append(
            {
                "run_id": result.run_id,
                "validated_at": result.validated_at,
                "dataset_name": result.dataset_name,
                "success": result.success,
                "success_percent": result.success_percent,
            }
        )
        history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return path

    def history(self, suite_name: str) -> list[dict[str, Any]]:
        """Return the run history of a suite, oldest first."""
        history_file = self.root / _slugify(suite_name) / "history.json"
        if not history_file.is_file():
            return []
        return json.loads(history_file.read_text(encoding="utf-8"))

    def load(self, suite_name: str, run_id: str) -> dict[str, Any]:
        """Return a stored result document."""
        path = self.root / _slugify(suite_name) / f"{run_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"no stored result for {suite_name!r} run {run_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def suites(self) -> list[str]:
        """Return the suites that have stored results."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def latest(self, suite_name: str) -> dict[str, Any] | None:
        """Return the most recent history entry for a suite, if any."""
        history = self.history(suite_name)
        return history[-1] if history else None


# ─── Expectation handlers ────────────────────────────────────────────────────

Handler = Callable[[Expectation, dict[str, list[Any]], int], ExpectationResult]


def _column_values(
    expectation: Expectation, columns: dict[str, list[Any]]
) -> tuple[str, list[Any]]:
    """Return the column name and values referenced by ``expectation``."""
    column = expectation.kwargs.get("column")
    if column is None:
        raise KeyError("expectation is missing the 'column' kwarg")
    if column not in columns:
        raise KeyError(f"column {column!r} is not present in the dataset")
    return str(column), columns[str(column)]


def _mostly(expectation: Expectation) -> float:
    """Return the expectation's ``mostly`` parameter, defaulting to 1.0."""
    return float(expectation.kwargs.get("mostly", 1.0))


def _elementwise(
    expectation: Expectation,
    columns: dict[str, list[Any]],
    predicate: Callable[[Any], bool],
    *,
    skip_nulls: bool = True,
) -> ExpectationResult:
    """Evaluate a per-value predicate over a column."""
    _, values = _column_values(expectation, columns)
    candidates = [v for v in values if not (skip_nulls and _is_null(v))]
    unexpected = [v for v in candidates if not predicate(v)]
    element_count = len(candidates)
    unexpected_percent = 100.0 * len(unexpected) / element_count if element_count else 0.0
    success = (element_count - len(unexpected)) >= _mostly(expectation) * element_count
    return ExpectationResult(
        expectation=expectation,
        success=success,
        observed_value=unexpected[:20],
        unexpected_count=len(unexpected),
        unexpected_percent=unexpected_percent,
        element_count=element_count,
    )


def _handle_column_exists(
    expectation: Expectation, columns: dict[str, list[Any]], _rows: int
) -> ExpectationResult:
    """Handle ``expect_column_to_exist``."""
    column = str(expectation.kwargs.get("column"))
    return ExpectationResult(
        expectation=expectation,
        success=column in columns,
        observed_value=sorted(columns),
        element_count=len(columns),
    )


def _handle_not_null(
    expectation: Expectation, columns: dict[str, list[Any]], _rows: int
) -> ExpectationResult:
    """Handle ``expect_column_values_to_not_be_null``."""
    _, values = _column_values(expectation, columns)
    unexpected = [v for v in values if _is_null(v)]
    element_count = len(values)
    success = (element_count - len(unexpected)) >= _mostly(expectation) * element_count
    return ExpectationResult(
        expectation=expectation,
        success=success,
        observed_value=len(unexpected),
        unexpected_count=len(unexpected),
        unexpected_percent=100.0 * len(unexpected) / element_count if element_count else 0.0,
        element_count=element_count,
    )


def _handle_between(
    expectation: Expectation, columns: dict[str, list[Any]], _rows: int
) -> ExpectationResult:
    """Handle ``expect_column_values_to_be_between``."""
    low = expectation.kwargs.get("min_value")
    high = expectation.kwargs.get("max_value")

    def within(value: Any) -> bool:
        if not isinstance(value, (int, float)):
            return False
        return (low is None or value >= low) and (high is None or value <= high)

    return _elementwise(expectation, columns, within)


def _handle_in_set(
    expectation: Expectation, columns: dict[str, list[Any]], _rows: int
) -> ExpectationResult:
    """Handle ``expect_column_values_to_be_in_set``."""
    allowed = set(map(_hashable, expectation.kwargs.get("value_set", [])))
    return _elementwise(expectation, columns, lambda v: _hashable(v) in allowed)


def _handle_unique(
    expectation: Expectation, columns: dict[str, list[Any]], _rows: int
) -> ExpectationResult:
    """Handle ``expect_column_values_to_be_unique``."""
    _, values = _column_values(expectation, columns)
    candidates = [v for v in values if not _is_null(v)]
    seen: set[Any] = set()
    duplicates: list[Any] = []
    for value in candidates:
        key = _hashable(value)
        if key in seen:
            duplicates.append(value)
            seen.add(key)
        else:
            seen.add(key)
    return ExpectationResult(
        expectation=expectation,
        success=not duplicates,
        observed_value=duplicates[:20],
        unexpected_count=len(duplicates),
        unexpected_percent=100.0 * len(duplicates) / len(candidates) if candidates else 0.0,
        element_count=len(candidates),
    )


#: Type names accepted by ``expect_column_values_to_be_of_type``.
_TYPE_ALIASES: dict[str, tuple[type, ...]] = {
    "int": (int,),
    "integer": (int,),
    "float": (float, int),
    "number": (float, int),
    "str": (str,),
    "string": (str,),
    "bool": (bool,),
    "boolean": (bool,),
}


def _handle_of_type(
    expectation: Expectation, columns: dict[str, list[Any]], _rows: int
) -> ExpectationResult:
    """Handle ``expect_column_values_to_be_of_type``."""
    type_name = str(expectation.kwargs.get("type_", "str")).lower()
    expected = _TYPE_ALIASES.get(type_name)
    if expected is None:
        raise ValueError(f"unsupported type {type_name!r}")
    # bool is a subclass of int; only 'bool' should accept booleans.
    if type_name in ("int", "integer", "float", "number"):
        return _elementwise(
            expectation,
            columns,
            lambda v: isinstance(v, expected) and not isinstance(v, bool),
        )
    return _elementwise(expectation, columns, lambda v: isinstance(v, expected))


def _handle_mean_between(
    expectation: Expectation, columns: dict[str, list[Any]], _rows: int
) -> ExpectationResult:
    """Handle ``expect_column_mean_to_be_between``."""
    _, values = _column_values(expectation, columns)
    numeric = [v for v in values if isinstance(v, (int, float)) and not _is_null(v)]
    if not numeric:
        return ExpectationResult(
            expectation=expectation,
            success=False,
            exception_message="column contains no numeric values",
        )
    mean = sum(numeric) / len(numeric)
    low = expectation.kwargs.get("min_value")
    high = expectation.kwargs.get("max_value")
    success = (low is None or mean >= low) and (high is None or mean <= high)
    return ExpectationResult(
        expectation=expectation,
        success=success,
        observed_value=mean,
        element_count=len(numeric),
    )


def _handle_regex(
    expectation: Expectation, columns: dict[str, list[Any]], _rows: int
) -> ExpectationResult:
    """Handle ``expect_column_values_to_match_regex``."""
    pattern = re.compile(str(expectation.kwargs["regex"]))
    return _elementwise(expectation, columns, lambda v: bool(pattern.search(str(v))))


def _handle_row_count(
    expectation: Expectation, _columns: dict[str, list[Any]], rows: int
) -> ExpectationResult:
    """Handle ``expect_table_row_count_to_be_between``."""
    low = expectation.kwargs.get("min_value")
    high = expectation.kwargs.get("max_value")
    success = (low is None or rows >= low) and (high is None or rows <= high)
    return ExpectationResult(
        expectation=expectation, success=success, observed_value=rows, element_count=rows
    )


def _handle_columns_match_set(
    expectation: Expectation, columns: dict[str, list[Any]], _rows: int
) -> ExpectationResult:
    """Handle ``expect_table_columns_to_match_set``."""
    expected = set(map(str, expectation.kwargs.get("column_set", [])))
    actual = set(columns)
    exact = bool(expectation.kwargs.get("exact_match", True))
    success = actual == expected if exact else expected <= actual
    return ExpectationResult(
        expectation=expectation,
        success=success,
        observed_value=sorted(actual),
        unexpected_count=len(actual.symmetric_difference(expected)),
        element_count=len(actual),
    )


_HANDLERS: dict[ExpectationType, Handler] = {
    ExpectationType.COLUMN_TO_EXIST: _handle_column_exists,
    ExpectationType.COLUMN_VALUES_TO_NOT_BE_NULL: _handle_not_null,
    ExpectationType.COLUMN_VALUES_TO_BE_BETWEEN: _handle_between,
    ExpectationType.COLUMN_VALUES_TO_BE_IN_SET: _handle_in_set,
    ExpectationType.COLUMN_VALUES_TO_BE_UNIQUE: _handle_unique,
    ExpectationType.COLUMN_VALUES_TO_BE_OF_TYPE: _handle_of_type,
    ExpectationType.COLUMN_MEAN_TO_BE_BETWEEN: _handle_mean_between,
    ExpectationType.COLUMN_VALUES_TO_MATCH_REGEX: _handle_regex,
    ExpectationType.TABLE_ROW_COUNT_TO_BE_BETWEEN: _handle_row_count,
    ExpectationType.TABLE_COLUMNS_TO_MATCH_SET: _handle_columns_match_set,
}


def _hashable(value: Any) -> Any:
    """Return a hashable stand-in for ``value``."""
    try:
        hash(value)
    except TypeError:
        return str(value)
    return value


def _slugify(name: str) -> str:
    """Return a filesystem-safe slug for ``name``."""
    slug = "".join(char if char.isalnum() else "-" for char in name.lower())
    return "-".join(part for part in slug.split("-") if part) or "suite"
