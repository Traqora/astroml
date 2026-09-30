"""Expectation suite construction, with optional Great Expectations backing.

Resolves part of #644.

Great Expectations is an optional dependency.  When it is installed, suites
built here are convertible to a native ``ExpectationSuite`` via
:meth:`ExpectationSuite.to_great_expectations`; when it is not, the suite still
builds, serialises and validates through the in-repo engine in
:mod:`astroml.validation.great_expectations.validator`.  That keeps data
validation available in the lightweight CI image while remaining a first-class
GE citizen wherever GE is present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "Expectation",
    "ExpectationSuite",
    "ExpectationType",
    "SuiteBuilder",
    "great_expectations_available",
]


def great_expectations_available() -> bool:
    """Return whether the ``great_expectations`` package is importable."""
    try:  # pragma: no cover - depends on the environment
        import great_expectations  # noqa: F401
    except ImportError:
        return False
    return True


class ExpectationType(str, Enum):
    """The expectation types this integration supports.

    Names match the Great Expectations vocabulary exactly, so suites round-trip
    to and from GE without a translation table.
    """

    COLUMN_TO_EXIST = "expect_column_to_exist"
    COLUMN_VALUES_TO_NOT_BE_NULL = "expect_column_values_to_not_be_null"
    COLUMN_VALUES_TO_BE_BETWEEN = "expect_column_values_to_be_between"
    COLUMN_VALUES_TO_BE_IN_SET = "expect_column_values_to_be_in_set"
    COLUMN_VALUES_TO_BE_UNIQUE = "expect_column_values_to_be_unique"
    COLUMN_VALUES_TO_BE_OF_TYPE = "expect_column_values_to_be_of_type"
    COLUMN_MEAN_TO_BE_BETWEEN = "expect_column_mean_to_be_between"
    COLUMN_VALUES_TO_MATCH_REGEX = "expect_column_values_to_match_regex"
    TABLE_ROW_COUNT_TO_BE_BETWEEN = "expect_table_row_count_to_be_between"
    TABLE_COLUMNS_TO_MATCH_SET = "expect_table_columns_to_match_set"


@dataclass(frozen=True)
class Expectation:
    """A single expectation, mirroring the GE ``ExpectationConfiguration`` shape."""

    expectation_type: ExpectationType
    kwargs: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def column(self) -> str | None:
        """Return the column this expectation applies to, if any."""
        column = self.kwargs.get("column")
        return str(column) if column is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Return the GE-compatible dictionary representation."""
        return {
            "expectation_type": self.expectation_type.value,
            "kwargs": dict(self.kwargs),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Expectation:
        """Rebuild an expectation from :meth:`to_dict` output."""
        return cls(
            expectation_type=ExpectationType(payload["expectation_type"]),
            kwargs=dict(payload.get("kwargs", {})),
            meta=dict(payload.get("meta", {})),
        )


@dataclass
class ExpectationSuite:
    """A named collection of expectations."""

    name: str
    expectations: list[Expectation] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    data_asset_type: str = "Dataset"

    def __post_init__(self) -> None:
        """Validate the suite name."""
        if not self.name:
            raise ValueError("suite name must not be empty")

    def add(self, expectation: Expectation) -> ExpectationSuite:
        """Append an expectation and return the suite for chaining."""
        self.expectations.append(expectation)
        return self

    def columns(self) -> list[str]:
        """Return every column referenced by the suite, in first-seen order."""
        seen: dict[str, None] = {}
        for expectation in self.expectations:
            column = expectation.column
            if column is not None:
                seen.setdefault(column, None)
        return list(seen)

    def for_column(self, column: str) -> list[Expectation]:
        """Return the expectations that apply to ``column``."""
        return [e for e in self.expectations if e.column == column]

    def to_dict(self) -> dict[str, Any]:
        """Return the GE-compatible dictionary representation."""
        return {
            "expectation_suite_name": self.name,
            "data_asset_type": self.data_asset_type,
            "expectations": [e.to_dict() for e in self.expectations],
            "meta": dict(self.meta),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return the suite as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> Path:
        """Write the suite to ``path`` as JSON and return the path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ExpectationSuite:
        """Read a suite previously written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExpectationSuite:
        """Rebuild a suite from :meth:`to_dict` output."""
        return cls(
            name=payload["expectation_suite_name"],
            expectations=[Expectation.from_dict(e) for e in payload.get("expectations", [])],
            meta=dict(payload.get("meta", {})),
            data_asset_type=payload.get("data_asset_type", "Dataset"),
        )

    def to_great_expectations(self) -> Any:
        """Return a native ``great_expectations.ExpectationSuite``.

        Raises ``ImportError`` when Great Expectations is not installed.
        """
        try:  # pragma: no cover - exercised only where GE is installed
            from great_expectations.core import (
                ExpectationConfiguration,
            )
            from great_expectations.core import (
                ExpectationSuite as GEExpectationSuite,
            )
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "great_expectations is not installed; install the 'validation' extra "
                "to convert suites to native GE objects"
            ) from exc
        return GEExpectationSuite(  # pragma: no cover
            expectation_suite_name=self.name,
            expectations=[
                ExpectationConfiguration(
                    expectation_type=e.expectation_type.value, kwargs=dict(e.kwargs)
                )
                for e in self.expectations
            ],
            meta=dict(self.meta),
        )

    def __len__(self) -> int:
        """Return the number of expectations in the suite."""
        return len(self.expectations)


class SuiteBuilder:
    """Builds expectation suites by hand or by profiling a dataset.

    ``SuiteBuilder`` accepts either a pandas ``DataFrame`` or a plain
    ``dict[str, list]`` of columns, so profiling works without pandas installed.

    Example::

        suite = (
            SuiteBuilder("transactions")
            .expect_columns(["account_id", "amount"])
            .expect_not_null("account_id")
            .expect_between("amount", 0.0, 1e9)
            .build()
        )
    """

    def __init__(self, name: str, *, meta: dict[str, Any] | None = None) -> None:
        self._suite = ExpectationSuite(name=name, meta=dict(meta or {}))

    # ── Manual construction ──────────────────────────────────────────────────

    def expect_columns(self, columns: Sequence[str], *, exact: bool = False) -> SuiteBuilder:
        """Require the listed columns to exist (and only those, when ``exact``)."""
        if exact:
            self._suite.add(
                Expectation(
                    ExpectationType.TABLE_COLUMNS_TO_MATCH_SET,
                    {"column_set": list(columns)},
                )
            )
        for column in columns:
            self._suite.add(Expectation(ExpectationType.COLUMN_TO_EXIST, {"column": column}))
        return self

    def expect_not_null(self, column: str, *, mostly: float = 1.0) -> SuiteBuilder:
        """Require ``column`` to be non-null for at least ``mostly`` of rows."""
        _check_mostly(mostly)
        self._suite.add(
            Expectation(
                ExpectationType.COLUMN_VALUES_TO_NOT_BE_NULL,
                {"column": column, "mostly": mostly},
            )
        )
        return self

    def expect_between(
        self,
        column: str,
        min_value: float | None,
        max_value: float | None,
        *,
        mostly: float = 1.0,
    ) -> SuiteBuilder:
        """Require ``column`` values to fall within ``[min_value, max_value]``."""
        _check_mostly(mostly)
        self._suite.add(
            Expectation(
                ExpectationType.COLUMN_VALUES_TO_BE_BETWEEN,
                {
                    "column": column,
                    "min_value": min_value,
                    "max_value": max_value,
                    "mostly": mostly,
                },
            )
        )
        return self

    def expect_in_set(
        self, column: str, values: Sequence[Any], *, mostly: float = 1.0
    ) -> SuiteBuilder:
        """Require ``column`` values to be drawn from ``values``."""
        _check_mostly(mostly)
        self._suite.add(
            Expectation(
                ExpectationType.COLUMN_VALUES_TO_BE_IN_SET,
                {"column": column, "value_set": list(values), "mostly": mostly},
            )
        )
        return self

    def expect_unique(self, column: str) -> SuiteBuilder:
        """Require ``column`` values to be unique."""
        self._suite.add(Expectation(ExpectationType.COLUMN_VALUES_TO_BE_UNIQUE, {"column": column}))
        return self

    def expect_type(self, column: str, type_name: str) -> SuiteBuilder:
        """Require ``column`` to hold values of ``type_name``."""
        self._suite.add(
            Expectation(
                ExpectationType.COLUMN_VALUES_TO_BE_OF_TYPE,
                {"column": column, "type_": type_name},
            )
        )
        return self

    def expect_mean_between(
        self, column: str, min_value: float | None, max_value: float | None
    ) -> SuiteBuilder:
        """Require the mean of ``column`` to fall within a range."""
        self._suite.add(
            Expectation(
                ExpectationType.COLUMN_MEAN_TO_BE_BETWEEN,
                {"column": column, "min_value": min_value, "max_value": max_value},
            )
        )
        return self

    def expect_matches_regex(
        self, column: str, pattern: str, *, mostly: float = 1.0
    ) -> SuiteBuilder:
        """Require ``column`` values to match ``pattern``."""
        _check_mostly(mostly)
        self._suite.add(
            Expectation(
                ExpectationType.COLUMN_VALUES_TO_MATCH_REGEX,
                {"column": column, "regex": pattern, "mostly": mostly},
            )
        )
        return self

    def expect_row_count_between(
        self, min_value: int | None, max_value: int | None
    ) -> SuiteBuilder:
        """Require the table row count to fall within a range."""
        self._suite.add(
            Expectation(
                ExpectationType.TABLE_ROW_COUNT_TO_BE_BETWEEN,
                {"min_value": min_value, "max_value": max_value},
            )
        )
        return self

    def build(self) -> ExpectationSuite:
        """Return the assembled suite."""
        return self._suite

    # ── Automated profiling ──────────────────────────────────────────────────

    @classmethod
    def from_dataset(
        cls,
        name: str,
        data: Any,
        *,
        tolerance: float = 0.1,
        null_tolerance: float = 0.0,
        max_categories: int = 20,
        infer_uniqueness: bool = True,
    ) -> ExpectationSuite:
        """Profile ``data`` and generate an expectation suite automatically.

        Numeric columns get range and mean expectations widened by
        ``tolerance``; low-cardinality columns get value-set expectations;
        columns with no repeats get a uniqueness expectation.  The generated
        suite is a *starting point* — review it before enforcing it in CI.
        """
        if not 0.0 <= tolerance < 10.0:
            raise ValueError("tolerance must be within [0, 10)")
        if not 0.0 <= null_tolerance <= 1.0:
            raise ValueError("null_tolerance must be within [0, 1]")

        columns = _as_column_mapping(data)
        builder = cls(
            name,
            meta={
                "generated_by": "astroml.validation.great_expectations.SuiteBuilder",
                "profiling_tolerance": tolerance,
            },
        )
        row_count = max((len(values) for values in columns.values()), default=0)
        builder.expect_columns(list(columns), exact=True)
        builder.expect_row_count_between(
            max(int(row_count * (1 - tolerance)), 0), int(row_count * (1 + tolerance)) or None
        )

        for column, values in columns.items():
            non_null = [v for v in values if not _is_null(v)]
            null_rate = 1.0 - (len(non_null) / len(values)) if values else 0.0
            if null_rate <= null_tolerance:
                builder.expect_not_null(column, mostly=max(1.0 - null_tolerance, 0.0) or 1.0)

            if not non_null:
                continue

            if all(isinstance(v, bool) for v in non_null):
                builder.expect_in_set(column, [True, False])
                continue

            if all(isinstance(v, (int, float)) for v in non_null):
                low, high = min(non_null), max(non_null)
                span = (high - low) or (abs(high) or 1.0)
                builder.expect_between(column, low - span * tolerance, high + span * tolerance)
                mean = sum(non_null) / len(non_null)
                builder.expect_mean_between(
                    column, mean - span * tolerance, mean + span * tolerance
                )
            else:
                distinct = {str(v) for v in non_null}
                if len(distinct) <= max_categories:
                    builder.expect_in_set(column, sorted(distinct))
                builder.expect_type(column, "str")

            if infer_uniqueness and len(set(map(str, non_null))) == len(non_null) > 1:
                builder.expect_unique(column)

        return builder.build()


def _check_mostly(mostly: float) -> None:
    """Validate a GE ``mostly`` parameter."""
    if not 0.0 < mostly <= 1.0:
        raise ValueError("mostly must be within (0, 1]")


def _is_null(value: Any) -> bool:
    """Return whether ``value`` counts as null for validation purposes."""
    if value is None:
        return True
    return isinstance(value, float) and value != value  # NaN


def _as_column_mapping(data: Any) -> dict[str, list[Any]]:
    """Normalise a DataFrame or column mapping into ``{column: values}``."""
    if hasattr(data, "to_dict") and hasattr(data, "columns"):
        return {str(column): list(data[column]) for column in data.columns}
    if isinstance(data, dict):
        return {str(column): list(values) for column, values in data.items()}
    raise TypeError("data must be a pandas DataFrame or a mapping of column name to values")
