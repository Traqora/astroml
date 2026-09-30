"""Tests for the pipeline testing framework (Issue #638).

Covers data quality assertions, schema validation, integrity checks,
pipeline fixtures, and the test runner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astroml.pipeline.testing.data_tests import (
    AssertionSeverity,
    DataAssertion,
    DataDiffReport,
    DataTestSuite,
    RegressionTest,
)
from astroml.pipeline.testing.fixtures import PipelineFixture, PipelineTestRunner
from astroml.pipeline.testing.integrity import (
    IntegrityChecker,
    IntegrityCheckSeverity,
)
from astroml.pipeline.testing.schema_validator import (
    ColumnExpectation,
    ColumnType,
    SchemaDefinition,
    SchemaValidator,
    ValidationStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "tx_id": ["tx_001", "tx_002", "tx_003", "tx_004", "tx_005"],
        "src_account": ["A1", "A2", "A3", "A4", "A5"],
        "dst_account": ["B1", "B2", "B3", "B4", "B5"],
        "amount": [100.0, 200.0, 150.0, 300.0, 250.0],
        "timestamp": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        ]),
        "status": ["success", "success", "pending", "success", "failed"],
    })


@pytest.fixture
def sample_df_with_nulls() -> pd.DataFrame:
    df = pd.DataFrame({
        "tx_id": ["tx_001", "tx_002", None, "tx_004"],
        "amount": [100.0, None, 150.0, 300.0],
    })
    return df


# ---------------------------------------------------------------------------
# DataTestResult & DataAssertion
# ---------------------------------------------------------------------------


class TestDataAssertion:
    """Tests for the DataAssertion helper."""

    def test_not_null_passes(self, sample_df: pd.DataFrame) -> None:
        a = DataAssertion()
        result = a.assert_not_null(sample_df, "tx_id")
        assert result.passed

    def test_not_null_fails(self, sample_df_with_nulls: pd.DataFrame) -> None:
        a = DataAssertion()
        result = a.assert_not_null(sample_df_with_nulls, "tx_id")
        assert not result.passed
        assert result.details["null_count"] == 1

    def test_unique_passes(self, sample_df: pd.DataFrame) -> None:
        a = DataAssertion()
        result = a.assert_unique(sample_df, "tx_id")
        assert result.passed

    def test_unique_fails_with_dupes(self) -> None:
        df = pd.DataFrame({"id": ["a", "b", "a"]})
        a = DataAssertion()
        result = a.assert_unique(df, "id")
        assert not result.passed

    def test_column_values_in_set(self, sample_df: pd.DataFrame) -> None:
        a = DataAssertion()
        result = a.assert_column_values_in_set(
            sample_df, "status", {"success", "pending", "failed"},
        )
        assert result.passed

    def test_column_values_in_set_fails(self, sample_df: pd.DataFrame) -> None:
        a = DataAssertion()
        result = a.assert_column_values_in_set(
            sample_df, "status", {"success", "pending"},
        )
        assert not result.passed

    def test_row_count(self, sample_df: pd.DataFrame) -> None:
        a = DataAssertion()
        result = a.assert_row_count(sample_df, min_rows=1, max_rows=10)
        assert result.passed

    def test_row_count_too_few(self, sample_df: pd.DataFrame) -> None:
        a = DataAssertion()
        result = a.assert_row_count(sample_df, min_rows=100)
        assert not result.passed

    def test_column_mean_between(self, sample_df: pd.DataFrame) -> None:
        a = DataAssertion()
        result = a.assert_column_mean_between(sample_df, "amount", 50.0, 500.0)
        assert result.passed

    def test_column_mean_between_outside(self, sample_df: pd.DataFrame) -> None:
        a = DataAssertion()
        result = a.assert_column_mean_between(sample_df, "amount", 0.0, 50.0)
        assert not result.passed

    def test_max_null_fraction(self, sample_df_with_nulls: pd.DataFrame) -> None:
        a = DataAssertion()
        result = a.assert_max_null_fraction(sample_df_with_nulls, "amount", max_fraction=0.5)
        assert result.passed

    def test_max_null_fraction_exceeded(self, sample_df_with_nulls: pd.DataFrame) -> None:
        a = DataAssertion()
        result = a.assert_max_null_fraction(sample_df_with_nulls, "amount", max_fraction=0.01)
        assert not result.passed

    def test_all_passed(self, sample_df: pd.DataFrame) -> None:
        a = DataAssertion()
        a.assert_not_null(sample_df, "tx_id")
        a.assert_unique(sample_df, "tx_id")
        a.assert_row_count(sample_df, min_rows=1)
        assert a.all_passed

    def test_clear_resets_results(self, sample_df: pd.DataFrame) -> None:
        a = DataAssertion()
        a.assert_not_null(sample_df, "tx_id")
        assert len(a.results) == 1
        a.clear()
        assert len(a.results) == 0


# ---------------------------------------------------------------------------
# DataTestSuite
# ---------------------------------------------------------------------------


class TestDataTestSuite:
    """Tests for DataTestSuite."""

    def test_run_suite(self, sample_df: pd.DataFrame) -> None:
        a = DataAssertion()
        suite = DataTestSuite(name="basic_checks")
        suite.add_test(lambda df: a.assert_not_null(df, "tx_id"))
        suite.add_test(lambda df: a.assert_unique(df, "tx_id"))
        suite.add_test(lambda df: a.assert_row_count(df, min_rows=1))

        results = suite.run(sample_df)
        assert len(results) == 3
        assert all(r.passed for r in results)

    def test_suite_to_dict(self) -> None:
        suite = DataTestSuite(name="test")
        d = suite.to_dict()
        assert d["name"] == "test"


# ---------------------------------------------------------------------------
# DataDiffReport
# ---------------------------------------------------------------------------


class TestDataDiffReport:
    """Tests for DataDiffReport."""

    def test_no_change(self, sample_df: pd.DataFrame) -> None:
        report = DataDiffReport.compare(sample_df, sample_df)
        assert not report.has_significant_change
        assert report.columns_added == []
        assert report.columns_removed == []

    def test_column_added(self) -> None:
        ref = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        cur = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        report = DataDiffReport.compare(cur, ref)
        assert "c" in report.columns_added

    def test_numeric_drift(self) -> None:
        ref = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        cur = pd.DataFrame({"val": [100.0, 200.0, 300.0]})  # Big drift
        report = DataDiffReport.compare(cur, ref, numeric_columns=["val"], max_mean_drift=0.1)
        assert report.has_significant_change

    def test_summary(self, sample_df: pd.DataFrame) -> None:
        report = DataDiffReport.compare(sample_df, sample_df)
        summary = report.summary()
        assert "DataDiffReport" in summary


# ---------------------------------------------------------------------------
# RegressionTest
# ---------------------------------------------------------------------------


class TestRegressionTest:
    """Tests for RegressionTest."""

    def test_numpy_baseline_match(self) -> None:
        rt = RegressionTest(name="test", tolerance=1e-6)
        rt.capture_baseline(np.array([1.0, 2.0, 3.0]))
        result = rt.check(np.array([1.0, 2.0, 3.0]))
        assert result.passed

    def test_numpy_baseline_mismatch(self) -> None:
        rt = RegressionTest(name="test", tolerance=1e-6)
        rt.capture_baseline(np.array([1.0, 2.0, 3.0]))
        result = rt.check(np.array([10.0, 20.0, 30.0]))
        assert not result.passed

    def test_no_baseline(self) -> None:
        rt = RegressionTest(name="test")
        result = rt.check(np.array([1.0]))
        assert not result.passed
        assert "No baseline" in result.message


# ---------------------------------------------------------------------------
# SchemaValidator
# ---------------------------------------------------------------------------


class TestSchemaValidator:
    """Tests for SchemaValidator and SchemaDefinition."""

    @pytest.fixture
    def tx_schema(self) -> SchemaDefinition:
        return SchemaDefinition(
            name="transactions",
            columns={
                "tx_id": ColumnType.STRING,
                "src_account": ColumnType.STRING,
                "amount": ColumnType.FLOAT,
                "timestamp": ColumnType.DATETIME,
                "status": ColumnType.STRING,
            },
            required_columns={"tx_id", "src_account", "amount"},
            expectations=[
                ColumnExpectation.not_null("tx_id"),
                ColumnExpectation.range("amount", min_value=0),
            ],
        )

    def test_validate_passes(self, sample_df: pd.DataFrame, tx_schema: SchemaDefinition) -> None:
        validator = SchemaValidator()
        result = validator.validate(sample_df, tx_schema)
        assert result.status == ValidationStatus.PASSED

    def test_missing_required_column(self, tx_schema: SchemaDefinition) -> None:
        df = pd.DataFrame({"col": [1, 2]})
        validator = SchemaValidator()
        result = validator.validate(df, tx_schema)
        assert result.status == ValidationStatus.FAILED

    def test_not_null_expectation_fails(self, tx_schema: SchemaDefinition) -> None:
        df = pd.DataFrame({
            "tx_id": [None, "tx_002"],
            "src_account": ["A1", "A2"],
            "amount": [100.0, 200.0],
        })
        validator = SchemaValidator()
        result = validator.validate(df, tx_schema)
        assert result.failed_count > 0

    def test_range_expectation_fails(self, tx_schema: SchemaDefinition) -> None:
        df = pd.DataFrame({
            "tx_id": ["tx_001", "tx_002"],
            "src_account": ["A1", "A2"],
            "amount": [-100.0, -200.0],
        })
        validator = SchemaValidator()
        result = validator.validate(df, tx_schema)
        assert result.failed_count > 0

    def test_register_and_get_schema(self, tx_schema: SchemaDefinition) -> None:
        validator = SchemaValidator()
        validator.register_schema(tx_schema)
        assert validator.get_schema("transactions") is tx_schema

    def test_validate_by_name(self, sample_df: pd.DataFrame, tx_schema: SchemaDefinition) -> None:
        validator = SchemaValidator()
        validator.register_schema(tx_schema)
        result = validator.validate(sample_df, "transactions")
        assert result.success

    def test_schema_not_found(self) -> None:
        validator = SchemaValidator()
        result = validator.validate(pd.DataFrame(), "nonexistent")
        assert result.status == ValidationStatus.FAILED

    def test_schema_to_dict(self, tx_schema: SchemaDefinition) -> None:
        d = tx_schema.to_dict()
        assert d["name"] == "transactions"
        assert "tx_id" in d["columns"]


# ---------------------------------------------------------------------------
# IntegrityChecker
# ---------------------------------------------------------------------------


class TestIntegrityChecker:
    """Tests for IntegrityChecker."""

    def test_check_input_output_match(self) -> None:
        checker = IntegrityChecker("test_pipeline")
        result = checker.check_input_output_match("ingestion", 100, 100)
        assert result.passed

    def test_check_input_output_mismatch(self) -> None:
        checker = IntegrityChecker("test_pipeline")
        result = checker.check_input_output_match("ingestion", 100, 50)
        assert not result.passed

    def test_check_stage_ordering(self) -> None:
        checker = IntegrityChecker("test_pipeline")
        result = checker.check_stage_ordering(
            ["ingest", "clean", "feature", "train"],
            ["ingest", "clean", "feature", "train"],
        )
        assert result.passed

    def test_check_stage_ordering_wrong(self) -> None:
        checker = IntegrityChecker("test_pipeline")
        result = checker.check_stage_ordering(
            ["ingest", "feature", "clean", "train"],
            ["ingest", "clean", "feature", "train"],
        )
        assert not result.passed

    def test_check_stage_present(self) -> None:
        checker = IntegrityChecker("test_pipeline")
        result = checker.check_stage_present(
            ["ingest", "clean", "feature"],
            {"ingest", "clean"},
        )
        assert result.passed

    def test_check_stage_missing(self) -> None:
        checker = IntegrityChecker("test_pipeline")
        result = checker.check_stage_present(
            ["ingest", "feature"],
            {"ingest", "clean"},
        )
        assert not result.passed
        assert "clean" in str(result.details["missing"])

    def test_check_idempotency(self) -> None:
        checker = IntegrityChecker("test_pipeline")

        def identity(data: pd.DataFrame) -> pd.DataFrame:
            return data.copy()

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = checker.check_idempotency("identity", identity, df)
        assert result.passed

    def test_check_determinism(self) -> None:
        checker = IntegrityChecker("test_pipeline")

        def stable_fn(data: np.ndarray) -> np.ndarray:
            return data * 2.0

        data = np.array([1.0, 2.0, 3.0])
        result = checker.check_determinism("stable", stable_fn, data)
        assert result.passed

    def test_check_no_data_loss(self) -> None:
        checker = IntegrityChecker("test_pipeline")
        result = checker.check_no_data_loss(
            "dedup",
            input_keys={"a", "b", "c"},
            output_keys={"a", "b", "c"},
        )
        assert result.passed

    def test_check_data_loss(self) -> None:
        checker = IntegrityChecker("test_pipeline")
        result = checker.check_no_data_loss(
            "dedup",
            input_keys={"a", "b", "c", "d"},
            output_keys={"a", "b"},
        )
        assert not result.passed
        assert result.details["lost_count"] == 2

    def test_generate_report(self) -> None:
        checker = IntegrityChecker("test_pipeline")
        checker.check_input_output_match("s1", 10, 10)
        checker.check_stage_present(["s1", "s2"], {"s1"})

        report = checker.generate_report()
        assert report.pipeline_name == "test_pipeline"
        assert report.passed
        assert len(report.checks) == 2

    def test_report_data_loss_details(self) -> None:
        checker = IntegrityChecker("test_pipeline")
        result = checker.check_no_data_loss(
            "filter",
            input_keys={"a", "b", "c", "d", "e"},
            output_keys={"a", "b"},
        )
        assert not result.passed
        # Verify lost_keys in details
        assert "c" in result.details["lost_keys"]

    def test_row_preservation(self) -> None:
        checker = IntegrityChecker("test_pipeline")
        result = checker.check_row_preservation("filter", 100, 100, max_loss_ratio=0.0)
        assert result.passed

    def test_row_preservation_loss(self) -> None:
        checker = IntegrityChecker("test_pipeline")
        result = checker.check_row_preservation("filter", 100, 80, max_loss_ratio=0.1)
        assert not result.passed


# ---------------------------------------------------------------------------
# PipelineFixture
# ---------------------------------------------------------------------------


class TestPipelineFixture:
    """Tests for PipelineFixture."""

    def test_create_from_dicts(self) -> None:
        fixture = PipelineFixture.from_dicts(
            name="test_fixture",
            input_dicts=[{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            expected_output_dicts=[{"x": 10}, {"x": 30}],
        )
        assert fixture.name == "test_fixture"
        assert len(fixture.input_data) == 2
        assert fixture.expected_output_columns == {"x"}
        assert fixture.expected_row_count == 2

    def test_expected_row_count_range_validation(self) -> None:
        """Fixture with range should validate output row count."""
        fixture = PipelineFixture(
            name="range_test",
            input_data=pd.DataFrame({"a": [1, 2, 3]}),
            expected_row_count_range=(1, 5),
        )
        assert fixture.expected_row_count_range == (1, 5)

    def test_to_dict(self) -> None:
        fixture = PipelineFixture.from_dicts(
            name="test",
            input_dicts=[{"a": 1}],
            tags=["critical"],
        )
        d = fixture.to_dict()
        assert d["name"] == "test"
        assert "critical" in d["tags"]


# ---------------------------------------------------------------------------
# PipelineTestRunner
# ---------------------------------------------------------------------------


class TestPipelineTestRunner:
    """Tests for PipelineTestRunner."""

    def test_run_single_fixture(self) -> None:
        fixture = PipelineFixture.from_dicts(
            name="square",
            input_dicts=[{"x": 1}, {"x": 2}, {"x": 3}],
            expected_output_dicts=[{"y": 1}, {"y": 4}, {"y": 9}],
        )

        def square_stage(df: pd.DataFrame, config: dict) -> pd.DataFrame:
            return pd.DataFrame({"y": df["x"] ** 2})

        runner = PipelineTestRunner("test")
        runner.add_fixture(fixture)
        runs = runner.run(square_stage)

        assert len(runs) == 1
        assert runs[0].passed

    def test_run_fails_on_missing_columns(self) -> None:
        fixture = PipelineFixture.from_dicts(
            name="wrong_output",
            input_dicts=[{"x": 1}],
            expected_output_dicts=[{"y": 1, "z": 2}],
        )

        def bad_stage(df: pd.DataFrame, config: dict) -> pd.DataFrame:
            return pd.DataFrame({"y": [1]})  # Missing "z"

        runner = PipelineTestRunner("test")
        runner.add_fixture(fixture)
        runs = runner.run(bad_stage)

        assert not runs[0].passed
        assert "Missing" in runs[0].error_message

    def test_generate_report(self) -> None:
        fixture = PipelineFixture.from_dicts(
            name="pass",
            input_dicts=[{"x": 1}],
            expected_output_dicts=[{"y": 1}],
        )

        def identity(df: pd.DataFrame, config: dict) -> pd.DataFrame:
            return pd.DataFrame({"y": [1]})

        runner = PipelineTestRunner("test")
        runner.add_fixture(fixture)
        runner.run(identity)

        report = runner.generate_report()
        assert report.passed
        assert report.pass_count == 1
        assert report.fail_count == 0

    def test_hooks(self) -> None:
        fixture = PipelineFixture.from_dicts(
            name="hook_test",
            input_dicts=[{"x": 1}],
        )

        hook_calls = []

        def after_hook(fixture: Any, run: Any) -> None:
            hook_calls.append("after")

        def on_pass_hook(fixture: Any, run: Any) -> None:
            hook_calls.append("pass")

        def stage(df: pd.DataFrame, config: dict) -> pd.DataFrame:
            return df.copy()

        runner = PipelineTestRunner("test")
        runner.add_fixture(fixture)
        runner.add_hook("after_run", after_hook)
        runner.add_hook("on_pass", on_pass_hook)
        runner.run(stage)

        assert "after" in hook_calls
        assert "pass" in hook_calls

    def test_clear(self) -> None:
        fixture = PipelineFixture.from_dicts(name="test", input_dicts=[{"x": 1}])
        runner = PipelineTestRunner("test")
        runner.add_fixture(fixture)
        assert len(runner.fixtures) == 1
        runner.clear()
        assert len(runner.fixtures) == 0
        assert len(runner.runs) == 0

    def test_exception_in_stage(self) -> None:
        fixture = PipelineFixture.from_dicts(name="error", input_dicts=[{"x": 1}])

        def raise_error(df: pd.DataFrame, config: dict) -> pd.DataFrame:
            raise ValueError("intentional error")

        runner = PipelineTestRunner("test")
        runner.add_fixture(fixture)
        runs = runner.run(raise_error)

        assert not runs[0].passed
        assert "ValueError" in runs[0].error_message


# ---------------------------------------------------------------------------
# ColumnExpectation
# ---------------------------------------------------------------------------


class TestColumnExpectation:
    """Tests for ColumnExpectation factory methods."""

    def test_not_null(self) -> None:
        e = ColumnExpectation.not_null("col")
        assert e.column == "col"
        assert e.expectation_type == "not_null"

    def test_unique(self) -> None:
        e = ColumnExpectation.unique("col")
        assert e.expectation_type == "unique"

    def test_in_set(self) -> None:
        e = ColumnExpectation.in_set("col", {"a", "b"})
        assert e.kwargs["values"] == ["a", "b"] or e.kwargs["values"] == ["b", "a"]

    def test_range(self) -> None:
        e = ColumnExpectation.range("col", min_value=0, max_value=100)
        assert e.kwargs["min"] == 0
        assert e.kwargs["max"] == 100

    def test_regex_match(self) -> None:
        e = ColumnExpectation.regex_match("col", r"^[A-Z]{2}\\d{4}$")
        assert e.kwargs["pattern"] == r"^[A-Z]{2}\\d{4}$"