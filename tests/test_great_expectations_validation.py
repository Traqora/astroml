"""Tests for the Great Expectations data validation integration.

Covers #644.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from astroml.validation.great_expectations.data_docs import DataDocsBuilder
from astroml.validation.great_expectations.suite_builder import (
    Expectation,
    ExpectationSuite,
    ExpectationType,
    SuiteBuilder,
    great_expectations_available,
)
from astroml.validation.great_expectations.validator import (
    DataValidationError,
    DataValidator,
    ValidationStore,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "validation" / "ge_config.yaml"


@pytest.fixture
def data() -> dict[str, list[Any]]:
    """A small, clean tabular dataset."""
    return {
        "transaction_id": ["t1", "t2", "t3", "t4"],
        "amount": [10.0, 25.5, 3.25, 99.0],
        "status": ["confirmed", "pending", "confirmed", "failed"],
        "flagged": [True, False, False, True],
    }


@pytest.fixture
def suite(data: dict[str, list[Any]]) -> ExpectationSuite:
    """A hand-authored suite matching ``data``."""
    return (
        SuiteBuilder("transactions")
        .expect_columns(["transaction_id", "amount", "status"])
        .expect_unique("transaction_id")
        .expect_not_null("amount")
        .expect_between("amount", 0.0, 1000.0)
        .expect_in_set("status", ["confirmed", "pending", "failed"])
        .expect_row_count_between(1, 100)
        .build()
    )


# ─── Suite construction ──────────────────────────────────────────────────────


class TestSuiteBuilder:
    """Manual suite construction and serialisation."""

    def test_expectations_are_added_in_order(self, suite: ExpectationSuite) -> None:
        types = [e.expectation_type for e in suite.expectations]
        assert types[0] is ExpectationType.COLUMN_TO_EXIST
        assert ExpectationType.COLUMN_VALUES_TO_BE_UNIQUE in types
        assert len(suite) == len(suite.expectations)

    def test_exact_column_set_adds_table_expectation(self) -> None:
        built = SuiteBuilder("s").expect_columns(["a", "b"], exact=True).build()
        assert built.expectations[0].expectation_type is (
            ExpectationType.TABLE_COLUMNS_TO_MATCH_SET
        )

    def test_columns_and_for_column_helpers(self, suite: ExpectationSuite) -> None:
        assert suite.columns() == ["transaction_id", "amount", "status"]
        assert len(suite.for_column("amount")) == 3

    def test_empty_suite_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            ExpectationSuite(name="")

    @pytest.mark.parametrize("mostly", [0.0, 1.5])
    def test_invalid_mostly_rejected(self, mostly: float) -> None:
        with pytest.raises(ValueError):
            SuiteBuilder("s").expect_not_null("a", mostly=mostly)

    def test_all_builder_methods_are_chainable(self) -> None:
        built = (
            SuiteBuilder("s")
            .expect_type("a", "int")
            .expect_mean_between("a", 0.0, 10.0)
            .expect_matches_regex("b", "^x")
            .build()
        )
        assert len(built) == 3

    def test_dict_round_trip(self, suite: ExpectationSuite) -> None:
        restored = ExpectationSuite.from_dict(suite.to_dict())
        assert restored.name == suite.name
        assert len(restored) == len(suite)

    def test_save_and_load(self, suite: ExpectationSuite, tmp_path: Path) -> None:
        path = suite.save(tmp_path / "nested" / "transactions.json")
        assert path.is_file()
        assert ExpectationSuite.load(path).to_dict() == suite.to_dict()

    def test_to_json_is_valid_json(self, suite: ExpectationSuite) -> None:
        assert json.loads(suite.to_json())["expectation_suite_name"] == "transactions"

    def test_expectation_round_trip(self) -> None:
        expectation = Expectation(ExpectationType.COLUMN_TO_EXIST, {"column": "a"}, {"note": "x"})
        assert Expectation.from_dict(expectation.to_dict()) == expectation

    def test_ge_conversion_reports_missing_dependency(self, suite: ExpectationSuite) -> None:
        if great_expectations_available():
            pytest.skip("great_expectations is installed in this environment")
        with pytest.raises(ImportError, match="great_expectations is not installed"):
            suite.to_great_expectations()


class TestProfiling:
    """Automated suite generation from a dataset."""

    def test_profiling_infers_expectations(self, data: dict[str, list[Any]]) -> None:
        profiled = SuiteBuilder.from_dataset("transactions", data)
        types = {e.expectation_type for e in profiled.expectations}
        assert ExpectationType.TABLE_COLUMNS_TO_MATCH_SET in types
        assert ExpectationType.COLUMN_VALUES_TO_BE_BETWEEN in types
        assert ExpectationType.COLUMN_VALUES_TO_BE_IN_SET in types
        assert profiled.meta["profiling_tolerance"] == 0.1

    def test_profiled_suite_validates_its_own_source(self, data: dict[str, list[Any]]) -> None:
        profiled = SuiteBuilder.from_dataset("transactions", data)
        assert DataValidator(profiled).validate(data).success

    def test_profiling_detects_unique_columns(self, data: dict[str, list[Any]]) -> None:
        profiled = SuiteBuilder.from_dataset("transactions", data)
        unique = [
            e
            for e in profiled.expectations
            if e.expectation_type is ExpectationType.COLUMN_VALUES_TO_BE_UNIQUE
        ]
        assert any(e.column == "transaction_id" for e in unique)

    def test_profiling_treats_booleans_as_a_value_set(self, data: dict[str, list[Any]]) -> None:
        profiled = SuiteBuilder.from_dataset("transactions", data)
        flagged = [e for e in profiled.for_column("flagged")]
        assert any(
            e.expectation_type is ExpectationType.COLUMN_VALUES_TO_BE_IN_SET for e in flagged
        )

    def test_high_cardinality_strings_get_no_value_set(self) -> None:
        wide = {"code": [f"c{i}" for i in range(50)]}
        profiled = SuiteBuilder.from_dataset("wide", wide, max_categories=5)
        assert not any(
            e.expectation_type is ExpectationType.COLUMN_VALUES_TO_BE_IN_SET
            for e in profiled.expectations
        )

    def test_profiling_validates_arguments(self, data: dict[str, list[Any]]) -> None:
        with pytest.raises(ValueError):
            SuiteBuilder.from_dataset("s", data, tolerance=-1.0)
        with pytest.raises(ValueError):
            SuiteBuilder.from_dataset("s", data, null_tolerance=2.0)

    def test_unsupported_data_type_rejected(self) -> None:
        with pytest.raises(TypeError):
            SuiteBuilder.from_dataset("s", [1, 2, 3])


# ─── Validation ──────────────────────────────────────────────────────────────


class TestDataValidator:
    """Expectation execution."""

    def test_clean_data_passes(self, suite: ExpectationSuite, data: dict[str, list[Any]]) -> None:
        result = DataValidator(suite).validate(data, dataset_name="transactions")
        assert result.success
        assert result.failed_expectations == ()
        assert result.success_percent == pytest.approx(100.0)
        assert "PASSED" in result.summary()

    def test_out_of_range_values_fail(
        self, suite: ExpectationSuite, data: dict[str, list[Any]]
    ) -> None:
        data["amount"] = [10.0, 25.5, 3.25, 100_000.0]
        result = DataValidator(suite).validate(data)
        assert not result.success
        failed = {r.expectation.expectation_type for r in result.failed_expectations}
        assert ExpectationType.COLUMN_VALUES_TO_BE_BETWEEN in failed

    def test_nulls_fail_not_null_expectation(
        self, suite: ExpectationSuite, data: dict[str, list[Any]]
    ) -> None:
        data["amount"] = [10.0, None, 3.25, 99.0]
        result = DataValidator(suite).validate(data)
        failure = next(
            r
            for r in result.failed_expectations
            if r.expectation.expectation_type is ExpectationType.COLUMN_VALUES_TO_NOT_BE_NULL
        )
        assert failure.unexpected_count == 1

    def test_mostly_tolerates_some_failures(self) -> None:
        built = SuiteBuilder("s").expect_not_null("a", mostly=0.7).build()
        assert DataValidator(built).validate({"a": [1, 2, 3, None]}).success

    def test_duplicate_values_fail_uniqueness(
        self, suite: ExpectationSuite, data: dict[str, list[Any]]
    ) -> None:
        data["transaction_id"] = ["t1", "t1", "t3", "t4"]
        result = DataValidator(suite).validate(data)
        failure = next(
            r
            for r in result.failed_expectations
            if r.expectation.expectation_type is ExpectationType.COLUMN_VALUES_TO_BE_UNIQUE
        )
        assert failure.unexpected_count == 1

    def test_value_set_violations_fail(
        self, suite: ExpectationSuite, data: dict[str, list[Any]]
    ) -> None:
        data["status"] = ["confirmed", "unknown", "confirmed", "failed"]
        assert not DataValidator(suite).validate(data).success

    def test_missing_column_is_reported_not_raised(
        self, suite: ExpectationSuite, data: dict[str, list[Any]]
    ) -> None:
        del data["status"]
        result = DataValidator(suite).validate(data)
        assert not result.success
        assert any(r.exception_message for r in result.failed_expectations)

    def test_row_count_expectation(self, data: dict[str, list[Any]]) -> None:
        built = SuiteBuilder("s").expect_row_count_between(10, None).build()
        result = DataValidator(built).validate(data)
        assert not result.success
        assert result.results[0].observed_value == 4

    def test_mean_expectation(self, data: dict[str, list[Any]]) -> None:
        built = SuiteBuilder("s").expect_mean_between("amount", 0.0, 10.0).build()
        result = DataValidator(built).validate(data)
        assert not result.success
        assert result.results[0].observed_value == pytest.approx(34.4375)

    def test_mean_expectation_on_non_numeric_column(self, data: dict[str, list[Any]]) -> None:
        built = SuiteBuilder("s").expect_mean_between("status", 0.0, 1.0).build()
        result = DataValidator(built).validate(data)
        assert result.results[0].exception_message == "column contains no numeric values"

    def test_regex_expectation(self) -> None:
        built = SuiteBuilder("s").expect_matches_regex("id", r"^G[A-Z]+$").build()
        assert DataValidator(built).validate({"id": ["GABC", "GXYZ"]}).success
        assert not DataValidator(built).validate({"id": ["GABC", "bad"]}).success

    def test_type_expectation_separates_bool_from_int(self) -> None:
        int_suite = SuiteBuilder("s").expect_type("a", "int").build()
        assert DataValidator(int_suite).validate({"a": [1, 2]}).success
        assert not DataValidator(int_suite).validate({"a": [True, False]}).success

        bool_suite = SuiteBuilder("s").expect_type("a", "bool").build()
        assert DataValidator(bool_suite).validate({"a": [True, False]}).success

    def test_unsupported_type_is_reported(self) -> None:
        built = SuiteBuilder("s").expect_type("a", "complex").build()
        result = DataValidator(built).validate({"a": [1]})
        assert "unsupported type" in (result.results[0].exception_message or "")

    def test_column_set_mismatch_fails(self, data: dict[str, list[Any]]) -> None:
        built = SuiteBuilder("s").expect_columns(["transaction_id"], exact=True).build()
        assert not DataValidator(built).validate(data).success

    def test_validate_or_raise_passes_clean_data(
        self, suite: ExpectationSuite, data: dict[str, list[Any]]
    ) -> None:
        assert DataValidator(suite).validate_or_raise(data).success

    def test_validate_or_raise_raises_on_failure(
        self, suite: ExpectationSuite, data: dict[str, list[Any]]
    ) -> None:
        data["amount"] = [-1.0, -2.0, -3.0, -4.0]
        with pytest.raises(DataValidationError, match="FAILED"):
            DataValidator(suite).validate_or_raise(data)

    def test_result_is_ge_shaped(self, suite: ExpectationSuite, data: dict[str, list[Any]]) -> None:
        payload = DataValidator(suite).validate(data).to_dict()
        assert payload["expectation_suite_name"] == "transactions"
        assert payload["statistics"]["success_percent"] == pytest.approx(100.0)
        assert "expectation_config" in payload["results"][0]

    def test_empty_suite_succeeds_vacuously(self, data: dict[str, list[Any]]) -> None:
        result = DataValidator(ExpectationSuite("empty")).validate(data)
        assert result.success
        assert result.success_percent == 100.0


# ─── Result storage ──────────────────────────────────────────────────────────


class TestValidationStore:
    """Persistence of validation results."""

    def test_save_writes_result_and_history(
        self, suite: ExpectationSuite, data: dict[str, list[Any]], tmp_path: Path
    ) -> None:
        store = ValidationStore(tmp_path)
        result = DataValidator(suite).validate(data)
        path = store.save(result)

        assert path.is_file()
        assert store.suites() == ["transactions"]
        assert len(store.history("transactions")) == 1
        assert store.latest("transactions")["success"] is True

    def test_history_accumulates_runs(
        self, suite: ExpectationSuite, data: dict[str, list[Any]], tmp_path: Path
    ) -> None:
        store = ValidationStore(tmp_path)
        validator = DataValidator(suite)
        store.save(validator.validate(data))
        store.save(validator.validate(data))
        assert len(store.history("transactions")) == 2

    def test_load_returns_the_stored_document(
        self, suite: ExpectationSuite, data: dict[str, list[Any]], tmp_path: Path
    ) -> None:
        store = ValidationStore(tmp_path)
        result = DataValidator(suite).validate(data)
        store.save(result)
        assert store.load("transactions", result.run_id)["run_id"] == result.run_id

    def test_missing_result_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ValidationStore(tmp_path).load("nope", "abc")

    def test_empty_store_reports_nothing(self, tmp_path: Path) -> None:
        store = ValidationStore(tmp_path / "missing")
        assert store.suites() == []
        assert store.history("nope") == []
        assert store.latest("nope") is None


# ─── Data docs ───────────────────────────────────────────────────────────────


class TestDataDocs:
    """Static documentation site generation."""

    def test_suite_page_lists_expectations(self, suite: ExpectationSuite) -> None:
        page = DataDocsBuilder().suite_page(suite)
        assert "expect_column_values_to_be_unique" in page.html
        assert "<!doctype html>" in page.html
        assert page.markdown.startswith("# Expectation suite — transactions")

    def test_result_page_marks_failures(
        self, suite: ExpectationSuite, data: dict[str, list[Any]]
    ) -> None:
        data["amount"] = [-1.0, -2.0, -3.0, -4.0]
        result = DataValidator(suite).validate(data)
        page = DataDocsBuilder().result_page(result)
        assert "class='fail'" in page.html
        assert "FAIL" in page.markdown

    def test_html_escapes_untrusted_content(self) -> None:
        built = SuiteBuilder("<script>alert(1)</script>").build()
        page = DataDocsBuilder().suite_page(built)
        assert "<script>alert(1)</script>" not in page.html
        assert "&lt;script&gt;" in page.html

    def test_index_page_summarises_the_store(
        self, suite: ExpectationSuite, data: dict[str, list[Any]], tmp_path: Path
    ) -> None:
        store = ValidationStore(tmp_path)
        store.save(DataValidator(suite).validate(data))
        page = DataDocsBuilder().index_page(store)
        assert "transactions" in page.html
        assert "PASS" in page.markdown

    def test_empty_index_renders_placeholder(self, tmp_path: Path) -> None:
        page = DataDocsBuilder().index_page(ValidationStore(tmp_path / "empty"))
        assert "Nothing to show" in page.html

    def test_build_writes_the_site(
        self, suite: ExpectationSuite, data: dict[str, list[Any]], tmp_path: Path
    ) -> None:
        store = ValidationStore(tmp_path / "results")
        result = DataValidator(suite).validate(data)
        store.save(result)

        output = DataDocsBuilder(tmp_path / "docs").build(
            suites=[suite], results=[result], store=store
        )
        assert (output / "index.html").is_file()
        assert (output / "suites" / "transactions.html").is_file()
        assert list((output / "validations").glob("transactions-*.html"))

    def test_page_write_returns_both_paths(self, suite: ExpectationSuite, tmp_path: Path) -> None:
        html_path, md_path = DataDocsBuilder().suite_page(suite).write(tmp_path)
        assert html_path.is_file() and md_path.is_file()


# ─── Configuration ───────────────────────────────────────────────────────────


class TestGeConfig:
    """The shipped ge_config.yaml stays loadable and complete."""

    def test_config_file_exists(self) -> None:
        assert CONFIG_PATH.is_file()

    def test_config_declares_required_sections(self) -> None:
        yaml = pytest.importorskip("yaml")
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        assert {"stores", "datasources", "profiling", "suites", "ci"} <= set(config)
        assert config["ci"]["fail_on_error"] is True

    def test_configured_suites_are_buildable(self) -> None:
        yaml = pytest.importorskip("yaml")
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        for name, definition in config["suites"].items():
            built = ExpectationSuite.from_dict(
                {
                    "expectation_suite_name": name,
                    "expectations": definition["expectations"],
                }
            )
            assert len(built) == len(definition["expectations"])
