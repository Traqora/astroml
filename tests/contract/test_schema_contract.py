from __future__ import annotations

import pandas as pd
import pytest

from astroml.pipeline.contracts.schema_contract import SchemaContract, SchemaValidationResult


class TestSchemaContractCreation:
    def test_init_defaults(self) -> None:
        contract = SchemaContract()
        assert contract.name == ""
        assert contract.version == "1.0.0"
        assert contract.columns == {}

    def test_init_custom(self) -> None:
        contract = SchemaContract(name="test", version="2.0.0")
        assert contract.name == "test"
        assert contract.version == "2.0.0"

    def test_from_schema_with_columns_wrapper(self) -> None:
        schema = {
            "columns": {
                "age": {"dtype": "int64", "nullable": False},
                "name": {"dtype": "object", "nullable": True, "unique": True},
            }
        }
        contract = SchemaContract.from_schema(schema, name="my_contract", version="1.0.0")
        assert contract.name == "my_contract"
        assert contract.version == "1.0.0"
        assert "age" in contract.columns
        assert "name" in contract.columns
        assert contract.columns["age"]["dtype"] == "int64"
        assert contract.columns["age"]["nullable"] is False
        assert contract.columns["name"]["unique"] is True

    def test_from_schema_without_columns_wrapper(self) -> None:
        schema = {
            "score": {"dtype": "float64", "nullable": True},
        }
        contract = SchemaContract.from_schema(schema)
        assert "score" in contract.columns

    def test_from_schema_defaults(self) -> None:
        schema = {"col": {}}
        contract = SchemaContract.from_schema(schema)
        assert contract.columns["col"]["dtype"] is None
        assert contract.columns["col"]["nullable"] is True
        assert contract.columns["col"]["unique"] is False
        assert contract.columns["col"]["regex"] is None

    def test_from_dataframe(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.0, 2.0, 3.0],
                "c": ["x", "y", "z"],
            }
        )
        contract = SchemaContract.from_dataframe(df, name="df_contract")
        assert contract.name == "df_contract"
        assert "a" in contract.columns
        assert contract.columns["a"]["dtype"] == "int64"
        assert contract.columns["b"]["dtype"] == "float64"
        assert contract.columns["c"]["dtype"] == "object"
        assert contract.columns["a"]["nullable"] is False

    def test_from_dataframe_with_nulls(self) -> None:
        df = pd.DataFrame({"a": [1, None, 3]})
        contract = SchemaContract.from_dataframe(df)
        assert contract.columns["a"]["nullable"] is True


class TestSchemaContractValidation:
    def test_valid_dataframe_passes(self) -> None:
        contract = SchemaContract.from_schema(
            {
                "columns": {
                    "id": {"dtype": "int64", "nullable": False},
                    "name": {"dtype": "object", "nullable": True},
                }
            },
            name="test",
        )
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        result = contract.validate(df)
        assert result.is_valid
        assert result.contract_name == "test"

    def test_missing_columns(self) -> None:
        contract = SchemaContract.from_schema(
            {
                "columns": {
                    "id": {"dtype": "int64"},
                    "name": {"dtype": "object"},
                    "email": {"dtype": "object"},
                }
            }
        )
        df = pd.DataFrame({"id": [1], "name": ["Alice"]})
        result = contract.validate(df)
        assert not result.is_valid
        assert "email" in result.missing_columns

    def test_extra_columns(self) -> None:
        contract = SchemaContract.from_schema({"columns": {"id": {"dtype": "int64"}}})
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "age": [30]})
        result = contract.validate(df)
        # Extra columns are reported but do not cause validation failure
        assert "name" in result.extra_columns
        assert "age" in result.extra_columns
        assert result.is_valid

    def test_type_mismatch(self) -> None:
        contract = SchemaContract.from_schema({"columns": {"value": {"dtype": "int64"}}})
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        result = contract.validate(df)
        assert not result.is_valid
        assert len(result.type_mismatches) == 1
        assert result.type_mismatches[0]["column"] == "value"
        assert result.type_mismatches[0]["expected_type"] == "int64"
        assert result.type_mismatches[0]["actual_type"] == "float64"

    def test_null_constraint_violation(self) -> None:
        contract = SchemaContract.from_schema(
            {"columns": {"name": {"dtype": "object", "nullable": False}}}
        )
        df = pd.DataFrame({"name": ["Alice", None, "Bob"]})
        result = contract.validate(df)
        assert not result.is_valid
        assert len(result.null_constraint_violations) == 1
        assert result.null_constraint_violations[0]["column"] == "name"
        assert result.null_constraint_violations[0]["null_count"] == 1

    def test_unique_constraint_violation(self) -> None:
        contract = SchemaContract.from_schema(
            {"columns": {"id": {"dtype": "int64", "unique": True}}}
        )
        df = pd.DataFrame({"id": [1, 2, 2, 3]})
        result = contract.validate(df)
        assert not result.is_valid
        assert len(result.unique_constraint_violations) == 1
        assert result.unique_constraint_violations[0]["column"] == "id"
        assert result.unique_constraint_violations[0]["duplicate_count"] > 0

    def test_regex_pattern_valid(self) -> None:
        contract = SchemaContract.from_schema(
            {"columns": {"email": {"dtype": "object", "regex": ".*@.*\\..*"}}}
        )
        df = pd.DataFrame({"email": ["a@b.com", "x@y.org", "test@test.com"]})
        result = contract.validate(df)
        assert result.is_valid

    def test_regex_pattern_violation(self) -> None:
        contract = SchemaContract.from_schema(
            {"columns": {"code": {"dtype": "object", "regex": "^[A-Z]{3}$"}}}
        )
        df = pd.DataFrame({"code": ["ABC", "DEF", "abc", "1234"]})
        result = contract.validate(df)
        assert not result.is_valid
        assert len(result.regex_violations) == 1
        assert result.regex_violations[0]["column"] == "code"

    def test_regex_invalid_pattern(self) -> None:
        contract = SchemaContract.from_schema(
            {"columns": {"col": {"dtype": "object", "regex": "[invalid"}}}
        )
        df = pd.DataFrame({"col": ["test"]})
        result = contract.validate(df)
        assert result.is_valid  # invalid regex -> error logged, not counted as failure
        assert len(result.errors) > 0

    def test_empty_dataframe(self) -> None:
        contract = SchemaContract.from_schema(
            {"columns": {"id": {"dtype": "int64"}, "name": {"dtype": "object"}}}
        )
        df = pd.DataFrame({"id": pd.Series(dtype="int64"), "name": pd.Series(dtype="object")})
        result = contract.validate(df)
        assert result.is_valid

    def test_all_checks_combined(self) -> None:
        contract = SchemaContract.from_schema(
            {
                "columns": {
                    "id": {"dtype": "int64", "nullable": False, "unique": True},
                    "email": {"dtype": "object", "nullable": False, "regex": ".*@.*"},
                }
            }
        )
        df = pd.DataFrame(
            {
                "id": [1, 2, 2],
                "email": ["a@b.com", "invalid", None],
            }
        )
        result = contract.validate(df)
        assert not result.is_valid
        assert len(result.null_constraint_violations) == 1
        assert len(result.unique_constraint_violations) == 1
        assert len(result.regex_violations) >= 1


class TestSchemaContractEdgeCases:
    def test_no_columns(self) -> None:
        contract = SchemaContract.from_schema({"columns": {}})
        df = pd.DataFrame({"unexpected": [1]})
        result = contract.validate(df)
        assert result.is_valid  # No expected columns means no constraints to fail

    def test_column_case_sensitivity(self) -> None:
        contract = SchemaContract.from_schema({"columns": {"Name": {"dtype": "object"}}})
        df = pd.DataFrame({"name": ["Alice"]})
        result = contract.validate(df)
        assert "Name" in result.missing_columns
        assert not result.is_valid

    def test_validate_returns_result_type(self) -> None:
        contract = SchemaContract(name="test_contract")
        df = pd.DataFrame({"a": [1]})
        result = contract.validate(df)
        assert isinstance(result, SchemaValidationResult)
        assert result.contract_name == "test_contract"

    def test_from_dataframe_with_categorical(self) -> None:
        df = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c"])})
        contract = SchemaContract.from_dataframe(df)
        assert contract.columns["cat"]["dtype"] == "category"
