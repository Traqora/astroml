from __future__ import annotations

import pandas as pd
import pytest

from astroml.pipeline.contracts.quality_contract import QualityContract, QualityValidationResult


class TestQualityContractCreation:
    def test_init_defaults(self) -> None:
        contract = QualityContract()
        assert contract.name == ""
        assert contract.version == "1.0.0"
        assert contract.constraints == []

    def test_init_custom(self) -> None:
        contract = QualityContract(name="qc", version="2.1.0")
        assert contract.name == "qc"
        assert contract.version == "2.1.0"

    def test_add_constraint(self) -> None:
        contract = QualityContract()
        contract.add_constraint("age", "range", {"min": 0, "max": 120})
        assert len(contract.constraints) == 1
        c = contract.constraints[0]
        assert c["column"] == "age"
        assert c["type"] == "range"
        assert c["threshold"] == {"min": 0, "max": 120}

    def test_add_constraint_invalid_type(self) -> None:
        contract = QualityContract()
        with pytest.raises(ValueError, match="Unsupported constraint type"):
            contract.add_constraint("col", "invalid_type")

    def test_from_config(self) -> None:
        config = {
            "name": "from_config",
            "version": "1.5.0",
            "constraints": [
                {"column": "age", "type": "range", "threshold": {"min": 0}},
                {"column": "name", "type": "unique"},
            ],
        }
        contract = QualityContract.from_config(config)
        assert contract.name == "from_config"
        assert contract.version == "1.5.0"
        assert len(contract.constraints) == 2

    def test_from_config_empty(self) -> None:
        contract = QualityContract.from_config({})
        assert contract.name == ""
        assert contract.version == "1.0.0"
        assert contract.constraints == []

    def test_merge(self) -> None:
        c1 = QualityContract(name="c1")
        c1.add_constraint("a", "range", {"min": 0})
        c2 = QualityContract(name="c2")
        c2.add_constraint("b", "unique")
        merged = c1.merge(c2)
        assert merged.name == "c1+c2"
        assert len(merged.constraints) == 2

    def test_merge_empty(self) -> None:
        c1 = QualityContract(name="c1")
        c1.add_constraint("a", "range", {"min": 0})
        c2 = QualityContract()
        merged = c1.merge(c2)
        assert merged.name == "c1"
        assert len(merged.constraints) == 1


class TestQualityContractValidation:
    def test_range_min_violation(self) -> None:
        contract = QualityContract()
        contract.add_constraint("age", "range", {"min": 0})
        df = pd.DataFrame({"age": [-1, 5, 10]})
        result = contract.validate(df)
        assert not result.is_valid
        assert result.failed_constraints == 1

    def test_range_max_violation(self) -> None:
        contract = QualityContract()
        contract.add_constraint("age", "range", {"max": 100})
        df = pd.DataFrame({"age": [50, 150]})
        result = contract.validate(df)
        assert not result.is_valid
        assert result.failed_constraints == 1

    def test_range_pass(self) -> None:
        contract = QualityContract()
        contract.add_constraint("age", "range", {"min": 0, "max": 120})
        df = pd.DataFrame({"age": [25, 50, 75]})
        result = contract.validate(df)
        assert result.is_valid
        assert result.passed_constraints == 1

    def test_range_non_numeric(self) -> None:
        contract = QualityContract()
        contract.add_constraint("name", "range", {"min": 0})
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        result = contract.validate(df)
        assert not result.is_valid

    def test_null_ratio_violation(self) -> None:
        contract = QualityContract()
        contract.add_constraint("col", "null_ratio", {"max": 0.1})
        df = pd.DataFrame({"col": [1, None, None, 4, None]})
        result = contract.validate(df)
        assert not result.is_valid
        assert result.failed_constraints == 1

    def test_null_ratio_pass(self) -> None:
        contract = QualityContract()
        contract.add_constraint("col", "null_ratio", {"max": 0.5})
        df = pd.DataFrame({"col": [1, None, 3]})
        result = contract.validate(df)
        assert result.is_valid

    def test_distinct_count_min_violation(self) -> None:
        contract = QualityContract()
        contract.add_constraint("col", "distinct_count", {"min": 5})
        df = pd.DataFrame({"col": [1, 1, 2, 2]})
        result = contract.validate(df)
        assert not result.is_valid

    def test_distinct_count_max_violation(self) -> None:
        contract = QualityContract()
        contract.add_constraint("col", "distinct_count", {"max": 3})
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        result = contract.validate(df)
        assert not result.is_valid

    def test_distinct_count_pass(self) -> None:
        contract = QualityContract()
        contract.add_constraint("col", "distinct_count", {"min": 2, "max": 5})
        df = pd.DataFrame({"col": [1, 2, 3, 1, 2]})
        result = contract.validate(df)
        assert result.is_valid

    def test_unique_violation(self) -> None:
        contract = QualityContract()
        contract.add_constraint("id", "unique")
        df = pd.DataFrame({"id": [1, 2, 2, 3]})
        result = contract.validate(df)
        assert not result.is_valid

    def test_unique_pass(self) -> None:
        contract = QualityContract()
        contract.add_constraint("id", "unique")
        df = pd.DataFrame({"id": [1, 2, 3, 4]})
        result = contract.validate(df)
        assert result.is_valid

    def test_value_set_violation(self) -> None:
        contract = QualityContract()
        contract.add_constraint("status", "value_set", {"values": ["active", "inactive"]})
        df = pd.DataFrame({"status": ["active", "unknown", "inactive"]})
        result = contract.validate(df)
        assert not result.is_valid

    def test_value_set_pass(self) -> None:
        contract = QualityContract()
        contract.add_constraint("status", "value_set", {"values": ["active", "inactive"]})
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})
        result = contract.validate(df)
        assert result.is_valid

    def test_value_set_empty_allowed(self) -> None:
        contract = QualityContract()
        contract.add_constraint("col", "value_set", {"values": []})
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = contract.validate(df)
        assert result.is_valid

    def test_missing_column(self) -> None:
        contract = QualityContract()
        contract.add_constraint("missing_col", "unique")
        df = pd.DataFrame({"other": [1, 2]})
        result = contract.validate(df)
        assert not result.is_valid
        assert result.failed_constraints == 1

    def test_no_constraints(self) -> None:
        contract = QualityContract()
        df = pd.DataFrame({"a": [1]})
        result = contract.validate(df)
        assert result.is_valid
        assert result.total_constraints == 0

    def test_result_type(self) -> None:
        contract = QualityContract(name="qc_test")
        df = pd.DataFrame({"a": [1]})
        result = contract.validate(df)
        assert isinstance(result, QualityValidationResult)
        assert result.contract_name == "qc_test"


class TestDistributionSimilarity:
    def test_distribution_pass(self) -> None:
        contract = QualityContract()
        contract.add_constraint(
            "value",
            "distribution_similarity",
            {"reference": [1.0, 2.0, 3.0, 4.0, 5.0], "tolerance": 1.0},
        )
        df = pd.DataFrame({"value": [1.1, 2.1, 3.1, 4.1, 5.1]})
        result = contract.validate(df)
        assert result.is_valid

    def test_distribution_no_reference(self) -> None:
        contract = QualityContract()
        contract.add_constraint("value", "distribution_similarity", {})
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = contract.validate(df)
        assert result.is_valid
