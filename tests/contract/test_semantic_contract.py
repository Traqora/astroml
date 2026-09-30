from __future__ import annotations

import pandas as pd
import pytest

from astroml.pipeline.contracts.semantic_contract import SemanticContract, SemanticValidationResult


class TestSemanticContractCreation:
    def test_init_defaults(self) -> None:
        contract = SemanticContract()
        assert contract.name == ""
        assert contract.version == "1.0.0"
        assert contract.rules == []

    def test_add_rule(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "rule1",
            {"type": "custom_expression", "expr": "df['a'].sum() > 0"},
            description="Test rule",
        )
        assert len(contract.rules) == 1
        assert contract.rules[0]["rule_name"] == "rule1"
        assert contract.rules[0]["description"] == "Test rule"

    def test_add_rule_invalid_type(self) -> None:
        contract = SemanticContract()
        with pytest.raises(ValueError, match="Unsupported rule type"):
            contract.add_rule("bad", {"type": "nonexistent"})


class TestReferentialIntegrity:
    def test_passes_when_values_exist(self) -> None:
        contract = SemanticContract()
        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "account_id": [10, 20, 30],
                "name": ["a", "b", "c"],
            }
        )
        passed, msg = contract.check_referential_integrity(
            df, "user_id", reference_column="account_id"
        )
        assert not passed  # 1,2,3 not in {10,20,30}

    def test_fails_when_values_missing(self) -> None:
        contract = SemanticContract()
        df = pd.DataFrame({"fk": [1, 2, 5], "ref": [1, 2, 3]})
        passed, msg = contract.check_referential_integrity(df, "fk", reference_column="ref")
        assert not passed
        assert "5" in msg

    def test_with_explicit_values(self) -> None:
        contract = SemanticContract()
        df = pd.DataFrame({"fk": [1, 2, 3]})
        passed, msg = contract.check_referential_integrity(df, "fk", reference_values=[1, 2, 3, 4])
        assert passed

    def test_with_explicit_values_missing(self) -> None:
        contract = SemanticContract()
        df = pd.DataFrame({"fk": [1, 2, 99]})
        passed, msg = contract.check_referential_integrity(df, "fk", reference_values=[1, 2, 3])
        assert not passed

    def test_missing_fk_column(self) -> None:
        contract = SemanticContract()
        df = pd.DataFrame({"a": [1]})
        passed, msg = contract.check_referential_integrity(df, "fk")
        assert not passed
        assert "not found" in msg

    def test_missing_reference(self) -> None:
        contract = SemanticContract()
        df = pd.DataFrame({"fk": [1]})
        passed, msg = contract.check_referential_integrity(df, "fk")
        assert not passed
        assert "No reference" in msg

    def test_referential_integrity_rule(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "ref_check",
            {
                "type": "referential_integrity",
                "foreign_key": "fk",
                "reference_column": "ref",
            },
        )
        df = pd.DataFrame({"fk": [1, 2, 3], "ref": [1, 2, 3]})
        result = contract.validate(df)
        assert result.is_valid


class TestFunctionalDependency:
    def test_fd_holds(self) -> None:
        contract = SemanticContract()
        df = pd.DataFrame(
            {
                "zip": ["10001", "10001", "90210"],
                "city": ["NYC", "NYC", "Beverly Hills"],
            }
        )
        passed, msg = contract.check_functional_dependency(df, "zip", "city")
        assert passed

    def test_fd_violated(self) -> None:
        contract = SemanticContract()
        df = pd.DataFrame(
            {
                "zip": ["10001", "10001", "10001"],
                "city": ["NYC", "LA", "NYC"],
            }
        )
        passed, msg = contract.check_functional_dependency(df, "zip", "city")
        assert not passed
        assert "violation" in msg

    def test_fd_empty_data(self) -> None:
        contract = SemanticContract()
        df = pd.DataFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="int64")})
        passed, msg = contract.check_functional_dependency(df, "a", "b")
        assert passed

    def test_fd_missing_determinant(self) -> None:
        contract = SemanticContract()
        df = pd.DataFrame({"x": [1]})
        passed, msg = contract.check_functional_dependency(df, "a", "x")
        assert not passed

    def test_fd_missing_dependent(self) -> None:
        contract = SemanticContract()
        df = pd.DataFrame({"a": [1]})
        passed, msg = contract.check_functional_dependency(df, "a", "x")
        assert not passed

    def test_fd_rule(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "fd_zip_city",
            {"type": "functional_dependency", "determinant": "zip", "dependent": "city"},
        )
        df = pd.DataFrame({"zip": ["A", "A"], "city": ["NYC", "NYC"]})
        result = contract.validate(df)
        assert result.is_valid


class TestConditionalRequired:
    def test_condition_met_has_value(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "if_active_then_email",
            {
                "type": "conditional_required",
                "condition_column": "status",
                "condition_value": "active",
                "required_column": "email",
            },
        )
        df = pd.DataFrame(
            {
                "status": ["active", "inactive", "active"],
                "email": ["a@b.com", None, "c@d.com"],
            }
        )
        result = contract.validate(df)
        assert result.is_valid

    def test_condition_met_missing_value(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "if_active_then_email",
            {
                "type": "conditional_required",
                "condition_column": "status",
                "condition_value": "active",
                "required_column": "email",
            },
        )
        df = pd.DataFrame(
            {
                "status": ["active", "inactive"],
                "email": [None, None],
            }
        )
        result = contract.validate(df)
        assert not result.is_valid

    def test_missing_condition_column(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "test",
            {
                "type": "conditional_required",
                "condition_column": "missing",
                "condition_value": "x",
                "required_column": "email",
            },
        )
        df = pd.DataFrame({"email": ["a@b.com"]})
        result = contract.validate(df)
        assert not result.is_valid

    def test_missing_required_column(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "test",
            {
                "type": "conditional_required",
                "condition_column": "status",
                "condition_value": "active",
                "required_column": "missing",
            },
        )
        df = pd.DataFrame({"status": ["active"]})
        result = contract.validate(df)
        assert not result.is_valid


class TestCustomExpression:
    def test_custom_expression_passes_series(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "positive_values",
            {"type": "custom_expression", "expr": "df['value'] > 0"},
        )
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = contract.validate(df)
        assert result.is_valid

    def test_custom_expression_fails_series(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "positive_values",
            {"type": "custom_expression", "expr": "df['value'] > 0"},
        )
        df = pd.DataFrame({"value": [1, -1, 3]})
        result = contract.validate(df)
        assert not result.is_valid

    def test_custom_expression_bool_true(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "check_sum",
            {"type": "custom_expression", "expr": "df['a'].sum() > 0"},
        )
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = contract.validate(df)
        assert result.is_valid

    def test_custom_expression_bool_false(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "check_sum",
            {"type": "custom_expression", "expr": "df['a'].sum() < 0"},
        )
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = contract.validate(df)
        assert not result.is_valid

    def test_custom_expression_empty(self) -> None:
        contract = SemanticContract()
        contract.add_rule("empty", {"type": "custom_expression", "expr": ""})
        df = pd.DataFrame({"a": [1]})
        result = contract.validate(df)
        assert not result.is_valid

    def test_custom_expression_syntax_error(self) -> None:
        contract = SemanticContract()
        contract.add_rule("bad", {"type": "custom_expression", "expr": "invalid syntax !@#"})
        df = pd.DataFrame({"a": [1]})
        result = contract.validate(df)
        assert not result.is_valid

    def test_custom_expression_restricted_builtins(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "safe_access",
            {"type": "custom_expression", "expr": "df['a'].notna().all()"},
        )
        df = pd.DataFrame({"a": [1]})
        result = contract.validate(df)
        assert result.is_valid

    def test_multiple_rules(self) -> None:
        contract = SemanticContract()
        contract.add_rule("pos", {"type": "custom_expression", "expr": "df['x'] > 0"})
        contract.add_rule("lt_100", {"type": "custom_expression", "expr": "df['x'] < 100"})
        df = pd.DataFrame({"x": [10, 20, 50]})
        result = contract.validate(df)
        assert result.is_valid
        assert len(result.rule_results) == 2
        assert all(r["passed"] for r in result.rule_results)


class TestSemanticContractEdgeCases:
    def test_no_rules(self) -> None:
        contract = SemanticContract()
        df = pd.DataFrame({"a": [1]})
        result = contract.validate(df)
        assert result.is_valid
        assert result.rule_results == []

    def test_rule_exception_handling(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "crash",
            {"type": "custom_expression", "expr": "1/0"},
        )
        df = pd.DataFrame({"a": [1]})
        result = contract.validate(df)
        assert not result.is_valid

    def test_result_type(self) -> None:
        contract = SemanticContract(name="sem_test")
        df = pd.DataFrame({"a": [1]})
        result = contract.validate(df)
        assert isinstance(result, SemanticValidationResult)
        assert result.contract_name == "sem_test"

    def test_unknown_rule_type(self) -> None:
        contract = SemanticContract()
        contract.rules.append(
            {
                "rule_name": "unknown",
                "type": "nonexistent_rule_type",
                "expression": {"type": "nonexistent_rule_type"},
                "description": "",
            }
        )
        df = pd.DataFrame({"a": [1]})
        result = contract.validate(df)
        assert not result.is_valid
        assert not result.rule_results[0]["passed"]
        assert "Unknown rule type" in result.rule_results[0]["message"]

    def test_validate_rule_raises_exception(self) -> None:
        contract = SemanticContract()
        contract.add_rule(
            "crash",
            {"type": "custom_expression", "expr": "df['missing_col']"},
        )
        df = pd.DataFrame({"a": [1]})
        result = contract.validate(df)
        assert not result.is_valid
        assert len(result.rule_results) == 1
        assert not result.rule_results[0]["passed"]
        assert "Expression evaluation failed" in result.rule_results[0]["message"]
