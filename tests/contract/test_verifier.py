from __future__ import annotations

import pandas as pd
import pytest

from astroml.pipeline.contracts.quality_contract import QualityContract
from astroml.pipeline.contracts.schema_contract import SchemaContract
from astroml.pipeline.contracts.semantic_contract import SemanticContract
from astroml.pipeline.contracts.verifier import (
    ContractBreach,
    ContractVerifier,
    MissingContractError,
    PipelineVerificationResult,
    VerificationResult,
)


class TestContractVerifier:
    def test_init_empty(self) -> None:
        verifier = ContractVerifier()
        assert verifier.contracts == {}
        assert verifier.breach_history == []

    def test_add_contract(self) -> None:
        verifier = ContractVerifier()
        contract = SchemaContract(name="test")
        verifier.add_contract(contract, "my_schema")
        assert "my_schema" in verifier.contracts
        assert verifier.contracts["my_schema"] is contract

    def test_add_contracts(self) -> None:
        verifier = ContractVerifier()
        schema = SchemaContract(name="schema")
        quality = QualityContract(name="quality")
        verifier.add_contracts({"s": schema, "q": quality})
        assert len(verifier.contracts) == 2

    def test_verify_all_passes(self) -> None:
        verifier = ContractVerifier()
        schema = SchemaContract.from_schema(
            {"columns": {"id": {"dtype": "int64", "nullable": False}}},
            name="schema",
        )
        verifier.add_contract(schema, "schema_check")
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = verifier.verify(df)
        assert result.passed
        assert result.total_contracts == 1
        assert result.passed_contracts == 1

    def test_verify_fails(self) -> None:
        verifier = ContractVerifier()
        schema = SchemaContract.from_schema(
            {"columns": {"id": {"dtype": "int64", "nullable": False}}},
            name="schema",
        )
        verifier.add_contract(schema, "schema_check")
        df = pd.DataFrame({"id": [1, None, 3]})
        result = verifier.verify(df)
        assert not result.passed
        assert result.failed_contracts == 1

    def test_verify_with_contract_names(self) -> None:
        verifier = ContractVerifier()
        schema = SchemaContract(name="schema")
        quality = QualityContract(name="quality")
        verifier.add_contract(schema, "s")
        verifier.add_contract(quality, "q")
        df = pd.DataFrame({"a": [1]})
        result = verifier.verify(df, contract_names=["s"])
        assert result.total_contracts == 1

    def test_verify_unknown_contract(self) -> None:
        # Pinning the existing, surprising default behaviour: a name that
        # is not registered is silently skipped (warning-logged only), so
        # a config where EVERY requested name is misspelled or stale
        # reports passed=True with total_contracts=0 rather than failing.
        # See test_verify_unknown_contract_raises_in_strict_mode below for
        # the opt-in fix (issue #968).
        verifier = ContractVerifier()
        schema = SchemaContract(name="schema")
        verifier.add_contract(schema, "s")
        df = pd.DataFrame({"a": [1]})
        result = verifier.verify(df, contract_names=["unknown"])
        assert result.passed
        assert result.total_contracts == 0

    def test_verify_unknown_contract_raises_in_strict_mode(self) -> None:
        # issue #968: a CI validation entry point must not be able to
        # silently no-op when its contract config drifts from what's
        # actually registered. strict=True turns that into a loud failure.
        verifier = ContractVerifier()
        schema = SchemaContract(name="schema")
        verifier.add_contract(schema, "s")
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(MissingContractError, match="unknown"):
            verifier.verify(df, contract_names=["unknown"], strict=True)

    def test_verify_strict_mode_does_not_affect_known_contracts(self) -> None:
        verifier = ContractVerifier()
        schema = SchemaContract.from_schema({"columns": {"a": {"dtype": "int64"}}}, name="schema")
        verifier.add_contract(schema, "s")
        df = pd.DataFrame({"a": [1]})
        result = verifier.verify(df, contract_names=["s"], strict=True)
        assert result.passed
        assert result.total_contracts == 1

    def test_verify_strict_mode_defaults_to_false(self) -> None:
        # Backward compatibility: existing callers that don't pass `strict`
        # keep the historical silent-skip behaviour.
        verifier = ContractVerifier()
        df = pd.DataFrame({"a": [1]})
        result = verifier.verify(df, contract_names=["unknown"])
        assert result.passed

    def test_verify_with_multiple_contracts(self) -> None:
        verifier = ContractVerifier()
        schema = SchemaContract.from_schema({"columns": {"a": {"dtype": "int64"}}}, name="schema")
        quality = QualityContract(name="quality")
        quality.add_constraint("a", "range", {"min": 0})
        verifier.add_contracts({"s": schema, "q": quality})
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = verifier.verify(df)
        assert result.passed
        assert result.total_contracts == 2
        assert result.passed_contracts == 2

    def test_verify_mixed_results(self) -> None:
        verifier = ContractVerifier()
        good = SchemaContract.from_schema({"columns": {"a": {"dtype": "int64"}}}, name="good")
        bad = SchemaContract.from_schema({"columns": {"b": {"dtype": "int64"}}}, name="bad")
        verifier.add_contracts({"good": good, "bad": bad})
        df = pd.DataFrame({"a": [1]})
        result = verifier.verify(df)
        assert not result.passed
        assert result.passed_contracts == 1
        assert result.failed_contracts == 1


class TestVerifierBreachHistory:
    def test_breach_recorded(self) -> None:
        verifier = ContractVerifier()
        schema = SchemaContract.from_schema(
            {"columns": {"id": {"dtype": "int64", "nullable": False}}}
        )
        verifier.add_contract(schema, "test")
        df = pd.DataFrame({"id": [None]})
        verifier.verify(df)
        assert len(verifier.breach_history) == 1
        breach = verifier.breach_history[0]
        assert breach.contract_name == "test"
        assert breach.contract_type == "schema"
        assert breach.timestamp != ""

    def test_breach_not_recorded_on_pass(self) -> None:
        verifier = ContractVerifier()
        schema = SchemaContract.from_schema({"columns": {"id": {"dtype": "int64"}}})
        verifier.add_contract(schema, "test")
        df = pd.DataFrame({"id": [1, 2]})
        verifier.verify(df)
        assert len(verifier.breach_history) == 0

    def test_breach_history_is_list_of_contract_breach(self) -> None:
        verifier = ContractVerifier()
        schema = SchemaContract.from_schema(
            {"columns": {"id": {"dtype": "int64", "nullable": False}}}
        )
        verifier.add_contract(schema, "test")
        df = pd.DataFrame({"id": [None]})
        verifier.verify(df)
        assert len(verifier.breach_history) == 1
        assert isinstance(verifier.breach_history[0], ContractBreach)


class TestVerifierCallbacks:
    def test_on_failure_callback_called(self) -> None:
        verifier = ContractVerifier()
        callback_results: list[ContractBreach] = []

        def callback(breach: ContractBreach) -> None:
            callback_results.append(breach)

        verifier.on_failure(callback)
        schema = SchemaContract.from_schema(
            {"columns": {"id": {"dtype": "int64", "nullable": False}}}
        )
        verifier.add_contract(schema, "test")
        df = pd.DataFrame({"id": [None]})
        verifier.verify(df)
        assert len(callback_results) == 1
        assert callback_results[0].contract_name == "test"

    def test_on_failure_not_called_on_pass(self) -> None:
        verifier = ContractVerifier()
        callback_results: list[ContractBreach] = []

        def callback(breach: ContractBreach) -> None:
            callback_results.append(breach)

        verifier.on_failure(callback)
        schema = SchemaContract.from_schema({"columns": {"id": {"dtype": "int64"}}})
        verifier.add_contract(schema, "test")
        df = pd.DataFrame({"id": [1]})
        verifier.verify(df)
        assert len(callback_results) == 0

    def test_multiple_callbacks(self) -> None:
        verifier = ContractVerifier()
        count = [0]

        def cb1(b: ContractBreach) -> None:
            count[0] += 1

        def cb2(b: ContractBreach) -> None:
            count[0] += 1

        verifier.on_failure(cb1)
        verifier.on_failure(cb2)
        schema = SchemaContract.from_schema(
            {"columns": {"id": {"dtype": "int64", "nullable": False}}}
        )
        verifier.add_contract(schema, "test")
        df = pd.DataFrame({"id": [None]})
        verifier.verify(df)
        assert count[0] == 2


class TestVerifierPipeline:
    def test_verify_pipeline_all_pass(self) -> None:
        verifier = ContractVerifier()
        schema = SchemaContract.from_schema({"columns": {"id": {"dtype": "int64"}}}, name="schema")
        quality = QualityContract(name="quality")
        verifier.add_contracts({"s": schema, "q": quality})
        df = pd.DataFrame({"id": [1, 2]})
        result = verifier.verify_pipeline(
            df,
            {
                "stage1": ["s"],
                "stage2": ["q"],
            },
        )
        assert isinstance(result, PipelineVerificationResult)
        assert result.passed
        assert len(result.stages) == 2
        assert result.stages[0].stage_name == "stage1"
        assert result.stages[1].stage_name == "stage2"
        assert result.stages[0].passed
        assert result.stages[1].passed

    def test_verify_pipeline_fails(self) -> None:
        verifier = ContractVerifier()
        schema = SchemaContract.from_schema(
            {"columns": {"id": {"dtype": "int64", "nullable": False}}}, name="schema"
        )
        verifier.add_contract(schema, "s")
        df = pd.DataFrame({"id": [None]})
        result = verifier.verify_pipeline(df, {"stage1": ["s"]})
        assert not result.passed
        assert not result.stages[0].passed

    def test_verify_pipeline_multiple_contracts_per_stage(self) -> None:
        verifier = ContractVerifier()
        s1 = SchemaContract(name="s1")
        s2 = SchemaContract(name="s2")
        verifier.add_contracts({"s1": s1, "s2": s2})
        df = pd.DataFrame({"a": [1]})
        result = verifier.verify_pipeline(df, {"pre": ["s1", "s2"]})
        assert result.passed
        assert len(result.stages[0].results) == 2

    def test_verify_pipeline_silently_passes_an_all_unknown_stage_by_default(self) -> None:
        # The bug issue #968 targets, reproduced at the pipeline level: a
        # stage whose contract names are entirely stale/misspelled reports
        # passed=True for that stage.
        verifier = ContractVerifier()
        df = pd.DataFrame({"a": [1]})
        result = verifier.verify_pipeline(df, {"stage1": ["typo_contract_name"]})
        assert result.passed
        assert result.stages[0].passed
        assert result.stages[0].results == []

    def test_verify_pipeline_strict_mode_raises_on_unknown_contract_in_any_stage(self) -> None:
        verifier = ContractVerifier()
        schema = SchemaContract(name="schema")
        verifier.add_contract(schema, "s")
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(MissingContractError):
            verifier.verify_pipeline(
                df, {"stage1": ["s"], "stage2": ["typo_contract_name"]}, strict=True
            )


class TestVerifierResultTypes:
    def test_verify_returns_verification_result(self) -> None:
        verifier = ContractVerifier()
        contract = SchemaContract(name="test")
        verifier.add_contract(contract, "t")
        df = pd.DataFrame({"a": [1]})
        result = verifier.verify(df)
        assert isinstance(result, VerificationResult)

    def test_verification_result_attributes(self) -> None:
        verifier = ContractVerifier()
        contract = SchemaContract.from_schema({"columns": {"a": {"dtype": "int64"}}}, name="test")
        verifier.add_contract(contract, "t")
        df = pd.DataFrame({"a": [1]})
        result = verifier.verify(df)
        assert hasattr(result, "passed")
        assert hasattr(result, "results")
        assert hasattr(result, "total_contracts")
        assert hasattr(result, "passed_contracts")
        assert hasattr(result, "failed_contracts")

    def test_verify_contract_validate_raises_exception(self) -> None:
        verifier = ContractVerifier()

        class BrokenContract:
            name = "broken"
            version = "1.0.0"

            def validate(self, df: pd.DataFrame) -> None:
                raise RuntimeError("Something broke")

        verifier.add_contract(BrokenContract(), "broken")  # type: ignore[arg-type]
        df = pd.DataFrame({"a": [1]})
        result = verifier.verify(df)
        assert not result.passed
        assert result.failed_contracts == 1
        # Breach should be recorded with error info
        assert len(verifier.breach_history) == 1
