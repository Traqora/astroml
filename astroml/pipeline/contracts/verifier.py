from astroml.utils.exceptions import AstroMLError
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from astroml.pipeline.contracts.quality_contract import QualityContract, QualityValidationResult
from astroml.pipeline.contracts.schema_contract import SchemaContract, SchemaValidationResult
from astroml.pipeline.contracts.semantic_contract import SemanticContract, SemanticValidationResult

logger = logging.getLogger(__name__)

_AnyContract = SchemaContract | QualityContract | SemanticContract
_AnyResult = SchemaValidationResult | QualityValidationResult | SemanticValidationResult


class MissingContractError(KeyError):
    """Raised in strict mode when a requested contract name is not registered.

    Without strict mode, `ContractVerifier.verify` silently skips an
    unknown contract name (logging a warning) and simply excludes it from
    `total_contracts`/`passed_contracts`/`failed_contracts`. If every
    contract in a stage's name list is misspelled or was removed without
    updating the pipeline config, `verify` returns `passed=True` with
    `total_contracts=0`: a CI data-contract gate configured this way
    silently stops validating anything instead of failing the build,
    which is exactly the failure mode "CI validation" should catch.
    """


@dataclass
class ContractBreach:
    """Record of a single contract breach.

    Attributes:
        contract_name: Name of the contract that was breached.
        contract_type: Type of contract (schema, quality, semantic).
        timestamp: When the breach occurred.
        details: Dict with breach details.
    """

    contract_name: str
    contract_type: str
    timestamp: str
    details: dict[str, Any]


@dataclass
class ContractResult:
    """Result of verifying a single contract.

    Attributes:
        name: Name of the contract.
        contract_type: Type of contract.
        passed: Whether the contract passed.
        details: Detailed validation result.
    """

    name: str
    contract_type: str
    passed: bool
    details: _AnyResult


@dataclass
class VerificationResult:
    """Overall result of running multiple contract verifications.

    Attributes:
        passed: Whether all contracts passed.
        results: List of individual ContractResult per contract.
        total_contracts: Number of contracts that were verified.
        passed_contracts: Number of contracts that passed.
        failed_contracts: Number of contracts that failed.
    """

    passed: bool
    results: list[ContractResult] = field(default_factory=list)
    total_contracts: int = 0
    passed_contracts: int = 0
    failed_contracts: int = 0


@dataclass
class PipelineStageResult:
    """Result of verifying a pipeline stage.

    Attributes:
        stage_name: Name of the pipeline stage.
        passed: Whether the stage passed verification.
        results: List of individual ContractResult for this stage.
    """

    stage_name: str
    passed: bool
    results: list[ContractResult]


@dataclass
class PipelineVerificationResult:
    """Overall result of verifying a multi-stage pipeline.

    Attributes:
        passed: Whether all stages passed.
        stages: List of PipelineStageResult per stage.
    """

    passed: bool
    stages: list[PipelineStageResult]


class ContractVerifier:
    """Runs schema, quality, and semantic contracts against DataFrames.

    Supports alerting callbacks on failure and maintains a breach history.

    Attributes:
        contracts: Dict mapping contract names to contract instances.
        _failure_callbacks: List of callbacks invoked on verification failure.
        breach_history: List of ContractBreach records.
    """

    def __init__(self) -> None:
        self.contracts: dict[str, _AnyContract] = {}
        self._failure_callbacks: list[Callable[[ContractBreach], None]] = []
        self.breach_history: list[ContractBreach] = []

    def add_contract(self, contract: _AnyContract, name: str) -> None:
        """Add a single contract with a given name.

        Args:
            contract: SchemaContract, QualityContract, or SemanticContract instance.
            name: Name to register the contract under.
        """
        self.contracts[name] = contract

    def add_contracts(self, contracts_dict: dict[str, _AnyContract]) -> None:
        """Add multiple contracts at once.

        Args:
            contracts_dict: Dict mapping names to contract instances.
        """
        self.contracts.update(contracts_dict)

    def verify(
        self,
        df: pd.DataFrame,
        contract_names: list[str] | None = None,
        *,
        strict: bool = False,
    ) -> VerificationResult:
        """Run verification on specified (or all) contracts.

        Args:
            df: DataFrame to validate.
            contract_names: Optional list of contract names to run.
                If None, runs all registered contracts.
            strict: If True, raise MissingContractError immediately when
                any name in `contract_names` is not registered, instead of
                logging a warning and silently excluding it. Defaults to
                False to preserve existing callers' behavior; a CI
                validation entry point should pass `strict=True` so a
                misconfigured (misspelled or stale) contract name fails
                the build loudly rather than quietly validating nothing.

        Returns:
            VerificationResult aggregating all contract results.

        Raises:
            MissingContractError: If `strict` is True and `contract_names`
                includes a name not registered on this verifier.
        """
        names_to_run = contract_names or list(self.contracts.keys())
        results: list[ContractResult] = []

        for name in names_to_run:
            contract = self.contracts.get(name)
            if contract is None:
                if strict:
                    raise MissingContractError(
                        f"Contract '{name}' is not registered on this verifier "
                        f"(known contracts: {sorted(self.contracts.keys())})"
                    )
                logger.warning("Contract '%s' not found, skipping", name)
                continue

            contract_type = self._get_contract_type(contract)
            try:
                result = contract.validate(df)
                passed = self._is_passed(result)
                cr = ContractResult(
                    name=name,
                    contract_type=contract_type,
                    passed=passed,
                    details=result,
                )
                if not passed:
                    self._record_breach(name, contract_type, result)
            except AstroMLError as e:
                logger.error("Contract '%s' validation failed with error: %s", name, e)
                cr = ContractResult(
                    name=name,
                    contract_type=contract_type,
                    passed=False,
                    details=str(e),
                )
                passed = False
                self._record_breach(name, contract_type, {"error": str(e)})

            results.append(cr)

        passed = all(r.passed for r in results)
        passed_count = sum(1 for r in results if r.passed)
        failed_count = sum(1 for r in results if not r.passed)

        return VerificationResult(
            passed=passed,
            results=results,
            total_contracts=len(results),
            passed_contracts=passed_count,
            failed_contracts=failed_count,
        )

    def verify_pipeline(
        self,
        df: pd.DataFrame,
        pipeline_stages: dict[str, list[str]],
        *,
        strict: bool = False,
    ) -> PipelineVerificationResult:
        """Verify contracts at each pipeline stage.

        Args:
            df: DataFrame to validate (same df verified at each stage).
            pipeline_stages: Dict mapping stage names to lists of contract names
                to verify at that stage.
            strict: Forwarded to `verify` for every stage; see its
                docstring. Recommended for CI validation entry points.

        Returns:
            PipelineVerificationResult with per-stage results.

        Raises:
            MissingContractError: If `strict` is True and any stage names
                a contract not registered on this verifier.
        """
        stage_results: list[PipelineStageResult] = []

        for stage_name, contract_names in pipeline_stages.items():
            verification = self.verify(df, contract_names=contract_names, strict=strict)
            stage_results.append(
                PipelineStageResult(
                    stage_name=stage_name,
                    passed=verification.passed,
                    results=verification.results,
                )
            )

        all_passed = all(s.passed for s in stage_results)
        return PipelineVerificationResult(passed=all_passed, stages=stage_results)

    def on_failure(self, callback: Callable[[ContractBreach], None]) -> None:
        """Register a callback to be invoked when a contract fails.

        Args:
            callback: Callable that accepts a ContractBreach.
        """
        self._failure_callbacks.append(callback)

    def _record_breach(self, name: str, contract_type: str, details: Any) -> None:
        breach = ContractBreach(
            contract_name=name,
            contract_type=contract_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details if isinstance(details, dict) else {"info": str(details)},
        )
        self.breach_history.append(breach)
        for cb in self._failure_callbacks:
            try:
                cb(breach)
            except AstroMLError as e:
                logger.error("Failure callback error: %s", e)

    @staticmethod
    def _get_contract_type(contract: _AnyContract) -> str:
        if isinstance(contract, SchemaContract):
            return "schema"
        elif isinstance(contract, QualityContract):
            return "quality"
        elif isinstance(contract, SemanticContract):
            return "semantic"
        return "unknown"

    @staticmethod
    def _is_passed(result: _AnyResult) -> bool:
        if hasattr(result, "is_valid"):
            return bool(result.is_valid)
        return False
