"""Comprehensive tests for ModelValidator, RobustnessEvaluator, and ComplianceChecker."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers.validation import router as validation_router
from astroml.validation.compliance import (
    ComplianceChecker,
    RegulationFramework,
    RuleSeverity,
)
from astroml.validation.model_validator import (
    ModelValidator,
    ValidationGateResult,
)
from astroml.validation.robustness import (
    ModelRobustnessEvaluator,
    PerturbationType,
)


@pytest.fixture
def synthetic_dataset():
    np.random.seed(42)
    X = np.random.randn(200, 5).astype(np.float32)
    # Binary target based on linear combination
    logits = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2]
    y = (logits >= 0).astype(np.float32)
    sensitive = np.random.choice(["group_A", "group_B"], size=200)
    return X, y, sensitive


@pytest.fixture
def dummy_linear_model():
    return {
        "weight": np.array([[1.5], [-2.0], [0.5], [0.0], [0.0]], dtype=np.float32),
        "bias": np.zeros(1, dtype=np.float32),
    }


class TestModelRobustnessEvaluator:
    def test_noise_perturbation(self, synthetic_dataset, dummy_linear_model):
        X, y, _ = synthetic_dataset
        evaluator = ModelRobustnessEvaluator()

        res_gauss = evaluator.evaluate_noise_perturbation(
            dummy_linear_model, X, y, noise_type="gaussian", max_allowed_drop=0.20
        )
        assert res_gauss.baseline_score > 0.70
        assert res_gauss.passed

        res_unif = evaluator.evaluate_noise_perturbation(
            dummy_linear_model, X, y, noise_type="uniform", max_allowed_drop=0.20
        )
        assert res_unif.passed

    def test_feature_dropout_and_fgsm(self, synthetic_dataset, dummy_linear_model):
        X, y, _ = synthetic_dataset
        evaluator = ModelRobustnessEvaluator()

        res_drop = evaluator.evaluate_feature_dropout(dummy_linear_model, X, y)
        assert res_drop.baseline_score > 0.70

        res_fgsm = evaluator.evaluate_adversarial_fgsm(dummy_linear_model, X, y, epsilon=0.02)
        assert res_fgsm.perturbation_type == PerturbationType.ADVERSARIAL_FGSM.value

    def test_comprehensive_suite(self, synthetic_dataset, dummy_linear_model):
        X, y, _ = synthetic_dataset
        evaluator = ModelRobustnessEvaluator()
        suite_res = evaluator.run_comprehensive_suite(dummy_linear_model, X, y)
        assert suite_res["robustness_score"] >= 70.0
        assert "gaussian_noise" in suite_res["results"]


class TestComplianceChecker:
    def test_eu_ai_act_and_gdpr_compliance(self):
        checker = ComplianceChecker()
        valid_meta = {
            "framework": "pytorch",
            "task_type": "binary_classification",
            "hyperparameters": {"lr": 0.001, "batch_size": 32},
            "input_schema": {"feature_1": "float32", "feature_2": "float32"},
            "output_schema": {"probability": "float32"},
            "lineage": {"dataset_id": "ds_stellar_q1", "commit_hash": "abcdef"},
            "dp_enabled": True,
        }
        robustness_rep = {"robustness_score": 85.0}

        report = checker.evaluate_compliance(
            model_name="stellar_compliance_model",
            version="1.0.0",
            metadata=valid_meta,
            validation_metrics={"accuracy": 0.92, "f1_score": 0.90},
            robustness_report=robustness_rep,
        )

        assert report.overall_compliant
        assert report.compliance_score >= 80.0
        assert report.mandatory_violations == 0
        assert report.certificate_id.startswith("CERT-AML-")

    def test_compliance_failure_on_missing_documentation(self):
        checker = ComplianceChecker()
        incomplete_meta = {"framework": "custom"}  # Missing lineage and schemas

        report = checker.evaluate_compliance(
            model_name="uncompliant_model",
            version="0.1.0",
            metadata=incomplete_meta,
        )
        assert not report.overall_compliant
        assert report.mandatory_violations > 0


class TestModelValidator:
    def test_validation_gate_pass(self, synthetic_dataset, dummy_linear_model):
        X, y, sensitive = synthetic_dataset
        validator = ModelValidator(min_accuracy=0.60, min_f1_score=0.60)

        metadata = {
            "framework": "pytorch",
            "hyperparameters": {"lr": 0.01},
            "input_schema": {"f": "float"},
            "output_schema": {"y": "int"},
            "lineage": {"dataset_id": "test_ds"},
        }

        gate_res = validator.validate_model(
            model=dummy_linear_model,
            model_name="stellar_detector",
            version="1.0.0",
            X_test=X,
            y_test=y,
            sensitive_features=sensitive,
            metadata=metadata,
        )

        assert gate_res.can_deploy
        assert gate_res.gate_decisions["performance"]
        assert gate_res.gate_decisions["robustness"]
        assert gate_res.overall_score > 60.0
        assert len(validator.get_validation_history()) == 1

    def test_regression_gate_failure(self, synthetic_dataset, dummy_linear_model):
        X, y, _ = synthetic_dataset
        validator = ModelValidator(max_regression_drop=0.02)

        # Inverted model has low accuracy (~0.0) compared to baseline (0.95)
        degraded_model = {
            "weight": -1.0 * dummy_linear_model["weight"],
            "bias": np.zeros(1, dtype=np.float32),
        }
        baseline = {"accuracy": 0.95, "f1_score": 0.95}
        gate_res = validator.validate_model(
            model=degraded_model,
            model_name="regressed_model",
            version="2.0.0",
            X_test=X,
            y_test=y,
            baseline_metrics=baseline,
        )

        assert not gate_res.can_deploy
        assert not gate_res.gate_decisions["regression"]
        assert any("Regression detected" in r for r in gate_res.blocking_reasons)


class TestValidationAPI:
    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.include_router(validation_router)
        return TestClient(app)

    def test_api_validation_gate_endpoint(self, client):
        payload = {
            "model_name": "api_model",
            "version": "1.0.0",
            "weights": {
                "weight": [[1.0], [2.0]],
                "bias": [0.0],
            },
            "X_test": [[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0], [-3.0, -4.0]],
            "y_test": [1.0, 1.0, 0.0, 0.0],
            "metadata": {
                "framework": "pytorch",
                "hyperparameters": {"lr": 0.01},
                "input_schema": {"x": "float"},
                "output_schema": {"y": "int"},
                "lineage": {"dataset_id": "test_ds"},
            },
        }

        res = client.post("/api/v1/validation/gate", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "can_deploy" in data
        assert "overall_score" in data

        # Test history endpoint
        res_hist = client.get("/api/v1/validation/history/api_model")
        assert res_hist.status_code == 200
        assert res_hist.json()["total"] >= 1
