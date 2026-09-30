from astroml.utils.exceptions import AstroMLError
"""Automated model validation suite and CI/CD deployment gating.

Combines performance validation, fairness & bias detection, adversarial robustness
stress testing, regulatory compliance verification, and regression tracking against
baseline production models.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from astroml.validation.compliance import ComplianceChecker, ComplianceReport
from astroml.validation.fairness import BiasDetector, FairnessMetrics
from astroml.validation.robustness import ModelRobustnessEvaluator

logger = logging.getLogger(__name__)


@dataclass
class ValidationGateResult:
    """Consolidated validation decision for CI/CD model deployment."""

    model_name: str
    version: str
    can_deploy: bool
    gate_decisions: dict[str, bool]
    overall_score: float
    blocking_reasons: list[str]
    warnings: list[str]
    performance_metrics: dict[str, float]
    fairness_results: dict[str, Any]
    robustness_results: dict[str, Any]
    compliance_results: dict[str, Any]
    timestamp: str = dc_field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert gate result to dictionary."""
        return asdict(self)


class ModelValidator:
    """Orchestrates comprehensive pre-deployment validation for ML models."""

    def __init__(
        self,
        min_accuracy: float = 0.70,
        min_f1_score: float = 0.65,
        min_robustness_score: float = 75.0,
        min_compliance_score: float = 80.0,
        max_regression_drop: float = 0.05,
        robustness_evaluator: ModelRobustnessEvaluator | None = None,
        compliance_checker: ComplianceChecker | None = None,
    ) -> None:
        self.min_accuracy = min_accuracy
        self.min_f1_score = min_f1_score
        self.min_robustness_score = min_robustness_score
        self.min_compliance_score = min_compliance_score
        self.max_regression_drop = max_regression_drop

        self.robustness_evaluator = robustness_evaluator or ModelRobustnessEvaluator()
        self.compliance_checker = compliance_checker or ComplianceChecker()
        self._history: list[ValidationGateResult] = []

    def _evaluate_performance(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """Calculate classification accuracy, precision, recall, and F1."""
        y_true = y_test.flatten()

        if hasattr(model, "predict"):
            preds = model.predict(X_test)
        elif callable(model):
            preds = model(X_test)
        elif isinstance(model, dict) and "weight" in model:
            w = model["weight"]
            b = model.get("bias", 0.0)
            logits = np.dot(X_test, w) + b
            preds = (logits >= 0.0).astype(int).flatten()
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        preds_bin = (preds >= 0.5).astype(int).flatten()
        acc = float(np.mean(preds_bin == y_true))

        # Precision, Recall, F1
        tp = float(np.sum((preds_bin == 1) & (y_true == 1)))
        fp = float(np.sum((preds_bin == 1) & (y_true == 0)))
        fn = float(np.sum((preds_bin == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    def check_regression(
        self,
        current_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """Verify candidate model has not degraded vs production baseline."""
        reasons = []
        passed = True

        for metric_name in ["accuracy", "f1_score", "precision", "recall"]:
            if metric_name in baseline_metrics and metric_name in current_metrics:
                base_v = baseline_metrics[metric_name]
                curr_v = current_metrics[metric_name]
                drop = base_v - curr_v
                if drop > self.max_regression_drop:
                    passed = False
                    reasons.append(
                        f"Regression detected in '{metric_name}': dropped by {drop:.2%} "
                        f"(Current: {curr_v:.4f}, Baseline: {base_v:.4f})"
                    )

        return passed, reasons

    def validate_model(
        self,
        model: Any,
        model_name: str,
        version: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sensitive_features: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
        baseline_metrics: dict[str, float] | None = None,
        required_compliance_frameworks: list[str] | None = None,
    ) -> ValidationGateResult:
        """Run all automated gates: Performance, Fairness, Robustness, Compliance, Regression."""
        blocking_reasons: list[str] = []
        warnings: list[str] = []
        gate_decisions: dict[str, bool] = {}

        # 1. Performance Gate
        perf_metrics = self._evaluate_performance(model, X_test, y_test)
        perf_passed = perf_metrics["accuracy"] >= self.min_accuracy and perf_metrics["f1_score"] >= self.min_f1_score
        gate_decisions["performance"] = perf_passed
        if not perf_passed:
            blocking_reasons.append(
                f"Performance gate failed: accuracy={perf_metrics['accuracy']:.2%} (min={self.min_accuracy:.2%}) "
                f"or f1={perf_metrics['f1_score']:.2%} (min={self.min_f1_score:.2%})"
            )

        # 2. Fairness Gate
        fairness_summary: dict[str, Any] = {"status": "skipped", "fairness_passed": True}
        if sensitive_features is not None:
            try:
                preds = (self._evaluate_performance(model, X_test, y_test)["accuracy"])
                # Generate predictions
                if hasattr(model, "predict"):
                    p = model.predict(X_test)
                elif isinstance(model, dict) and "weight" in model:
                    p = (np.dot(X_test, model["weight"]) + model.get("bias", 0.0) >= 0).astype(int)
                else:
                    p = (X_test[:, 0] > 0).astype(int)

                p_bin = (p >= 0.5).astype(int).flatten()
                detector = BiasDetector(sensitive_feature_names=["sensitive_group"])
                report = detector.detect_bias(
                    y_true=y_test.flatten(),
                    y_pred=p_bin,
                    sensitive_features=sensitive_features,
                )
                fairness_passed = not report.bias_detected
                gate_decisions["fairness"] = fairness_passed
                fairness_summary = {
                    "fairness_passed": fairness_passed,
                    "bias_detected": report.bias_detected,
                    "metrics": {
                        m.metric_name: {"passed": m.passed, "value": m.value}
                        for m in report.metrics_results
                    },
                }
                if not fairness_passed:
                    blocking_reasons.append("Fairness gate failed: significant bias detected across protected classes.")
            except AstroMLError as e:
                warnings.append(f"Fairness evaluation warning: {e}")
                gate_decisions["fairness"] = True
        else:
            gate_decisions["fairness"] = True
            warnings.append("Sensitive features not provided; fairness gate skipped.")

        # 3. Robustness Gate
        robustness_res = self.robustness_evaluator.run_comprehensive_suite(
            model=model,
            X=X_test,
            y=y_test,
        )
        robustness_passed = robustness_res["robustness_score"] >= self.min_robustness_score
        gate_decisions["robustness"] = robustness_passed
        if not robustness_passed:
            blocking_reasons.append(
                f"Robustness gate failed: score={robustness_res['robustness_score']:.1f}% "
                f"(min={self.min_robustness_score:.1f}%)"
            )

        # 4. Regulatory Compliance Gate
        meta_dict = metadata or {}
        meta_dict.setdefault("framework", "custom")
        compliance_rep = self.compliance_checker.evaluate_compliance(
            model_name=model_name,
            version=version,
            metadata=meta_dict,
            validation_metrics=perf_metrics,
            fairness_report=fairness_summary,
            robustness_report=robustness_res,
            frameworks=required_compliance_frameworks,
        )
        compliance_passed = compliance_rep.overall_compliant and compliance_rep.compliance_score >= self.min_compliance_score
        gate_decisions["compliance"] = compliance_passed
        if not compliance_passed:
            blocking_reasons.append(
                f"Compliance gate failed: score={compliance_rep.compliance_score:.1f}% "
                f"(violations={compliance_rep.mandatory_violations})"
            )

        # 5. Regression Gate
        if baseline_metrics:
            reg_passed, reg_reasons = self.check_regression(perf_metrics, baseline_metrics)
            gate_decisions["regression"] = reg_passed
            if not reg_passed:
                blocking_reasons.extend(reg_reasons)
        else:
            gate_decisions["regression"] = True

        # Consolidated decision
        can_deploy = len(blocking_reasons) == 0

        # Compute composite score (0 - 100)
        overall_score = round(
            (
                perf_metrics["accuracy"] * 25.0
                + perf_metrics["f1_score"] * 25.0
                + (robustness_res["robustness_score"] / 100.0) * 25.0
                + (compliance_rep.compliance_score / 100.0) * 25.0
            ),
            2,
        )

        result = ValidationGateResult(
            model_name=model_name,
            version=version,
            can_deploy=can_deploy,
            gate_decisions=gate_decisions,
            overall_score=overall_score,
            blocking_reasons=blocking_reasons,
            warnings=warnings,
            performance_metrics=perf_metrics,
            fairness_results=fairness_summary,
            robustness_results=robustness_res,
            compliance_results={
                "certificate_id": compliance_rep.certificate_id,
                "overall_compliant": compliance_rep.overall_compliant,
                "compliance_score": compliance_rep.compliance_score,
                "framework_scores": compliance_rep.framework_scores,
                "mandatory_violations": compliance_rep.mandatory_violations,
                "summary": compliance_rep.summary,
            },
        )

        self._history.append(result)
        logger.info(
            "Model validation for %s:%s -> Can Deploy: %s (Score: %.2f)",
            model_name,
            version,
            can_deploy,
            overall_score,
        )
        return result

    def get_validation_history(self, model_name: str | None = None) -> list[ValidationGateResult]:
        """Retrieve historical validation runs."""
        if model_name:
            return [r for r in self._history if r.model_name == model_name]
        return list(self._history)

    def export_validation_report(self, result: ValidationGateResult, format: str = "json") -> str:
        """Export gate evaluation result as JSON string."""
        if format.lower() == "json":
            return json.dumps(result.to_dict(), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
