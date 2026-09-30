"""Tests for FairnessReport."""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest

from astroml.validation.fairness.bias_detector import BiasDetectionResult, BiasDetector
from astroml.validation.fairness.metrics import FairnessMetrics
from astroml.validation.fairness.report import FairnessReport


@pytest.fixture
def report() -> FairnessReport:
    return FairnessReport()


@pytest.fixture
def populated_report(fair_dataset) -> FairnessReport:
    y_true, y_pred, sensitive = fair_dataset
    metrics = FairnessMetrics()
    detector = BiasDetector()
    all_metrics = metrics.compute_all(y_true, y_pred, sensitive)
    bias_result = detector.detect_bias(y_true, y_pred, sensitive)
    report = FairnessReport()
    report.generate_report(fairness_metrics=all_metrics, bias_results=bias_result)
    return report


class TestFairnessReport:
    def test_generate_report(self, report, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        metrics = FairnessMetrics()
        detector = BiasDetector()
        all_metrics = metrics.compute_all(y_true, y_pred, sensitive)
        bias_result = detector.detect_bias(y_true, y_pred, sensitive)
        result = report.generate_report(fairness_metrics=all_metrics, bias_results=bias_result)
        assert isinstance(result, FairnessReport)
        assert result.report_data != {}

    def test_generate_report_minimal(self, report):
        result = report.generate_report()
        assert isinstance(result, FairnessReport)

    def test_to_dict(self, populated_report):
        data = populated_report.to_dict()
        assert "generated_at" in data
        assert "overall_scores" in data
        assert "per_attribute_breakdown" in data
        assert "recommendations" in data

    def test_to_json(self, populated_report):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            populated_report.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert "overall_scores" in data
            assert "generated_at" in data
        finally:
            import os

            os.unlink(path)

    def test_to_html(self, populated_report):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            populated_report.to_html(path)
            with open(path) as f:
                content = f.read()
            assert "<html" in content
            assert "Fairness Report" in content
        finally:
            import os

            os.unlink(path)

    def test_summary(self, populated_report):
        summary = populated_report.summary()
        assert isinstance(summary, str)
        assert "FAIRNESS REPORT SUMMARY" in summary
        assert "Overall bias detected" in summary

    def test_summary_empty_report(self, report):
        summary = report.summary()
        assert isinstance(summary, str)

    def test_compare_reports(self, populated_report, fair_dataset):
        report2 = FairnessReport()
        y_true, y_pred, sensitive = fair_dataset
        metrics = FairnessMetrics()
        all_metrics = metrics.compute_all(y_true, y_pred, sensitive)
        report2.generate_report(fairness_metrics=all_metrics)
        comparison = FairnessReport.compare_reports(populated_report, report2)
        assert "bias_detected_change" in comparison
        assert "severity_change" in comparison

    def test_compare_reports_empty(self):
        r1 = FairnessReport()
        r2 = FairnessReport()
        comparison = FairnessReport.compare_reports(r1, r2)
        assert isinstance(comparison, dict)

    def test_recommend_mitigation_no_bias(self):
        result = BiasDetectionResult(
            overall_bias_detected=False,
            per_attribute=[],
            severity=0.0,
        )
        recs = FairnessReport.recommend_mitigation(result)
        assert any("no mitigation required" in r for r in recs)

    def test_recommend_mitigation_with_bias(self):
        from astroml.validation.fairness.bias_detector import PerAttributeResult

        result = BiasDetectionResult(
            overall_bias_detected=True,
            per_attribute=[
                PerAttributeResult(
                    attribute_name="gender",
                    metrics={},
                    bias_detected=True,
                    severity=0.7,
                )
            ],
            severity=0.7,
        )
        recs = FairnessReport.recommend_mitigation(result)
        assert len(recs) > 0
        assert any("adversarial" in r for r in recs)

    def test_recommend_mitigation_moderate_bias(self):
        from astroml.validation.fairness.bias_detector import PerAttributeResult

        result = BiasDetectionResult(
            overall_bias_detected=True,
            per_attribute=[
                PerAttributeResult(
                    attribute_name="age",
                    metrics={},
                    bias_detected=True,
                    severity=0.4,
                )
            ],
            severity=0.4,
        )
        recs = FairnessReport.recommend_mitigation(result)
        assert any("post-processing" in r for r in recs)

    def test_recommend_mitigation_low_bias(self):
        from astroml.validation.fairness.bias_detector import PerAttributeResult

        result = BiasDetectionResult(
            overall_bias_detected=True,
            per_attribute=[
                PerAttributeResult(
                    attribute_name="race",
                    metrics={},
                    bias_detected=True,
                    severity=0.1,
                )
            ],
            severity=0.1,
        )
        recs = FairnessReport.recommend_mitigation(result)
        assert any("reweighing" in r for r in recs)

    def test_recommend_mitigation_no_data(self):
        recs = FairnessReport.recommend_mitigation(None)
        assert any("No bias data" in r for r in recs)

    def test_recommend_mitigation_multi_attribute(self):
        from astroml.validation.fairness.bias_detector import PerAttributeResult

        result = BiasDetectionResult(
            overall_bias_detected=True,
            per_attribute=[
                PerAttributeResult(
                    attribute_name="gender", metrics={}, bias_detected=True, severity=0.5
                ),
                PerAttributeResult(
                    attribute_name="race", metrics={}, bias_detected=True, severity=0.5
                ),
            ],
            severity=0.5,
        )
        recs = FairnessReport.recommend_mitigation(result)
        assert any("intersectional" in r for r in recs)

    def test_report_with_mitigation_results(self, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        metrics = FairnessMetrics()
        from astroml.validation.fairness.mitigation import MitigationResult

        all_metrics = metrics.compute_all(y_true, y_pred, sensitive)
        mit_result = MitigationResult(strategy_used="reweighing")
        report = FairnessReport()
        report.generate_report(fairness_metrics=all_metrics, mitigation_results=mit_result)
        data = report.to_dict()
        assert data.get("mitigation_results") is not None
        assert data["mitigation_results"]["strategy_used"] == "reweighing"

    def test_report_generated_at(self):
        r1 = FairnessReport(generated_at="2024-01-01T00:00:00Z")
        assert r1.generated_at == "2024-01-01T00:00:00Z"

    def test_report_default_generated_at(self):
        r = FairnessReport()
        assert r.generated_at != ""

    def test_overall_scores_no_bias(self, populated_report):
        data = populated_report.to_dict()
        scores = data["overall_scores"]
        assert "overall_bias_detected" in scores
        assert "overall_severity" in scores
        assert scores["metrics_total"] >= 4
