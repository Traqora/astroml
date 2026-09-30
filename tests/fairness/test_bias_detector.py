"""Tests for BiasDetector."""

from __future__ import annotations

import numpy as np
import pytest

from astroml.validation.fairness.bias_detector import (
    BiasDetectionResult,
    BiasDetector,
    FeatureBiasResult,
    IntersectionalResult,
    PerAttributeResult,
)


@pytest.fixture
def detector() -> BiasDetector:
    return BiasDetector()


class TestBiasDetector:
    def test_detect_bias_single_attribute(self, detector, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        result = detector.detect_bias(y_true, y_pred, sensitive)
        assert isinstance(result, BiasDetectionResult)
        assert isinstance(result.overall_bias_detected, bool)
        assert isinstance(result.severity, float)
        assert len(result.per_attribute) == 1

    def test_detect_bias_multi_attribute(self, detector, multi_attribute_dataset):
        y_true, y_pred, sensitive = multi_attribute_dataset
        result = detector.detect_bias(y_true, y_pred, sensitive, attributes=["gender", "age"])
        assert len(result.per_attribute) == 2
        for attr in result.per_attribute:
            assert isinstance(attr, PerAttributeResult)
            assert attr.attribute_name in ("gender", "age")

    def test_detect_bias_custom_names(self, detector):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
        result = detector.detect_bias(y_true, y_pred, sensitive, attributes=["race", "gender"])
        names = [a.attribute_name for a in result.per_attribute]
        assert "race" in names
        assert "gender" in names

    def test_detect_bias_no_attributes_default_names(self, detector):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array([[0], [0], [1], [1]])
        result = detector.detect_bias(y_true, y_pred, sensitive)
        assert result.per_attribute[0].attribute_name == "attribute_0"

    def test_detect_bias_auto_1d(self, detector, biased_dataset):
        y_true, y_pred, sensitive = biased_dataset
        result = detector.detect_bias(y_true, y_pred, sensitive)
        assert len(result.per_attribute) == 1

    def test_detect_bias_mismatched_names_raises(self, detector):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
        with pytest.raises(ValueError, match="Number of attribute names"):
            detector.detect_bias(y_true, y_pred, sensitive, attributes=["only_one"])

    def test_intersectional_analysis(self, detector, multi_attribute_dataset):
        y_true, y_pred, sensitive = multi_attribute_dataset
        results = detector.intersectional_analysis(y_true, y_pred, sensitive)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, IntersectionalResult)
            assert isinstance(r.groups, tuple)
            assert isinstance(r.bias_detected, bool)
            assert r.sample_size >= 2

    def test_intersectional_analysis_custom_groups(self, detector):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        sensitive = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [1, 1]])
        results = detector.intersectional_analysis(
            y_true, y_pred, sensitive, intersection_groups=[[0, 1]]
        )
        assert len(results) > 0

    def test_intersectional_analysis_1d_input(self, detector):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array([0, 0, 1, 1])
        results = detector.intersectional_analysis(y_true, y_pred, sensitive)
        assert len(results) == 0

    def test_feature_bias_analysis(self, detector):
        X = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]])
        y = np.array([0, 1, 0, 1])
        sensitive = np.array([0, 0, 1, 1])
        results = detector.feature_bias_analysis(X, y, sensitive)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, FeatureBiasResult)
            assert r.feature_name.startswith("feature_")
            assert len(r.group_means) >= 2

    def test_distribution_alignment(self, detector):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        sensitive = np.array([0, 0, 1, 1])
        result = detector.distribution_alignment(X, sensitive)
        assert "feature_stats" in result
        assert "overall_alignment_score" in result
        assert 0 <= result["overall_alignment_score"] <= 1

    def test_report_bias(self, detector, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        bias_result = detector.detect_bias(y_true, y_pred, sensitive)
        report = detector.report_bias(bias_result)
        assert "overall_bias_detected" in report
        assert "severity" in report
        assert "per_attribute_breakdown" in report
        assert "recommendations" in report

    def test_report_bias_with_intersectional(self, detector, multi_attribute_dataset):
        y_true, y_pred, sensitive = multi_attribute_dataset
        bias_result = detector.detect_bias(y_true, y_pred, sensitive)
        bias_result.intersectional_results = detector.intersectional_analysis(
            y_true, y_pred, sensitive
        )
        report = detector.report_bias(bias_result)
        assert "intersectional_findings" in report

    def test_report_recommendations_no_bias(self, detector, fair_dataset):
        y_true, y_pred, sensitive = fair_dataset
        bias_result = detector.detect_bias(y_true, y_pred, sensitive)
        report = detector.report_bias(bias_result)
        if not bias_result.overall_bias_detected:
            assert any("No significant bias" in r for r in report["recommendations"])

    def test_report_recommendations_with_bias(self, detector, biased_dataset_unequal):
        y_true, y_pred, sensitive = biased_dataset_unequal
        bias_result = detector.detect_bias(y_true, y_pred, sensitive)
        report = detector.report_bias(bias_result)
        if bias_result.overall_bias_detected:
            assert any("Bias detected" in r for r in report["recommendations"])

    def test_per_attribute_result_severity(self, detector, biased_dataset):
        y_true, y_pred, sensitive = biased_dataset
        result = detector.detect_bias(y_true, y_pred, sensitive)
        for attr in result.per_attribute:
            assert 0 <= attr.severity <= 1

    def test_empty_per_attribute_no_severity(self, detector):
        from astroml.validation.fairness.bias_detector import BiasDetectionResult

        result = BiasDetectionResult(overall_bias_detected=False, per_attribute=[], severity=0.0)
        assert result.severity == 0.0
