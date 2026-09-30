"""Tests for metrics comparison tooling across model versions."""

from __future__ import annotations

import pytest

from astroml.utils.metrics_comparison import ComparisonReport, MetricsComparator, VersionMetrics


class TestVersionMetrics:
    def test_create_and_to_dict(self):
        metrics = VersionMetrics(
            version_id="v1", model_name="fraud-model",
            latency_ms=12.5, f1_score=0.95, drift_score=0.1, accuracy=0.92,
        )
        d = metrics.to_dict()
        assert d["version_id"] == "v1"
        assert d["f1_score"] == 0.95
        assert d["latency_ms"] == 12.5

    def test_to_dict_includes_additional_metrics(self):
        metrics = VersionMetrics(
            version_id="v1", model_name="fraud-model",
            latency_ms=12.5, f1_score=0.95, drift_score=0.1,
            precision=0.9, recall=0.88,
        )
        d = metrics.to_dict()
        assert d["precision"] == 0.9
        assert d["recall"] == 0.88


class TestMetricsComparator:
    def test_empty_comparison(self):
        comparator = MetricsComparator()
        report = comparator.compare()
        assert report.version_ids == []
        assert report.best is None

    def test_single_version(self):
        comparator = MetricsComparator()
        comparator.add_version(VersionMetrics(
            version_id="v1", model_name="model-a",
            latency_ms=12.5, f1_score=0.95, drift_score=0.1,
        ))
        report = comparator.compare()
        assert report.best == "v1"
        assert len(report.version_ids) == 1

    def test_comparison_ranks_by_f1(self):
        comparator = MetricsComparator()
        comparator.add_version(VersionMetrics(
            version_id="v1", model_name="model-a",
            latency_ms=12.5, f1_score=0.95, drift_score=0.1,
        ))
        comparator.add_version(VersionMetrics(
            version_id="v2", model_name="model-a",
            latency_ms=8.0, f1_score=0.87, drift_score=0.05,
        ))
        comparator.add_version(VersionMetrics(
            version_id="v3", model_name="model-a",
            latency_ms=20.0, f1_score=0.72, drift_score=0.3,
        ))

        report = comparator.compare()
        assert report.best == "v1"
        assert report.version_ids == ["v1", "v2", "v3"]

    def test_comparison_ranks_by_accuracy_as_tiebreaker(self):
        comparator = MetricsComparator()
        comparator.add_version(VersionMetrics(
            version_id="v1", model_name="model-a",
            latency_ms=12.5, f1_score=0.90, drift_score=0.1, accuracy=0.85,
        ))
        comparator.add_version(VersionMetrics(
            version_id="v2", model_name="model-a",
            latency_ms=8.0, f1_score=0.90, drift_score=0.05, accuracy=0.92,
        ))

        report = comparator.compare()
        assert report.best == "v2"  # Higher accuracy wins tie

    def test_comparison_ranks_by_latency_as_secondary_tiebreaker(self):
        comparator = MetricsComparator()
        comparator.add_version(VersionMetrics(
            version_id="v1", model_name="model-a",
            latency_ms=20.0, f1_score=0.90, drift_score=0.1, accuracy=0.90,
        ))
        comparator.add_version(VersionMetrics(
            version_id="v2", model_name="model-a",
            latency_ms=8.0, f1_score=0.90, drift_score=0.05, accuracy=0.90,
        ))

        report = comparator.compare()
        assert report.best == "v2"  # Lower latency wins tie

    def test_drift_warning(self):
        comparator = MetricsComparator()
        comparator.add_version(VersionMetrics(
            version_id="v1", model_name="model-a",
            latency_ms=12.5, f1_score=0.95, drift_score=0.8,
        ))
        report = comparator.compare()
        assert any("high drift" in r for r in report.recommendations)

    def test_low_f1_warning(self):
        comparator = MetricsComparator()
        comparator.add_version(VersionMetrics(
            version_id="v1", model_name="model-a",
            latency_ms=12.5, f1_score=0.5, drift_score=0.1,
        ))
        report = comparator.compare()
        assert any("low F1" in r for r in report.recommendations)

    def test_competitive_alternative(self):
        comparator = MetricsComparator()
        comparator.add_version(VersionMetrics(
            version_id="v1", model_name="model-a",
            latency_ms=20.0, f1_score=0.90, drift_score=0.1,
        ))
        comparator.add_version(VersionMetrics(
            version_id="v2", model_name="model-a",
            latency_ms=8.0, f1_score=0.88, drift_score=0.05,
        ))
        report = comparator.compare()
        assert any("competitive alternative" in r for r in report.recommendations)

    def test_remove_version(self):
        comparator = MetricsComparator()
        comparator.add_version(VersionMetrics(
            version_id="v1", model_name="model-a",
            latency_ms=12.5, f1_score=0.95, drift_score=0.1,
        ))
        comparator.add_version(VersionMetrics(
            version_id="v2", model_name="model-b",
            latency_ms=8.0, f1_score=0.87, drift_score=0.05,
        ))
        comparator.remove_version("v1")
        report = comparator.compare()
        assert report.best == "v2"
        assert report.version_ids == ["v2"]

    def test_comparison_report_to_dict(self):
        comparator = MetricsComparator()
        comparator.add_version(VersionMetrics(
            version_id="v1", model_name="model-a",
            latency_ms=12.5, f1_score=0.95, drift_score=0.1,
        ))
        report = comparator.compare()
        d = report.to_dict()
        assert d["best"] == "v1"
        assert len(d["all_metrics"]) == 1

    def test_all_versions_in_report(self):
        comparator = MetricsComparator()
        for i in range(5):
            comparator.add_version(VersionMetrics(
                version_id=f"v{i}", model_name="model-a",
                latency_ms=float(10 + i * 5), f1_score=0.9 - i * 0.05,
                drift_score=float(i) * 0.1,
            ))
        report = comparator.compare()
        assert len(report.version_ids) == 5
        assert report.best == "v0"


class TestMetricsComparatorPromotion:
    def test_recommendation_for_promotion(self):
        comparator = MetricsComparator()
        comparator.add_version(VersionMetrics(
            version_id="v1", model_name="model-a",
            latency_ms=12.5, f1_score=0.95, drift_score=0.1,
        ))
        report = comparator.compare()
        assert any("leading candidate" in r for r in report.recommendations)

    def test_all_versions_have_drift_warning(self):
        comparator = MetricsComparator()
        comparator.add_version(VersionMetrics(
            version_id="v1", model_name="model-a",
            latency_ms=12.5, f1_score=0.95, drift_score=0.9,
        ))
        report = comparator.compare()
        assert any("high drift" in r for r in report.recommendations)
