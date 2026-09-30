"""Tests for the data profiling toolkit and its API router."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers.data_profiling import router as profiling_router
from astroml.preprocessing.profiling.data_profiler import DataProfiler
from astroml.preprocessing.profiling.insights import Insight, InsightGenerator
from astroml.preprocessing.profiling.report_generator import ReportGenerator
from astroml.preprocessing.profiling.visualizations import ProfilingVisualizer

app = FastAPI()
app.include_router(profiling_router)
client = TestClient(app)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a DataFrame with assorted data quality issues."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "numeric_ok": rng.normal(size=100),
            # Deterministic: exactly the first 20 rows are missing.
            "numeric_missing": pd.Series(rng.normal(size=100)).mask(np.arange(100) < 20),
            "constant": np.full(100, 7),
            "categorical": rng.choice(["a", "b", "c"], size=100),
            "outliers": pd.Series(np.r_[rng.normal(size=98), [100.0, -100.0]]),
        }
    )


@pytest.fixture
def profile(sample_df: pd.DataFrame):
    """Compute a profile for the sample DataFrame."""
    return DataProfiler().profile(sample_df)


class TestDataProfiler:
    def test_profile_basic(self, sample_df: pd.DataFrame):
        result = DataProfiler().profile(sample_df)
        assert result.row_count == 100
        assert result.column_count == 5
        assert result.missing_total == 20
        assert 0 <= result.quality_score <= 100

    def test_column_profiles(self, profile):
        assert set(profile.columns.keys()) == {
            "numeric_ok",
            "numeric_missing",
            "constant",
            "categorical",
            "outliers",
        }
        numeric = profile.columns["numeric_ok"]
        assert numeric.dtype == "numeric"
        assert numeric.count == 100
        assert numeric.missing_rate == 0.0
        assert numeric.mean is not None
        assert numeric.min is not None and numeric.max is not None

    def test_missing_column(self, profile):
        missing = profile.columns["numeric_missing"]
        assert missing.missing_count == 20
        assert missing.missing_rate == pytest.approx(0.2)

    def test_constant_column(self, profile):
        constant = profile.columns["constant"]
        assert constant.constant is True
        assert constant.unique_count == 1

    def test_categorical_dtype(self, profile):
        categorical = profile.columns["categorical"]
        assert categorical.dtype == "categorical"
        assert categorical.unique_count == 3

    def test_outlier_detection(self, profile):
        outliers = profile.columns["outliers"]
        assert outliers.outlier_count >= 2
        assert outliers.max == 100.0
        assert outliers.min == -100.0

    def test_duplicate_rows(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
        result = DataProfiler().profile(df)
        assert result.duplicate_rows == 1

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            DataProfiler().profile(pd.DataFrame())

    def test_quality_score_perfect(self):
        df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) * 2})
        result = DataProfiler().profile(df)
        assert result.quality_score == pytest.approx(100.0)

    def test_quality_score_penalties(self, sample_df: pd.DataFrame):
        result = DataProfiler().profile(sample_df)
        assert result.quality_score < 100.0

    def test_profile_column_series_name(self, sample_df: pd.DataFrame):
        col = DataProfiler().profile_column(sample_df["numeric_ok"])
        assert col.name == "numeric_ok"

    def test_to_dict(self, profile):
        data = profile.columns["numeric_ok"].to_dict()
        assert data["name"] == "numeric_ok"
        assert data["dtype"] == "numeric"
        assert "mean" in data

    def test_detect_dtype_text(self):
        df = pd.DataFrame({"text": ["x" * 100, "y" * 100]})
        result = DataProfiler().profile(df)
        assert result.columns["text"].dtype == "text"

    def test_detect_dtype_datetime(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"])})
        result = DataProfiler().profile(df)
        assert result.columns["date"].dtype == "datetime"

    def test_detect_dtype_boolean(self):
        df = pd.DataFrame({"flag": [True, False, True]})
        result = DataProfiler().profile(df)
        assert result.columns["flag"].dtype == "boolean"

    def test_detect_dtype_unknown_all_missing(self):
        df = pd.DataFrame({"empty": [None, None]})
        result = DataProfiler().profile(df)
        assert result.columns["empty"].dtype == "unknown"


class TestInsightGenerator:
    def test_generate_column_insights(self, profile):
        insights = InsightGenerator().generate(profile)
        types = {insight.type for insight in insights}
        assert "high_missing_rate" in types
        assert "constant_column" in types
        assert "outliers" in types

    def test_no_outlier_insight_when_clean(self):
        df = pd.DataFrame({"a": np.arange(50, dtype=float)})
        profile = DataProfiler().profile(df)
        insights = InsightGenerator().generate(profile)
        assert not any(i.type == "outliers" for i in insights)

    def test_high_cardinality_insight(self):
        df = pd.DataFrame({"id": [f"x{i}" for i in range(50)]})
        profile = DataProfiler().profile(df)
        insights = InsightGenerator().generate(profile)
        assert any(i.type == "high_cardinality" for i in insights)

    def test_duplicate_rows_insight(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        profile = DataProfiler().profile(df)
        insights = InsightGenerator().generate(profile)
        assert any(i.type == "duplicate_rows" for i in insights)

    def test_missing_values_insight(self, profile):
        insights = InsightGenerator().generate(profile)
        assert any(i.type == "missing_values" for i in insights)

    def test_skewed_distribution_insight(self):
        rng = np.random.default_rng(0)
        skewed = rng.exponential(scale=1.0, size=1000)
        df = pd.DataFrame({"skewed": skewed})
        profile = DataProfiler().profile(df)
        insights = InsightGenerator().generate(profile)
        assert any(i.type == "skewed_distribution" for i in insights)

    def test_heavy_tails_insight(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"heavy": rng.standard_t(df=2.0, size=1000)})
        profile = DataProfiler().profile(df)
        insights = InsightGenerator().generate(profile)
        assert any(i.type == "heavy_tails" for i in insights)

    def test_summarize(self, profile):
        insights = InsightGenerator().generate(profile)
        summary = InsightGenerator().summarize(insights)
        assert summary["critical"] >= 0
        assert sum(summary.values()) == len(insights)

    def test_insight_to_dict(self):
        insight = Insight(type="test", severity="info", message="hello", column="a")
        data = insight.to_dict()
        assert data["type"] == "test"
        assert data["column"] == "a"
        assert data["details"] == {}

    def test_high_missing_critical_severity(self):
        df = pd.DataFrame({"a": [1, None, None, None, None]})
        profile = DataProfiler().profile(df)
        insights = InsightGenerator().generate(profile)
        missing = next(i for i in insights if i.type == "high_missing_rate")
        assert missing.severity == "critical"


class TestProfilingVisualizer:
    def test_distribution_plot(self, sample_df: pd.DataFrame, tmp_path: Path):
        fig = ProfilingVisualizer().distribution_plot(
            sample_df["numeric_ok"], save_path=tmp_path / "dist.png"
        )
        assert (tmp_path / "dist.png").exists()
        plt.close(fig)

    def test_correlation_heatmap(self, sample_df: pd.DataFrame, tmp_path: Path):
        fig = ProfilingVisualizer().correlation_heatmap(sample_df)
        assert fig is not None
        plt.close(fig)

    def test_correlation_heatmap_no_numeric(self, tmp_path: Path):
        df = pd.DataFrame({"a": ["x", "y"]})
        fig = ProfilingVisualizer().correlation_heatmap(df)
        plt.close(fig)

    def test_missing_value_plot(self, sample_df: pd.DataFrame, tmp_path: Path):
        fig = ProfilingVisualizer().missing_value_plot(sample_df)
        plt.close(fig)

    def test_missing_value_plot_none_missing(self, tmp_path: Path):
        df = pd.DataFrame({"a": [1, 2]})
        fig = ProfilingVisualizer().missing_value_plot(df)
        plt.close(fig)

    def test_outlier_boxplot(self, sample_df: pd.DataFrame, tmp_path: Path):
        fig = ProfilingVisualizer().outlier_boxplot(
            sample_df, columns=["numeric_ok", "outliers"], save_path=tmp_path / "box.png"
        )
        assert (tmp_path / "box.png").exists()
        plt.close(fig)

    def test_outlier_boxplot_no_numeric(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        fig = ProfilingVisualizer().outlier_boxplot(df)
        plt.close(fig)


class TestReportGenerator:
    def test_generate_markdown(self, profile, tmp_path: Path):
        insights = InsightGenerator().generate(profile)
        report = ReportGenerator().generate(profile, insights, fmt="markdown")
        assert report.startswith("# Data Profiling Report")
        assert "numeric_ok" in report
        assert "Quality score" in report
        assert "Insights" in report

    def test_generate_html(self, profile, tmp_path: Path):
        insights = InsightGenerator().generate(profile)
        report = ReportGenerator().generate(profile, insights, fmt="html")
        assert "<html" in report
        assert "numeric_ok" in report
        assert "Quality score" in report

    def test_generate_json(self, profile):
        insights = InsightGenerator().generate(profile)
        report = ReportGenerator().generate(profile, insights, fmt="json")
        payload = json.loads(report)
        assert payload["row_count"] == profile.row_count
        assert "numeric_ok" in payload["columns"]
        assert isinstance(payload["insights"], list)

    def test_generate_pdf(self, profile, tmp_path: Path):
        insights = InsightGenerator().generate(profile)
        path = ReportGenerator().generate(
            profile, insights, fmt="pdf", output_path=tmp_path / "report.pdf"
        )
        assert Path(path).exists()
        assert Path(path).read_bytes().startswith(b"%PDF")

    def test_write_to_disk(self, profile, tmp_path: Path):
        insights = InsightGenerator().generate(profile)
        target = tmp_path / "nested" / "report.md"
        ReportGenerator().generate(profile, insights, fmt="markdown", output_path=target)
        assert target.exists()
        assert "Data Profiling Report" in target.read_text()

    def test_unsupported_format(self, profile):
        with pytest.raises(ValueError, match="Unsupported"):
            ReportGenerator().generate(profile, [], fmt="docx")

    def test_markdown_empty_insights(self, profile):
        report = ReportGenerator().generate_markdown(profile, [])
        assert "No insights generated." in report

    def test_html_empty_insights(self, profile):
        report = ReportGenerator().generate_html(profile, [])
        assert "No insights generated." in report


class TestProfilingAPI:
    def test_analyze_columns(self):
        payload = {
            "columns": {
                "a": [1.0, 2.0, 3.0, None],
                "b": ["x", "y", "x", "z"],
            }
        }
        response = client.post("/api/v1/profiling/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["row_count"] == 4
        assert "a" in data["columns"]
        assert data["columns"]["a"]["missing_count"] == 1
        assert "insights" in data
        assert "insight_summary" in data
        assert data["quality_score"] >= 0

    def test_analyze_rows(self):
        payload = {"data": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]}
        response = client.post("/api/v1/profiling/analyze", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["row_count"] == 2

    def test_analyze_no_data(self):
        response = client.post("/api/v1/profiling/analyze", json={})
        assert response.status_code == 400

    def test_analyze_empty_columns(self):
        response = client.post("/api/v1/profiling/analyze", json={"columns": {"a": []}})
        assert response.status_code == 400

    def test_report_markdown(self):
        payload = {
            "columns": {"a": [1.0, 2.0, None]},
            "format": "markdown",
        }
        response = client.post("/api/v1/profiling/report", json=payload)
        assert response.status_code == 200
        assert "# Data Profiling Report" in response.json()["data"]["report"]

    def test_report_json(self):
        payload = {"columns": {"a": [1.0, 2.0]}, "format": "json"}
        response = client.post("/api/v1/profiling/report", json=payload)
        assert response.status_code == 200
        report = json.loads(response.json()["data"]["report"])
        assert report["row_count"] == 2

    def test_report_unsupported_format(self):
        payload = {"columns": {"a": [1.0]}, "format": "docx"}
        response = client.post("/api/v1/profiling/report", json=payload)
        assert response.status_code == 400
