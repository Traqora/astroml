"""Automated continuous and batch data quality monitoring service.

Coordinates metric computation, historical trend evaluation, baseline comparisons,
alert triggering, and reporting.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone
from typing import Any

from astroml.validation.data_quality.alerts import AlertManager, QualityAlert
from astroml.validation.data_quality.checks import (
    AccuracyChecker,
    CheckResult,
    CompletenessChecker,
    ConsistencyChecker,
    DataQualityReport,
    DataQualityValidator,
    TimelinessChecker,
)
from astroml.validation.data_quality.reporter import DataQualityReporter

logger = logging.getLogger(__name__)


@dataclass
class MonitoredMetricPoint:
    """Individual recorded metric data point."""

    metric_name: str
    value: float
    dimension: str
    tags: dict[str, Any] = dc_field(default_factory=dict)
    timestamp: str = dc_field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class DataQualityMonitor:
    """Comprehensive data quality monitoring engine."""

    def __init__(
        self,
        alert_manager: AlertManager | None = None,
        reporter: DataQualityReporter | None = None,
        required_fields: list[str] | None = None,
        max_null_rate: float = 0.05,
        degradation_threshold: float = 10.0,
        baseline_score: float | None = None,
    ) -> None:
        self.alert_manager = alert_manager or AlertManager()
        self.reporter = reporter or DataQualityReporter()
        self.degradation_threshold = degradation_threshold
        self.baseline_score = baseline_score

        # Checkers
        self.completeness = CompletenessChecker(required_fields=required_fields, max_null_rate=max_null_rate)
        self.consistency = ConsistencyChecker()
        self.accuracy = AccuracyChecker()
        self.timeliness = TimelinessChecker()
        self.validator = DataQualityValidator()

        # State storage
        self._reports_history: list[DataQualityReport] = []
        self._metric_points: list[MonitoredMetricPoint] = []
        self._latest_report: DataQualityReport | None = None

    def set_baseline_score(self, score: float) -> None:
        """Set or update baseline quality score."""
        self.baseline_score = score

    def process_batch(
        self,
        records: list[dict[str, Any]],
        batch_id: str | None = None,
    ) -> DataQualityReport:
        """Run monitoring pipeline on an incoming batch of records."""
        effective_batch_id = batch_id or f"batch_{uuid.uuid4().hex[:8]}"

        # 1. Execute check suites
        c_res = self.completeness.check_records(records)
        cs_res = self.consistency.check_records(records)
        a_res = self.accuracy.check_records(records)
        t_res = self.timeliness.check_records(records)

        all_checks = c_res + cs_res + a_res + t_res

        def dim_avg(items: list[CheckResult]) -> float:
            return (sum(i.score for i in items) / len(items) * 100.0) if items else 100.0

        c_score = dim_avg(c_res)
        cs_score = dim_avg(cs_res)
        a_score = dim_avg(a_res)
        t_score = dim_avg(t_res)
        overall_score = (c_score + cs_score + a_score + t_score) / 4.0

        dim_scores = {
            "completeness": round(c_score, 2),
            "consistency": round(cs_score, 2),
            "accuracy": round(a_score, 2),
            "timeliness": round(t_score, 2),
            "overall_score": round(overall_score, 2),
        }

        # 2. Build report
        report = DataQualityReport(
            total_records=len(records),
            valid_records=sum(1 for c in all_checks if c.is_valid),
            check_results=all_checks,
            dimension_scores=dim_scores,
            batch_id=effective_batch_id,
        )

        # 3. Record metrics
        for dim_name, score_val in dim_scores.items():
            self.record_metric(
                metric_name=dim_name,
                value=score_val,
                dimension=dim_name if dim_name != "overall_score" else "overall",
                tags={"batch_id": effective_batch_id},
            )

        # 4. Evaluate alerts
        metrics_dict = {
            "completeness": c_score,
            "consistency": cs_score,
            "accuracy": a_score,
            "timeliness": t_score,
            "overall_score": overall_score,
        }
        self.alert_manager.evaluate_metrics(
            metrics=metrics_dict,
            batch_id=effective_batch_id,
            details={"dimension_scores": dim_scores},
        )

        # 5. Check degradation against baseline / previous run
        comparison_baseline = self.baseline_score
        if comparison_baseline is None and self._reports_history:
            comparison_baseline = self._reports_history[-1].quality_score

        if comparison_baseline is not None:
            self.alert_manager.check_quality_degradation(
                current_score=overall_score,
                baseline_score=comparison_baseline,
                max_drop_percentage=self.degradation_threshold,
                batch_id=effective_batch_id,
            )

        self._latest_report = report
        self._reports_history.append(report)

        return report

    def record_metric(
        self,
        metric_name: str,
        value: float,
        dimension: str = "general",
        tags: dict[str, Any] | None = None,
    ) -> None:
        """Manually or programmatically log a data quality metric point."""
        point = MonitoredMetricPoint(
            metric_name=metric_name,
            value=value,
            dimension=dimension,
            tags=tags or {},
        )
        self._metric_points.append(point)

    def get_metrics_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recently monitored metric points."""
        points = self._metric_points[-limit:]
        return [
            {
                "metric_name": p.metric_name,
                "value": p.value,
                "dimension": p.dimension,
                "tags": p.tags,
                "timestamp": p.timestamp,
            }
            for p in points
        ]

    def get_quality_trend(self) -> dict[str, Any]:
        """Generate trend analysis over historical batches."""
        return self.reporter.generate_trend_report(self._reports_history)

    def get_active_alerts(self) -> list[QualityAlert]:
        """Return currently active alerts."""
        return self.alert_manager.get_active_alerts()

    def get_latest_report(self) -> DataQualityReport | None:
        """Get latest generated report."""
        return self._latest_report

    def export_latest_report(self, format: str = "json") -> str:
        """Export latest report in specified format ('json', 'markdown', 'html')."""
        if not self._latest_report:
            return ""
        alerts = self.get_active_alerts()
        if format.lower() == "json":
            return self.reporter.to_json(self._latest_report, alerts)
        elif format.lower() in ("markdown", "md"):
            return self.reporter.to_markdown(self._latest_report, alerts)
        elif format.lower() == "html":
            return self.reporter.to_html(self._latest_report, alerts)
        else:
            raise ValueError(f"Unsupported report format: {format}")
