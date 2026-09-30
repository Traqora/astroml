"""Automated data quality reporting dashboard and document generator.

Formats data quality inspection results into structured dictionaries, JSON,
Markdown, and interactive HTML dashboards.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from astroml.validation.data_quality.alerts import QualityAlert
from astroml.validation.data_quality.checks import CheckResult, DataQualityReport

logger = logging.getLogger(__name__)


class DataQualityReporter:
    """Generates human-readable and machine-readable data quality reports."""

    def __init__(self, title: str = "AstroML Data Quality Report") -> None:
        self.title = title

    def generate_report(
        self,
        report: DataQualityReport,
        alerts: list[QualityAlert] | None = None,
    ) -> dict[str, Any]:
        """Compile a comprehensive dictionary report from validation outputs."""
        alerts_list = alerts or []

        # Group check results by dimension
        by_dimension: dict[str, list[dict[str, Any]]] = {}
        for check in report.check_results:
            dim = check.dimension.value if hasattr(check.dimension, "value") else str(check.dimension)
            by_dimension.setdefault(dim, []).append(
                {
                    "check_name": check.check_name,
                    "is_valid": check.is_valid,
                    "score": round(check.score * 100.0, 2),
                    "severity": check.severity.value if hasattr(check.severity, "value") else str(check.severity),
                    "field": check.field,
                    "message": check.message,
                    "details": check.details,
                }
            )

        passed_checks = sum(1 for c in report.check_results if c.is_valid)
        total_checks = len(report.check_results)

        return {
            "title": self.title,
            "batch_id": report.batch_id,
            "created_at": report.created_at,
            "total_records": report.total_records,
            "valid_records": report.valid_records,
            "quality_score": round(report.quality_score, 2),
            "dimension_scores": report.dimension_scores,
            "check_summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": total_checks - passed_checks,
                "pass_rate": round(passed_checks / total_checks * 100.0, 2) if total_checks else 100.0,
            },
            "dimensions": by_dimension,
            "active_alerts": [
                {
                    "alert_id": a.alert_id,
                    "rule_name": a.rule_name,
                    "metric_name": a.metric_name,
                    "current_value": a.current_value,
                    "threshold": a.threshold,
                    "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
                    "message": a.message,
                }
                for a in alerts_list
            ],
            "legacy_summary": report.summary,
        }

    def to_json(
        self,
        report: DataQualityReport,
        alerts: list[QualityAlert] | None = None,
        indent: int = 2,
    ) -> str:
        """Export report as formatted JSON."""
        data = self.generate_report(report, alerts)
        return json.dumps(data, indent=indent)

    def to_markdown(
        self,
        report: DataQualityReport,
        alerts: list[QualityAlert] | None = None,
    ) -> str:
        """Generate a Markdown formatted report."""
        rep = self.generate_report(report, alerts)
        score = rep["quality_score"]
        status_emoji = "🟢" if score >= 90 else ("🟡" if score >= 75 else "🔴")

        lines = [
            f"# {self.title}",
            f"**Generated:** {rep['created_at']} | **Batch ID:** `{rep['batch_id'] or 'N/A'}`",
            "",
            "## Executive Summary",
            f"- **Overall Quality Score:** {status_emoji} **{score:.1f}%**",
            f"- **Total Records Analyzed:** {rep['total_records']}",
            f"- **Valid Records:** {rep['valid_records']}",
            f"- **Checks Passed:** {rep['check_summary']['passed_checks']}/{rep['check_summary']['total_checks']} ({rep['check_summary']['pass_rate']}%)",
            "",
            "## Dimension Scores",
            "| Dimension | Score | Status |",
            "| :--- | :---: | :---: |",
        ]

        for dim, d_score in rep.get("dimension_scores", {}).items():
            if dim == "overall_score":
                continue
            dim_emoji = "✅" if d_score >= 90 else ("⚠️" if d_score >= 75 else "❌")
            lines.append(f"| {dim.capitalize()} | {d_score:.1f}% | {dim_emoji} |")

        if rep["active_alerts"]:
            lines.extend([
                "",
                "## ⚠️ Active Alerts",
                "| Severity | Rule | Message | Current Value | Threshold |",
                "| :--- | :--- | :--- | :---: | :---: |",
            ])
            for a in rep["active_alerts"]:
                lines.append(
                    f"| **{a['severity'].upper()}** | {a['rule_name']} | {a['message']} | {a['current_value']} | {a['threshold']} |"
                )

        lines.extend([
            "",
            "## Check Details",
        ])
        for dim, checks in rep.get("dimensions", {}).items():
            lines.append(f"### {dim.capitalize()}")
            for c in checks:
                check_status = "PASS" if c["is_valid"] else f"FAIL ({c['severity'].upper()})"
                lines.append(f"- **{c['check_name']}**: `{check_status}` - {c['message']}")

        return "\n".join(lines)

    def to_html(
        self,
        report: DataQualityReport,
        alerts: list[QualityAlert] | None = None,
    ) -> str:
        """Generate a standalone HTML dashboard report."""
        rep = self.generate_report(report, alerts)
        score = rep["quality_score"]
        bg_color = "#10b981" if score >= 90 else ("#f59e0b" if score >= 75 else "#ef4444")

        alerts_html = ""
        if rep["active_alerts"]:
            rows = "".join(
                f"<tr><td><span class='badge {a['severity']}'>{a['severity'].upper()}</span></td><td>{a['rule_name']}</td><td>{a['message']}</td><td>{a['current_value']}</td><td>{a['threshold']}</td></tr>"
                for a in rep["active_alerts"]
            )
            alerts_html = f"""
            <div class="card alert-section">
                <h2>⚠️ Triggered Quality Alerts</h2>
                <table>
                    <thead><tr><th>Severity</th><th>Rule</th><th>Message</th><th>Value</th><th>Threshold</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        dim_cards = "".join(
            f"""
            <div class="score-card">
                <h3>{dim.capitalize()}</h3>
                <div class="score-val">{val:.1f}%</div>
            </div>
            """
            for dim, val in rep.get("dimension_scores", {}).items()
            if dim != "overall_score"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{self.title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 16px; margin-bottom: 24px; }}
        .overall-score {{ background: {bg_color}; color: #fff; font-size: 32px; font-weight: bold; padding: 12px 24px; border-radius: 8px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .score-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; text-align: center; }}
        .score-val {{ font-size: 24px; font-weight: bold; color: #38bdf8; margin-top: 8px; }}
        .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #334155; font-size: 14px; }}
        th {{ background: #0f172a; color: #94a3b8; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .badge.critical {{ background: #ef4444; color: white; }}
        .badge.warning {{ background: #f59e0b; color: black; }}
        .badge.info {{ background: #3b82f6; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>{self.title}</h1>
                <p>Generated: {rep['created_at']} | Batch ID: {rep['batch_id'] or 'N/A'}</p>
            </div>
            <div class="overall-score">{score:.1f}%</div>
        </div>
        <div class="grid">
            {dim_cards}
        </div>
        {alerts_html}
    </div>
</body>
</html>"""

    def generate_trend_report(self, reports: list[DataQualityReport]) -> dict[str, Any]:
        """Analyze trend across a historical series of reports."""
        if not reports:
            return {"count": 0, "scores": [], "trend": "flat"}

        sorted_reps = sorted(reports, key=lambda r: r.created_at)
        scores = [r.quality_score for r in sorted_reps]
        first_score = scores[0]
        last_score = scores[-1]

        diff = last_score - first_score
        trend = "improving" if diff > 1.0 else ("degrading" if diff < -1.0 else "stable")

        return {
            "count": len(reports),
            "first_score": round(first_score, 2),
            "latest_score": round(last_score, 2),
            "score_change": round(diff, 2),
            "trend": trend,
            "history": [
                {
                    "batch_id": r.batch_id,
                    "created_at": r.created_at,
                    "score": round(r.quality_score, 2),
                    "dimension_scores": r.dimension_scores,
                }
                for r in sorted_reps
            ],
        }
