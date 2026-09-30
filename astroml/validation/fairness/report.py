"""Fairness report generation and comparison.

Provides the FairnessReport class for generating comprehensive fairness
evaluation reports in dictionary, JSON, and HTML formats.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from astroml.validation.fairness.bias_detector import (
    BiasDetectionResult,
    BiasDetector,
)
from astroml.validation.fairness.metrics import FairnessMetricResult, FairnessMetrics
from astroml.validation.fairness.mitigation import BiasMitigation, MitigationResult

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fairness Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; color: #333; }}
h1, h2, h3 {{ color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #f5f5f5; }}
.pass {{ color: #2e7d32; font-weight: bold; }}
.fail {{ color: #c62828; font-weight: bold; }}
.severity {{ display: inline-block; padding: 2px 8px; border-radius: 4px; }}
.severity-low {{ background: #c8e6c9; }}
.severity-medium {{ background: #fff3e0; }}
.severity-high {{ background: #ffcdd2; }}
</style>
</head>
<body>
<h1>Fairness Report</h1>
<p>Generated: {generated_at}</p>
<h2>Overall Assessment</h2>
<p>Bias Detected: <span class="{overall_class}">{overall_bias}</span></p>
<p>Overall Severity: <span class="severity severity-{severity_level}">{overall_severity:.2f}</span></p>
<h2>Per-Attribute Breakdown</h2>
{table_html}
<h2>Recommendations</h2>
<ul>
{recommendations_html}
</ul>
</body>
</html>"""


@dataclass
class FairnessReport:
    """Comprehensive fairness evaluation report.

    Aggregates results from fairness metrics computation, bias detection,
    and mitigation evaluation into a structured report.

    Attributes:
        generated_at: Timestamp of report generation.
        fairness_metrics: Dictionary of FairnessMetricResult objects.
        bias_results: Optional BiasDetectionResult.
        mitigation_results: Optional MitigationResult.
        report_data: Raw report data dictionary.
    """

    generated_at: str = ""
    fairness_metrics: dict[str, FairnessMetricResult] = field(default_factory=dict)
    bias_results: BiasDetectionResult | None = None
    mitigation_results: MitigationResult | None = None
    report_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def generate_report(
        self,
        fairness_metrics: dict[str, FairnessMetricResult] | None = None,
        bias_results: BiasDetectionResult | None = None,
        mitigation_results: MitigationResult | None = None,
    ) -> FairnessReport:
        """Generate a comprehensive fairness report.

        Args:
            fairness_metrics: Results from FairnessMetrics.compute_all().
            bias_results: Results from BiasDetector.detect_bias().
            mitigation_results: Results from BiasMitigation.evaluate_mitigation().

        Returns:
            Self with populated report data.
        """
        if fairness_metrics is not None:
            self.fairness_metrics = fairness_metrics
        if bias_results is not None:
            self.bias_results = bias_results
        if mitigation_results is not None:
            self.mitigation_results = mitigation_results

        self.report_data = self._build_report_data()
        return self

    def _build_report_data(self) -> dict[str, Any]:
        """Build the raw report data dictionary.

        Returns:
            Structured report data.
        """
        data: dict[str, Any] = {
            "generated_at": self.generated_at,
            "overall_scores": self._compute_overall_scores(),
            "per_attribute_breakdown": [],
            "intersectional_analysis": [],
            "mitigation_results": None,
            "recommendations": [],
        }

        if self.bias_results is not None:
            detector = BiasDetector()
            report = detector.report_bias(self.bias_results)
            data["per_attribute_breakdown"] = report.get("per_attribute_breakdown", [])
            data["intersectional_analysis"] = report.get("intersectional_findings", [])
            data["recommendations"] = report.get("recommendations", [])

        if self.mitigation_results is not None:
            data["mitigation_results"] = {
                "strategy_used": self.mitigation_results.strategy_used,
                "improvement": self.mitigation_results.improvement,
                "before_metrics": self._serialize_metrics(self.mitigation_results.before_metrics),
                "after_metrics": self._serialize_metrics(self.mitigation_results.after_metrics),
            }

        if not data["recommendations"]:
            data["recommendations"] = self.recommend_mitigation(self.bias_results)

        return data

    def _compute_overall_scores(self) -> dict[str, Any]:
        """Compute overall fairness scores.

        Returns:
            Dictionary of overall scores.
        """
        scores: dict[str, Any] = {
            "overall_bias_detected": False,
            "overall_severity": 0.0,
            "metrics_passed": 0,
            "metrics_total": 0,
        }

        if self.bias_results is not None:
            scores["overall_bias_detected"] = self.bias_results.overall_bias_detected
            scores["overall_severity"] = self.bias_results.severity

        for metric in self.fairness_metrics.values():
            scores["metrics_total"] += 1
            if metric.passed:
                scores["metrics_passed"] += 1

        return scores

    def _serialize_metrics(
        self,
        metrics: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Serialize metrics dict for JSON output.

        Args:
            metrics: Metrics dictionary.

        Returns:
            Serializable metrics dict.
        """
        serialized: dict[str, Any] = {}
        for key, value in metrics.items():
            serialized[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.floating, float)):
                    serialized[key][k] = float(v)
                elif isinstance(v, (np.integer, int)):
                    serialized[key][k] = int(v)
                elif isinstance(v, np.bool_):
                    serialized[key][k] = bool(v)
                else:
                    serialized[key][k] = v
        return serialized

    def to_dict(self) -> dict[str, Any]:
        """Export report as a dictionary.

        Returns:
            Report data dictionary.
        """
        return self.report_data if self.report_data else self._build_report_data()

    def to_json(self, output_path: str) -> None:
        """Export report to a JSON file.

        Args:
            output_path: Path for the output JSON file.
        """
        data = self.to_dict()
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Fairness report saved to %s", output_path)

    def to_html(self, output_path: str) -> None:
        """Export report to a simple HTML file.

        Args:
            output_path: Path for the output HTML file.
        """
        data = self.to_dict()
        overall = data.get("overall_scores", {})
        overall_bias = overall.get("overall_bias_detected", False)
        overall_severity = overall.get("overall_severity", 0.0)

        severity_level = "low"
        if overall_severity > 0.5:
            severity_level = "high"
        elif overall_severity > 0.2:
            severity_level = "medium"

        table_rows = ""
        for attr in data.get("per_attribute_breakdown", []):
            attr_name = attr.get("attribute", "N/A")
            bias = "Yes" if attr.get("bias_detected") else "No"
            cls = "fail" if attr.get("bias_detected") else "pass"
            sev = attr.get("severity", 0)
            table_rows += (
                f"<tr><td>{attr_name}</td>"
                f'<td class="{cls}">{bias}</td>'
                f"<td>{sev:.2f}</td></tr>\n"
            )

        recs_html = "".join(f"<li>{r}</li>\n" for r in data.get("recommendations", []))

        html = HTML_TEMPLATE.format(
            generated_at=self.generated_at,
            overall_class="fail" if overall_bias else "pass",
            overall_bias="Yes" if overall_bias else "No",
            overall_severity=overall_severity,
            severity_level=severity_level,
            table_html=table_rows,
            recommendations_html=recs_html,
        )

        with open(output_path, "w") as f:
            f.write(html)
        logger.info("HTML fairness report saved to %s", output_path)

    def summary(self) -> str:
        """Generate a text summary of the report.

        Returns:
            Multi-line string summary.
        """
        data = self.to_dict()
        overall = data.get("overall_scores", {})
        lines = [
            "=" * 60,
            "FAIRNESS REPORT SUMMARY",
            "=" * 60,
            f"Generated: {self.generated_at}",
            f"Overall bias detected: {overall.get('overall_bias_detected', 'N/A')}",
            f"Overall severity: {overall.get('overall_severity', 0.0):.2f}",
            f"Metrics passed: {overall.get('metrics_passed', 0)}"
            f"/{overall.get('metrics_total', 0)}",
            "",
            "Per-Attribute Breakdown:",
        ]
        for attr in data.get("per_attribute_breakdown", []):
            lines.append(
                f"  - {attr.get('attribute')}: "
                f"bias={attr.get('bias_detected')}, "
                f"severity={attr.get('severity', 0):.2f}"
            )

        if data.get("mitigation_results"):
            lines.append("")
            lines.append("Mitigation Results:")
            mit = data["mitigation_results"]
            lines.append(f"  Strategy: {mit.get('strategy_used')}")
            lines.append(f"  Improvement: {mit.get('improvement', {})}")

        lines.append("")
        lines.append("Recommendations:")
        for rec in data.get("recommendations", []):
            lines.append(f"  - {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def compare_reports(
        report1: FairnessReport,
        report2: FairnessReport,
    ) -> dict[str, Any]:
        """Compare two fairness reports.

        Args:
            report1: First fairness report.
            report2: Second fairness report.

        Returns:
            Dictionary with comparison results.
        """
        data1 = report1.to_dict()
        data2 = report2.to_dict()

        scores1 = data1.get("overall_scores", {})
        scores2 = data2.get("overall_scores", {})

        comparison: dict[str, Any] = {
            "report1_generated": data1.get("generated_at"),
            "report2_generated": data2.get("generated_at"),
            "bias_detected_change": None,
            "severity_change": None,
            "metrics_change": {},
        }

        if scores1 and scores2:
            b1 = scores1.get("overall_bias_detected", False)
            b2 = scores2.get("overall_bias_detected", False)
            s1 = scores1.get("overall_severity", 0.0)
            s2 = scores2.get("overall_severity", 0.0)

            comparison["bias_detected_change"] = (
                "improved" if (b1 and not b2) else "worsened" if (not b1 and b2) else "unchanged"
            )
            comparison["severity_change"] = s2 - s1

        return comparison

    @staticmethod
    def recommend_mitigation(
        bias_results: BiasDetectionResult | None,
    ) -> list[str]:
        """Recommend the best mitigation strategy based on bias patterns.

        Args:
            bias_results: BiasDetectionResult to analyze.

        Returns:
            List of recommended mitigation strategies.
        """
        if bias_results is None:
            return ["No bias data available for recommendations."]

        if not bias_results.overall_bias_detected:
            return ["No significant bias detected; no mitigation required."]

        recommendations: list[str] = []

        severity = bias_results.severity
        if severity > 0.6:
            recommendations.append(
                "High severity bias detected. Recommended: "
                "adversarial debiasing (in-processing) combined with "
                "reweighing (pre-processing)."
            )
        elif severity > 0.3:
            recommendations.append(
                "Moderate bias detected. Recommended: "
                "equalized odds post-processing or sampling (pre-processing)."
            )
        else:
            recommendations.append("Low bias detected. Recommended: reweighing (pre-processing).")

        if len(bias_results.per_attribute) > 1:
            recommendations.append(
                "Multiple biased attributes detected. Consider "
                "intersectional analysis and adversarial debiasing."
            )

        return recommendations
