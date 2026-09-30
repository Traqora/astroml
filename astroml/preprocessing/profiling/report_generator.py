"""Automated report generation for data profiles.

Renders a :class:`DataProfileResult` and its insights as Markdown, HTML,
JSON or a printable PDF so profiling results can be shared with
stakeholders.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from astroml.preprocessing.profiling.data_profiler import DataProfileResult
from astroml.preprocessing.profiling.insights import Insight

ReportFormat = str  # one of "markdown", "html", "json", "pdf"


class ReportGenerator:
    """Generate profiling reports in multiple formats."""

    def generate(
        self,
        profile: DataProfileResult,
        insights: list[Insight],
        fmt: ReportFormat = "html",
        output_path: str | Path | None = None,
    ) -> str:
        """Generate a report and optionally write it to disk.

        Args:
            profile: The data profile to report on.
            insights: Insights to include.
            fmt: Output format: "markdown", "html", "json" or "pdf".
            output_path: Optional destination file path.

        Returns:
            The generated report (file path for PDF output).

        Raises:
            ValueError: If ``fmt`` is not supported.
        """
        if fmt == "markdown":
            report = self.generate_markdown(profile, insights)
        elif fmt == "html":
            report = self.generate_html(profile, insights)
        elif fmt == "json":
            report = self.generate_json(profile, insights)
        elif fmt == "pdf":
            return self.generate_pdf(profile, insights, output_path)
        else:
            raise ValueError(f"Unsupported report format: {fmt!r}")
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report, encoding="utf-8")
        return report

    def generate_markdown(self, profile: DataProfileResult, insights: list[Insight]) -> str:
        """Render a Markdown report.

        Args:
            profile: The data profile.
            insights: Insights to include.

        Returns:
            The Markdown report as a string.
        """
        lines = [
            "# Data Profiling Report",
            "",
            f"- **Rows:** {profile.row_count}",
            f"- **Columns:** {profile.column_count}",
            f"- **Duplicate rows:** {profile.duplicate_rows}",
            f"- **Missing cells:** {profile.missing_total}",
            f"- **Quality score:** {profile.quality_score:.1f}/100",
            "",
            "## Column Profiles",
            "",
        ]
        for column in profile.columns.values():
            lines.append(f"### {column.name} (`{column.dtype}`)")
            lines.append("")
            lines.append(f"- Count: {column.count} | Missing: {column.missing_rate:.1%}")
            lines.append(f"- Unique: {column.unique_count}")
            if column.mean is not None:
                lines.append(
                    f"- Mean: {column.mean:.3f} | Std: {column.std:.3f} | "
                    f"Median: {column.median:.3f}"
                )
                lines.append(
                    f"- Range: [{column.min:.3f}, {column.max:.3f}] | "
                    f"Outliers: {column.outlier_count}"
                )
            lines.append("")
        lines.append("## Insights")
        lines.append("")
        if not insights:
            lines.append("No insights generated.")
        for insight in insights:
            lines.append(f"- **[{insight.severity}]** {insight.message}")
        lines.append("")
        return "\n".join(lines)

    def generate_html(self, profile: DataProfileResult, insights: list[Insight]) -> str:
        """Render an HTML report.

        Args:
            profile: The data profile.
            insights: Insights to include.

        Returns:
            The HTML report as a string.
        """
        rows = []
        for column in profile.columns.values():
            rows.append(
                "<tr>"
                f"<td>{column.name}</td><td>{column.dtype}</td>"
                f"<td>{column.count}</td><td>{column.missing_rate:.1%}</td>"
                f"<td>{column.unique_count}</td>"
                f"<td>{column.mean if column.mean is None else round(column.mean, 3)}</td>"
                f"<td>{column.outlier_count}</td>"
                "</tr>"
            )
        insight_items = "".join(
            f"<li class='{insight.severity}'>{insight.message}</li>" for insight in insights
        )
        table = (
            "<table><thead><tr><th>Column</th><th>Type</th><th>Count</th>"
            "<th>Missing</th><th>Unique</th><th>Mean</th><th>Outliers</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Data Profiling Report</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f4f4f4; }}
li.critical {{ color: #b00020; }}
li.warning {{ color: #b26a00; }}
</style></head>
<body>
<h1>Data Profiling Report</h1>
<p>Rows: <b>{profile.row_count}</b> | Columns: <b>{profile.column_count}</b> |
Duplicate rows: <b>{profile.duplicate_rows}</b> |
Missing cells: <b>{profile.missing_total}</b> |
Quality score: <b>{profile.quality_score:.1f}/100</b></p>
<h2>Column Profiles</h2>
{table}
<h2>Insights</h2>
<ul>{insight_items or '<li>No insights generated.</li>'}</ul>
</body></html>
"""

    def generate_json(self, profile: DataProfileResult, insights: list[Insight]) -> str:
        """Render a JSON report.

        Args:
            profile: The data profile.
            insights: Insights to include.

        Returns:
            The JSON report as a string.
        """
        payload = {
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "duplicate_rows": profile.duplicate_rows,
            "missing_total": profile.missing_total,
            "quality_score": profile.quality_score,
            "columns": {name: column.to_dict() for name, column in profile.columns.items()},
            "insights": [insight.to_dict() for insight in insights],
        }
        return json.dumps(payload, indent=2, default=str)

    def generate_pdf(
        self,
        profile: DataProfileResult,
        insights: list[Insight],
        output_path: str | Path | None = None,
    ) -> str:
        """Render a PDF report using matplotlib's PDF backend.

        Args:
            profile: The data profile.
            insights: Insights to include.
            output_path: Destination file path. Defaults to
                ``data_profiling_report.pdf`` in the current directory.

        Returns:
            The path of the generated PDF file.
        """
        path = Path(output_path) if output_path is not None else Path("data_profiling_report.pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        markdown = self.generate_markdown(profile, insights)
        with PdfPages(path) as pdf:
            fig = Figure(figsize=(8.5, 11))
            fig.text(0.02, 0.98, markdown, fontsize=8, family="monospace", va="top")
            pdf.savefig(fig)
        return str(path)
