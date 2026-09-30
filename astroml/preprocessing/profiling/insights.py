"""Automated insight generation from data profiles.

Transforms computed profiles into actionable, human-readable insights with
severity levels, covering missingness, cardinality, distribution shape,
outliers and duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from astroml.preprocessing.profiling.data_profiler import (
    ColumnProfile,
    DataProfileResult,
)

MISSING_RATE_THRESHOLD = 0.1
SKEWNESS_THRESHOLD = 2.0
KURTOSIS_THRESHOLD = 5.0
OUTLIER_RATE_THRESHOLD = 0.01
HIGH_CARDINALITY_RATIO = 0.95


@dataclass
class Insight:
    """A single data quality insight.

    Attributes:
        type: Machine-readable category of the insight.
        severity: "info", "warning" or "critical".
        message: Human-readable description.
        column: Column the insight applies to, if any.
        details: Optional structured details.
    """

    type: str
    severity: str
    message: str
    column: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the insight to a plain dictionary.

        Returns:
            A JSON-friendly dictionary.
        """
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "column": self.column,
            "details": self.details,
        }


class InsightGenerator:
    """Generate insights from a :class:`DataProfileResult`."""

    def __init__(
        self,
        missing_rate_threshold: float = MISSING_RATE_THRESHOLD,
        skewness_threshold: float = SKEWNESS_THRESHOLD,
        kurtosis_threshold: float = KURTOSIS_THRESHOLD,
        outlier_rate_threshold: float = OUTLIER_RATE_THRESHOLD,
    ) -> None:
        """Initialize the generator with configurable thresholds.

        Args:
            missing_rate_threshold: Missing rate above which a column is flagged.
            skewness_threshold: Absolute skewness above which a column is flagged.
            kurtosis_threshold: Kurtosis above which a column is flagged.
            outlier_rate_threshold: Outlier rate above which a column is flagged.
        """
        self.missing_rate_threshold = missing_rate_threshold
        self.skewness_threshold = skewness_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.outlier_rate_threshold = outlier_rate_threshold

    def generate(self, profile: DataProfileResult) -> list[Insight]:
        """Generate all insights for a dataset profile.

        Args:
            profile: The computed data profile.

        Returns:
            A list of :class:`Insight` objects.
        """
        insights: list[Insight] = []
        for column in profile.columns.values():
            insights.extend(self._column_insights(column))
        if profile.duplicate_rows > 0:
            insights.append(
                Insight(
                    type="duplicate_rows",
                    severity="warning",
                    message=(
                        f"Dataset contains {profile.duplicate_rows} duplicate rows "
                        f"({profile.duplicate_rows / profile.row_count:.1%})."
                    ),
                    details={"count": profile.duplicate_rows},
                )
            )
        if profile.missing_total > 0:
            insights.append(
                Insight(
                    type="missing_values",
                    severity="info",
                    message=(
                        f"Dataset has {profile.missing_total} missing cells across "
                        f"{profile.column_count} columns."
                    ),
                    details={"count": profile.missing_total},
                )
            )
        return insights

    def summarize(self, insights: list[Insight]) -> dict[str, int]:
        """Count insights by severity.

        Args:
            insights: Insights to summarize.

        Returns:
            Mapping of severity to count.
        """
        summary: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
        for insight in insights:
            summary[insight.severity] = summary.get(insight.severity, 0) + 1
        return summary

    def _column_insights(self, column: ColumnProfile) -> list[Insight]:
        """Generate insights for a single column profile.

        Args:
            column: The column profile to analyze.

        Returns:
            A list of :class:`Insight` objects for the column.
        """
        insights: list[Insight] = []
        if column.missing_rate >= self.missing_rate_threshold:
            insights.append(
                Insight(
                    type="high_missing_rate",
                    severity="critical" if column.missing_rate >= 0.5 else "warning",
                    message=(
                        f"Column '{column.name}' has {column.missing_rate:.1%} missing "
                        "values; consider imputation or dropping it."
                    ),
                    column=column.name,
                    details={"missing_rate": column.missing_rate},
                )
            )
        if column.constant:
            insights.append(
                Insight(
                    type="constant_column",
                    severity="warning",
                    message=(
                        f"Column '{column.name}' is constant and carries no "
                        "discriminative information."
                    ),
                    column=column.name,
                    details={"unique_count": column.unique_count},
                )
            )
        if column.dtype == "categorical" and column.cardinality_ratio >= HIGH_CARDINALITY_RATIO:
            insights.append(
                Insight(
                    type="high_cardinality",
                    severity="warning",
                    message=(
                        f"Column '{column.name}' has near-unique values "
                        f"({column.unique_count} unique), which may hurt generalization."
                    ),
                    column=column.name,
                    details={"unique_count": column.unique_count},
                )
            )
        if column.skewness is not None and abs(column.skewness) >= self.skewness_threshold:
            insights.append(
                Insight(
                    type="skewed_distribution",
                    severity="info",
                    message=(
                        f"Column '{column.name}' is heavily skewed "
                        f"(skewness={column.skewness:.2f}); consider a transform."
                    ),
                    column=column.name,
                    details={"skewness": column.skewness},
                )
            )
        if column.kurtosis is not None and column.kurtosis >= self.kurtosis_threshold:
            insights.append(
                Insight(
                    type="heavy_tails",
                    severity="info",
                    message=(
                        f"Column '{column.name}' has heavy tails "
                        f"(kurtosis={column.kurtosis:.2f})."
                    ),
                    column=column.name,
                    details={"kurtosis": column.kurtosis},
                )
            )
        outlier_rate = column.outlier_count / column.count if column.count else 0.0
        if outlier_rate >= self.outlier_rate_threshold:
            insights.append(
                Insight(
                    type="outliers",
                    severity="warning",
                    message=(
                        f"Column '{column.name}' contains {column.outlier_count} "
                        f"IQR outliers ({outlier_rate:.1%} of values)."
                    ),
                    column=column.name,
                    details={"outlier_count": column.outlier_count},
                )
            )
        return insights
