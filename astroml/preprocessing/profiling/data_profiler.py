"""Statistical data profiling for tabular datasets.

Computes per-column statistics (counts, missingness, distributions,
outliers) plus dataset-level summaries and a data quality score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OUTLIER_IQR_MULTIPLIER = 1.5


@dataclass
class ColumnProfile:
    """Statistical profile of a single column.

    Attributes:
        name: Column name.
        dtype: Detected data type category ("numeric", "categorical",
            "datetime", "boolean", "text", "unknown").
        count: Number of non-null values.
        missing_count: Number of missing values.
        missing_rate: Fraction of missing values.
        unique_count: Number of unique values.
        cardinality_ratio: Unique count divided by non-null count.
        mean: Mean for numeric columns.
        std: Standard deviation for numeric columns.
        min: Minimum for numeric columns.
        q1: First quartile for numeric columns.
        median: Median for numeric columns.
        q3: Third quartile for numeric columns.
        max: Maximum for numeric columns.
        skewness: Skewness for numeric columns.
        kurtosis: Excess kurtosis for numeric columns.
        outlier_count: Number of IQR-based outliers for numeric columns.
        constant: Whether the column has a single unique value.
    """

    name: str
    dtype: str
    count: int
    missing_count: int
    missing_rate: float
    unique_count: int
    cardinality_ratio: float
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    max: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    outlier_count: int = 0
    constant: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize the profile to a plain dictionary.

        Returns:
            A JSON-friendly dictionary of the profile fields.
        """
        return {
            "name": self.name,
            "dtype": self.dtype,
            "count": self.count,
            "missing_count": self.missing_count,
            "missing_rate": self.missing_rate,
            "unique_count": self.unique_count,
            "cardinality_ratio": self.cardinality_ratio,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "q1": self.q1,
            "median": self.median,
            "q3": self.q3,
            "max": self.max,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "outlier_count": self.outlier_count,
            "constant": self.constant,
        }


@dataclass
class DataProfileResult:
    """Overall profile of a dataset.

    Attributes:
        row_count: Number of rows.
        column_count: Number of columns.
        duplicate_rows: Number of fully duplicated rows.
        missing_total: Total missing cells across the dataset.
        quality_score: Overall data quality score in ``[0, 100]``.
        columns: Per-column profiles keyed by column name.
    """

    row_count: int
    column_count: int
    duplicate_rows: int
    missing_total: int
    quality_score: float
    columns: dict[str, ColumnProfile] = field(default_factory=dict)


class DataProfiler:
    """Profile tabular data stored in pandas DataFrames."""

    def profile(self, df: pd.DataFrame) -> DataProfileResult:
        """Profile an entire DataFrame.

        Args:
            df: The dataset to profile.

        Returns:
            A :class:`DataProfileResult`.

        Raises:
            ValueError: If the DataFrame is empty.
        """
        if df.empty:
            raise ValueError("Cannot profile an empty DataFrame")
        columns = {name: self.profile_column(df[name]) for name in df.columns}
        duplicate_rows = int(df.duplicated().sum())
        missing_total = int(df.isna().sum().sum())
        result = DataProfileResult(
            row_count=len(df),
            column_count=len(df.columns),
            duplicate_rows=duplicate_rows,
            missing_total=missing_total,
            quality_score=self.quality_score(
                row_count=len(df),
                duplicate_rows=duplicate_rows,
                columns=list(columns.values()),
            ),
            columns=columns,
        )
        return result

    def profile_column(self, series: pd.Series) -> ColumnProfile:
        """Profile a single column.

        Args:
            series: The column to profile.

        Returns:
            A :class:`ColumnProfile`.
        """
        name = str(series.name)
        total = len(series)
        missing_count = int(series.isna().sum())
        missing_rate = missing_count / total if total else 0.0
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        cardinality_ratio = unique_count / len(non_null) if len(non_null) else 0.0
        constant = unique_count <= 1 and len(non_null) > 0
        dtype = self._detect_dtype(series)

        profile = ColumnProfile(
            name=name,
            dtype=dtype,
            count=int(len(non_null)),
            missing_count=missing_count,
            missing_rate=missing_rate,
            unique_count=unique_count,
            cardinality_ratio=cardinality_ratio,
            constant=constant,
        )

        if dtype == "numeric":
            self._fill_numeric_stats(profile, non_null)
        return profile

    def quality_score(
        self,
        row_count: int,
        duplicate_rows: int,
        columns: list[ColumnProfile],
    ) -> float:
        """Compute a data quality score in ``[0, 100]``.

        The score penalizes missing values, duplicate rows, constant
        columns and extremely high-cardinality categorical columns.

        Args:
            row_count: Number of rows in the dataset.
            duplicate_rows: Number of duplicate rows.
            columns: Per-column profiles.

        Returns:
            A quality score where 100 is perfect.
        """
        if row_count == 0:
            return 0.0
        score = 100.0
        total_cells = row_count * len(columns)
        missing_cells = sum(col.missing_count for col in columns)
        score -= 50.0 * (missing_cells / total_cells) if total_cells else 0.0
        score -= 20.0 * (duplicate_rows / row_count)
        score -= 10.0 * (sum(1 for col in columns if col.constant) / len(columns))
        high_cardinality = sum(
            1 for col in columns if col.dtype == "categorical" and col.cardinality_ratio > 0.95
        )
        score -= 5.0 * (high_cardinality / len(columns))
        return float(max(0.0, min(100.0, score)))

    def _fill_numeric_stats(self, profile: ColumnProfile, non_null: pd.Series) -> None:
        """Populate numeric statistics on a profile in place.

        Args:
            profile: The profile to update.
            non_null: Non-null numeric values.
        """
        values = non_null.astype(float)
        profile.mean = float(values.mean())
        profile.std = float(values.std(ddof=0))
        profile.min = float(values.min())
        profile.q1 = float(values.quantile(0.25))
        profile.median = float(values.median())
        profile.q3 = float(values.quantile(0.75))
        profile.max = float(values.max())
        if len(values) > 1:
            profile.skewness = float(values.skew())
            profile.kurtosis = float(values.kurt())
        iqr = profile.q3 - profile.q1
        lower = profile.q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper = profile.q3 + OUTLIER_IQR_MULTIPLIER * iqr
        profile.outlier_count = int(np.count_nonzero((values < lower) | (values > upper)))

    def _detect_dtype(self, series: pd.Series) -> str:
        """Detect the category of a column's data type.

        Args:
            series: The column to inspect.

        Returns:
            One of "numeric", "datetime", "boolean", "categorical",
            "text" or "unknown".
        """
        non_null = series.dropna()
        if len(non_null) == 0:
            return "unknown"
        if pd.api.types.is_bool_dtype(non_null):
            return "boolean"
        if pd.api.types.is_numeric_dtype(non_null):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(non_null):
            return "datetime"
        sample = non_null.astype(str)
        avg_length = float(sample.str.len().mean())
        if avg_length > 50:
            return "text"
        return "categorical"
