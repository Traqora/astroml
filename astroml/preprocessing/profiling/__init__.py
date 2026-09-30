"""Automated data profiling and exploratory analysis.

Generates comprehensive exploratory data analysis reports with statistics,
visualizations and insights for tabular datasets.
"""

from astroml.preprocessing.profiling.data_profiler import (
    ColumnProfile,
    DataProfiler,
    DataProfileResult,
)
from astroml.preprocessing.profiling.insights import (
    Insight,
    InsightGenerator,
)
from astroml.preprocessing.profiling.report_generator import ReportGenerator
from astroml.preprocessing.profiling.visualizations import ProfilingVisualizer

__all__ = [
    "ColumnProfile",
    "DataProfileResult",
    "DataProfiler",
    "Insight",
    "InsightGenerator",
    "ReportGenerator",
    "ProfilingVisualizer",
]
