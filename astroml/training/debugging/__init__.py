"""Automated model debugging and error analysis toolkit.

Provides error analysis, confusion analysis, slice-based performance
analysis and failure mode identification for trained models.
"""

from astroml.training.debugging.confusion_analysis import (
    ClassMetrics,
    ConfusionAnalysisResult,
    ConfusionAnalyzer,
)
from astroml.training.debugging.error_analysis import (
    ClassErrorMetrics,
    ErrorAnalysisResult,
    ErrorAnalyzer,
)
from astroml.training.debugging.failure_modes import (
    FailureMode,
    FailureModeIdentifier,
    FailureModeReport,
)
from astroml.training.debugging.slice_analysis import (
    SliceAnalysisResult,
    SliceAnalyzer,
    SliceMetrics,
)

__all__ = [
    "ClassMetrics",
    "ConfusionAnalysisResult",
    "ConfusionAnalyzer",
    "ClassErrorMetrics",
    "ErrorAnalysisResult",
    "ErrorAnalyzer",
    "FailureMode",
    "FailureModeIdentifier",
    "FailureModeReport",
    "SliceAnalysisResult",
    "SliceAnalyzer",
    "SliceMetrics",
]
