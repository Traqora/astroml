"""Automated data quality monitoring and reporting system."""

from astroml.validation.data_quality.alerts import (
    AlertChannel,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    CallbackAlertChannel,
    LoggingAlertChannel,
    QualityAlert,
)
from astroml.validation.data_quality.checks import (
    AccuracyChecker,
    BusinessRulesValidator,
    CheckResult,
    CheckSeverity,
    CompletenessChecker,
    ConsistencyChecker,
    DataQualityError,
    DataQualityReport,
    DataQualityValidator,
    MetricDimension,
    ReferentialIntegrityValidator,
    StatisticalValidator,
    TemporalValidator,
    TimelinessChecker,
    ValidationResult,
    check_referential_integrity,
    check_temporal_consistency,
    validate_data_quality,
)
from astroml.validation.data_quality.monitor import DataQualityMonitor
from astroml.validation.data_quality.reporter import DataQualityReporter

__all__ = [
    # Checks & dimensions
    "MetricDimension",
    "CheckSeverity",
    "CheckResult",
    "DataQualityError",
    "ValidationResult",
    "DataQualityReport",
    "CompletenessChecker",
    "ConsistencyChecker",
    "AccuracyChecker",
    "TimelinessChecker",
    # Legacy validators
    "DataQualityValidator",
    "TemporalValidator",
    "ReferentialIntegrityValidator",
    "BusinessRulesValidator",
    "StatisticalValidator",
    "validate_data_quality",
    "check_temporal_consistency",
    "check_referential_integrity",
    # Alerting
    "AlertSeverity",
    "AlertStatus",
    "AlertRule",
    "QualityAlert",
    "AlertChannel",
    "LoggingAlertChannel",
    "CallbackAlertChannel",
    "AlertManager",
    # Monitor & Reporter
    "DataQualityMonitor",
    "DataQualityReporter",
]
