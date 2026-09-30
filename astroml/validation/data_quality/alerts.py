from astroml.utils.exceptions import AstroMLError
"""Alerting mechanisms for data quality degradation and threshold breaches.

Provides rule-based alerting, severity classification, degradation detection,
and alert dispatching channels.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from astroml.validation.data_quality.checks import CheckSeverity, MetricDimension

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Lifecycle status of a quality alert."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    SILENCED = "silenced"


@dataclass
class AlertRule:
    """Threshold rule for generating data quality alerts."""

    rule_id: str
    name: str
    metric_name: str
    dimension: MetricDimension | str
    operator: str  # "<", ">", "<=", ">=", "==", "!="
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    description: str = ""
    enabled: bool = True


@dataclass
class QualityAlert:
    """Represents an active or resolved data quality alert."""

    alert_id: str
    rule_id: str
    rule_name: str
    metric_name: str
    dimension: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    message: str = ""
    batch_id: str | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)
    created_at: str = dc_field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at: str | None = None
    resolved_by: str | None = None


class AlertChannel:
    """Base class for alert dispatch channels."""

    def dispatch(self, alert: QualityAlert) -> None:
        raise NotImplementedError


class LoggingAlertChannel(AlertChannel):
    """Dispatches alerts via standard Python logging."""

    def dispatch(self, alert: QualityAlert) -> None:
        log_fn = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(alert.severity, logger.warning)

        log_fn(
            "DATA QUALITY ALERT [%s] [%s]: %s (Value: %s, Threshold: %s)",
            alert.severity.value.upper(),
            alert.rule_name,
            alert.message,
            alert.current_value,
            alert.threshold,
        )


class CallbackAlertChannel(AlertChannel):
    """Dispatches alerts to a user-provided callback function."""

    def __init__(self, callback: Callable[[QualityAlert], None]) -> None:
        self.callback = callback

    def dispatch(self, alert: QualityAlert) -> None:
        try:
            self.callback(alert)
        except AstroMLError as e:
            logger.error("Failed to execute alert callback: %s", e)


class AlertManager:
    """Manages rules, evaluations, alerts, and dispatching."""

    def __init__(
        self,
        channels: list[AlertChannel] | None = None,
        auto_default_rules: bool = True,
    ) -> None:
        self._rules: dict[str, AlertRule] = {}
        self._alerts: dict[str, QualityAlert] = {}
        self._channels: list[AlertChannel] = channels or [LoggingAlertChannel()]

        if auto_default_rules:
            self._load_default_rules()

    def _load_default_rules(self) -> None:
        """Register default quality monitoring rules."""
        defaults = [
            AlertRule(
                rule_id="rule_completeness_crit",
                name="Critical Completeness Drop",
                metric_name="completeness",
                dimension=MetricDimension.COMPLETENESS,
                operator="<",
                threshold=80.0,
                severity=AlertSeverity.CRITICAL,
                description="Completeness dropped below 80%",
            ),
            AlertRule(
                rule_id="rule_completeness_warn",
                name="Low Completeness Warning",
                metric_name="completeness",
                dimension=MetricDimension.COMPLETENESS,
                operator="<",
                threshold=95.0,
                severity=AlertSeverity.WARNING,
                description="Completeness dropped below 95%",
            ),
            AlertRule(
                rule_id="rule_consistency_warn",
                name="Consistency Breach",
                metric_name="consistency",
                dimension=MetricDimension.CONSISTENCY,
                operator="<",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
                description="Consistency dropped below 90%",
            ),
            AlertRule(
                rule_id="rule_accuracy_warn",
                name="Accuracy Anomaly Warning",
                metric_name="accuracy",
                dimension=MetricDimension.ACCURACY,
                operator="<",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                description="Accuracy score dropped below 85%",
            ),
            AlertRule(
                rule_id="rule_timeliness_warn",
                name="Timeliness SLA Breach",
                metric_name="timeliness",
                dimension=MetricDimension.TIMELINESS,
                operator="<",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                description="Timeliness / freshness SLA breach",
            ),
            AlertRule(
                rule_id="rule_overall_crit",
                name="Critical Overall Quality Degradation",
                metric_name="overall_score",
                dimension="overall",
                operator="<",
                threshold=75.0,
                severity=AlertSeverity.CRITICAL,
                description="Overall quality score fell below 75%",
            ),
        ]
        for r in defaults:
            self.add_rule(r)

    def add_rule(self, rule: AlertRule) -> None:
        """Add or update an alert rule."""
        self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def get_rule(self, rule_id: str) -> AlertRule | None:
        """Get an alert rule."""
        return self._rules.get(rule_id)

    def list_rules(self) -> list[AlertRule]:
        """List all configured rules."""
        return list(self._rules.values())

    def add_channel(self, channel: AlertChannel) -> None:
        """Add a notification channel."""
        self._channels.append(channel)

    def _eval_condition(self, val: float, op: str, threshold: float) -> bool:
        if op == "<":
            return val < threshold
        elif op == "<=":
            return val <= threshold
        elif op == ">":
            return val > threshold
        elif op == ">=":
            return val >= threshold
        elif op == "==":
            return val == threshold
        elif op == "!=":
            return val != threshold
        return False

    def evaluate_metrics(
        self,
        metrics: dict[str, float],
        batch_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> list[QualityAlert]:
        """Evaluate current metrics against all active rules."""
        triggered: list[QualityAlert] = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            if rule.metric_name in metrics:
                val = metrics[rule.metric_name]
                if self._eval_condition(val, rule.operator, rule.threshold):
                    alert_id = f"alert_{uuid.uuid4().hex[:10]}"
                    msg = (
                        f"Metric '{rule.metric_name}' is {val:.2f}, violating condition "
                        f"'{rule.operator} {rule.threshold:.2f}'"
                    )
                    alert = QualityAlert(
                        alert_id=alert_id,
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        metric_name=rule.metric_name,
                        dimension=str(rule.dimension),
                        current_value=val,
                        threshold=rule.threshold,
                        severity=rule.severity,
                        status=AlertStatus.ACTIVE,
                        message=msg,
                        batch_id=batch_id,
                        details=details or {},
                    )
                    self._alerts[alert_id] = alert
                    triggered.append(alert)
                    self._dispatch_alert(alert)

        return triggered

    def check_quality_degradation(
        self,
        current_score: float,
        baseline_score: float,
        max_drop_percentage: float = 10.0,
        batch_id: str | None = None,
    ) -> QualityAlert | None:
        """Detect significant quality drops compared to baseline or previous runs."""
        if baseline_score <= 0:
            return None

        drop = baseline_score - current_score
        if drop >= max_drop_percentage:
            alert_id = f"alert_degrade_{uuid.uuid4().hex[:8]}"
            msg = (
                f"Data quality degraded by {drop:.1f}% "
                f"(Current: {current_score:.1f}%, Baseline: {baseline_score:.1f}%)"
            )
            alert = QualityAlert(
                alert_id=alert_id,
                rule_id="degradation_detector",
                rule_name="Quality Score Degradation",
                metric_name="overall_score",
                dimension="overall",
                current_value=current_score,
                threshold=baseline_score - max_drop_percentage,
                severity=AlertSeverity.ERROR if drop < 25.0 else AlertSeverity.CRITICAL,
                status=AlertStatus.ACTIVE,
                message=msg,
                batch_id=batch_id,
                details={
                    "baseline_score": baseline_score,
                    "current_score": current_score,
                    "score_drop": drop,
                    "max_tolerated_drop": max_drop_percentage,
                },
            )
            self._alerts[alert_id] = alert
            self._dispatch_alert(alert)
            return alert

        return None

    def _dispatch_alert(self, alert: QualityAlert) -> None:
        for ch in self._channels:
            try:
                ch.dispatch(alert)
            except AstroMLError as e:
                logger.error("Failed to dispatch alert to channel %s: %s", ch, e)

    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Mark an alert as resolved."""
        if alert_id in self._alerts:
            alert = self._alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc).isoformat()
            alert.resolved_by = resolved_by
            return True
        return False

    def silence_alert(self, alert_id: str) -> bool:
        """Silence an active alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.SILENCED
            return True
        return False

    def list_alerts(
        self,
        status: AlertStatus | str | None = None,
        severity: AlertSeverity | str | None = None,
    ) -> list[QualityAlert]:
        """List alerts with optional status and severity filtering."""
        alerts = list(self._alerts.values())
        if status:
            s_val = status.value if isinstance(status, AlertStatus) else status
            alerts = [a for a in alerts if a.status.value == s_val]
        if severity:
            sev_val = severity.value if isinstance(severity, AlertSeverity) else severity
            alerts = [a for a in alerts if a.severity.value == sev_val]
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_active_alerts(self) -> list[QualityAlert]:
        """Get all currently active alerts."""
        return self.list_alerts(status=AlertStatus.ACTIVE)
