from typing import Any

from .collector import get_metrics_collector


class AlertManager:
    def __init__(self) -> Any:
        self.alerts: list[dict[str, Any]] = []
        self.thresholds = {
            "latency_p95_limit": 2.5,  # seconds
            "error_rate_limit": 0.10,  # 10%
            "monthly_spend_limit": 500.0,  # USD
            "cost_spike_limit": 5.0,  # USD / hour increase
        }

    def check_alerts(self) -> list[dict[str, Any]]:
        collector = get_metrics_collector()
        summary = collector.get_summary_metrics()

        # Latency check
        if summary["p95_latency"] > self.thresholds["latency_p95_limit"]:
            self._add_alert(
                "high_latency",
                f"P95 latency is {summary['p95_latency']:.2f}s, exceeding limit of {self.thresholds['latency_p95_limit']}s",
            )

        # Error rate check
        error_rate = summary.get("error_rate", 0.0)
        if error_rate > self.thresholds["error_rate_limit"]:
            self._add_alert(
                "high_error_rate",
                f"Error rate is {error_rate * 100:.1f}%, exceeding limit of {self.thresholds['error_rate_limit'] * 100}%",
            )

        # Monthly spend check
        if summary["monthly_spend"] > self.thresholds["monthly_spend_limit"]:
            self._add_alert(
                "spend_limit_exceeded",
                f"Monthly spend is ${summary['monthly_spend']:.2f}, exceeding threshold of ${self.thresholds['monthly_spend_limit']:.2f}",
            )

        # Anomaly detection alerts
        anomalies = collector.detect_anomalies()
        for anomaly in anomalies:
            self._add_alert("anomaly_detected", anomaly)

        return self.alerts

    def _add_alert(self, alert_type -> Any: str, message: str):
        # Prevent duplicate alerts in a short span
        for alert in self.alerts:
            if alert["type"] == alert_type and alert["message"] == message:
                return
        self.alerts.append(
            {"type": alert_type, "message": message, "timestamp": time.time(), "status": "active"}
        )


_alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    return _alert_manager
