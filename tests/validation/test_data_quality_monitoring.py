"""Tests for automated data quality monitoring, reporting, and alerting."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers.data_quality import router as data_quality_router
from astroml.validation.data_quality.alerts import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    CallbackAlertChannel,
    QualityAlert,
)
from astroml.validation.data_quality.checks import (
    AccuracyChecker,
    CheckSeverity,
    CompletenessChecker,
    ConsistencyChecker,
    DataQualityReport,
    MetricDimension,
    TimelinessChecker,
)
from astroml.validation.data_quality.monitor import DataQualityMonitor
from astroml.validation.data_quality.reporter import DataQualityReporter

# ---------------------------------------------------------------------------
# Checkers Unit Tests
# ---------------------------------------------------------------------------


class TestCompletenessChecker:
    def test_completeness_all_present(self):
        checker = CompletenessChecker(required_fields=["id", "amount", "account"])
        records = [
            {"id": "tx_1", "amount": 100.0, "account": "ACC1"},
            {"id": "tx_2", "amount": 50.0, "account": "ACC2"},
        ]
        results = checker.check_records(records)
        assert all(r.is_valid for r in results)
        req_res = next(r for r in results if r.check_name == "required_fields_check")
        assert req_res.score == 1.0

    def test_completeness_missing_required(self):
        checker = CompletenessChecker(required_fields=["id", "amount", "account"])
        records = [
            {"id": "tx_1", "amount": 100.0},  # Missing account
            {"id": "tx_2", "amount": None, "account": "ACC2"},  # None amount
            {"id": "tx_3", "amount": 50.0, "account": "   "},  # Empty string account
        ]
        results = checker.check_records(records)
        req_res = next(r for r in results if r.check_name == "required_fields_check")
        assert not req_res.is_valid
        assert req_res.score == 0.0
        assert req_res.details["total_affected"] == 3

    def test_completeness_empty_dataset(self):
        checker = CompletenessChecker()
        results = checker.check_records([])
        assert len(results) == 1
        assert results[0].is_valid


class TestConsistencyChecker:
    def test_type_and_format_consistency(self):
        checker = ConsistencyChecker(schema_types={"amount": float, "id": str})
        records = [
            {
                "id": "tx_1",
                "amount": 100.0,
                "source_account": "GABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
                "asset_code": "XLM",
            },
            {
                "id": "tx_2",
                "amount": "not_a_float",  # type mismatch
                "source_account": "BAD_ACCOUNT",  # malformed account
                "asset_code": "toolongassetcode123456",  # malformed asset
            },
        ]
        results = checker.check_records(records)
        type_res = next(r for r in results if r.check_name == "type_consistency_amount")
        assert not type_res.is_valid
        assert type_res.details["mismatches"] == 1

        acc_res = next(r for r in results if r.check_name == "account_format_consistency")
        assert not acc_res.is_valid
        assert acc_res.details["invalid_count"] == 1

    def test_custom_consistency_rules(self):
        def amount_greater_than_fee(rec):
            return rec.get("amount", 0) >= rec.get("fee", 0)

        checker = ConsistencyChecker(
            custom_rules=[("amt_gte_fee", amount_greater_than_fee, "Amount must be >= fee")]
        )
        records = [
            {"amount": 100, "fee": 10},
            {"amount": 5, "fee": 50},  # Violation
        ]
        results = checker.check_records(records)
        rule_res = next(r for r in results if r.check_name == "custom_rule_amt_gte_fee")
        assert not rule_res.is_valid
        assert rule_res.score == 0.5


class TestAccuracyChecker:
    def test_iqr_outliers(self):
        checker = AccuracyChecker(iqr_multiplier=1.5)
        # Normal spread + 1 extreme outlier
        vals = [10.0, 11.0, 12.0, 10.5, 11.5, 12.5, 1000.0]
        res = checker.check_outliers_iqr(vals, "amount")
        assert not res.is_valid
        assert res.details["outlier_count"] == 1

    def test_numeric_bounds(self):
        checker = AccuracyChecker(numeric_bounds={"ratio": (0.0, 1.0)})
        records = [{"ratio": 0.5}, {"ratio": 1.5}]  # 1.5 out of bounds
        results = checker.check_records(records)
        bound_res = next(r for r in results if r.check_name == "range_bounds_ratio")
        assert not bound_res.is_valid
        assert bound_res.details["violations"] == 1


class TestTimelinessChecker:
    def test_freshness_sla(self):
        checker = TimelinessChecker(max_latency_seconds=600.0)
        now = datetime.now(timezone.utc)
        # Stale record (2 hours ago)
        records = [{"timestamp": (now - timedelta(hours=2)).isoformat()}]
        results = checker.check_records(records, reference_time=now)
        fresh_res = next(r for r in results if r.check_name == "data_freshness_latency")
        assert not fresh_res.is_valid
        assert fresh_res.details["latency_seconds"] >= 7200.0


# ---------------------------------------------------------------------------
# Alert Manager Tests
# ---------------------------------------------------------------------------


class TestAlertManager:
    def test_alert_rule_triggering(self):
        dispatched_alerts = []
        channel = CallbackAlertChannel(lambda a: dispatched_alerts.append(a))
        mgr = AlertManager(channels=[channel], auto_default_rules=False)

        mgr.add_rule(
            AlertRule(
                rule_id="r1",
                name="Low Completeness",
                metric_name="completeness",
                dimension=MetricDimension.COMPLETENESS,
                operator="<",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
            )
        )

        alerts = mgr.evaluate_metrics({"completeness": 75.0, "accuracy": 99.0})
        assert len(alerts) == 1
        assert alerts[0].rule_id == "r1"
        assert len(dispatched_alerts) == 1

    def test_degradation_detection(self):
        mgr = AlertManager()
        alert = mgr.check_quality_degradation(
            current_score=80.0,
            baseline_score=95.0,
            max_drop_percentage=15.0,
        )
        assert alert is not None
        assert alert.rule_id == "degradation_detector"
        assert alert.severity in (AlertSeverity.ERROR, AlertSeverity.CRITICAL)

    def test_resolve_and_silence_alerts(self):
        mgr = AlertManager()
        alerts = mgr.evaluate_metrics({"completeness": 50.0})
        assert len(alerts) > 0
        aid = alerts[0].alert_id

        assert mgr.silence_alert(aid)
        assert mgr.get_rule(aid) is None
        assert len(mgr.get_active_alerts()) == len(alerts) - 1

        assert mgr.resolve_alert(aid)
        resolved_list = mgr.list_alerts(status=AlertStatus.RESOLVED)
        assert any(a.alert_id == aid for a in resolved_list)


# ---------------------------------------------------------------------------
# Reporter & Monitor Tests
# ---------------------------------------------------------------------------


class TestDataQualityMonitorAndReporter:
    def test_monitor_process_batch_and_trend(self):
        monitor = DataQualityMonitor()
        now = datetime.now(timezone.utc)
        batch1 = [
            {
                "id": "tx_1",
                "timestamp": now.isoformat(),
                "source_account": "GABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
                "asset_code": "XLM",
                "ledger_sequence": 100,
                "fee": 100,
                "amount": 50.0,
                "operation_count": 1,
            }
        ]
        rep1 = monitor.process_batch(batch1, batch_id="b1")
        assert rep1.quality_score > 0

        # Export report in json, markdown, html
        json_rep = monitor.export_latest_report(format="json")
        md_rep = monitor.export_latest_report(format="markdown")
        html_rep = monitor.export_latest_report(format="html")

        assert "b1" in json_rep
        assert "AstroML Data Quality Report" in md_rep
        assert "<!DOCTYPE html>" in html_rep

        # Metrics history & trends
        history = monitor.get_metrics_history()
        assert len(history) > 0
        trend = monitor.get_quality_trend()
        assert trend["count"] == 1


# ---------------------------------------------------------------------------
# FastAPI Router Tests
# ---------------------------------------------------------------------------


class TestDataQualityAPI:
    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.include_router(data_quality_router)
        return TestClient(app)

    def test_api_check_and_monitor(self, client):
        now = datetime.now(timezone.utc)
        payload = {
            "records": [
                {
                    "id": "tx_1",
                    "timestamp": now.isoformat(),
                    "source_account": "GABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
                    "asset_code": "XLM",
                    "ledger_sequence": 100,
                    "fee": 100,
                    "amount": 50.0,
                    "operation_count": 1,
                }
            ]
        }
        res = client.post("/api/v1/data-quality/check", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "quality_score" in data

        # Monitor batch
        res_mon = client.post("/api/v1/data-quality/monitor", json={"records": payload["records"], "batch_id": "api_b1"})
        assert res_mon.status_code == 200
        assert res_mon.json()["batch_id"] == "api_b1"

        # Get latest report
        res_rep = client.get("/api/v1/data-quality/reports/latest?format=json")
        assert res_rep.status_code == 200

        # Get metrics
        res_met = client.get("/api/v1/data-quality/metrics")
        assert res_met.status_code == 200

        # Get trends
        res_tr = client.get("/api/v1/data-quality/trends")
        assert res_tr.status_code == 200

        # Add rule
        rule_payload = {
            "rule_id": "test_r1",
            "name": "Test Rule",
            "metric_name": "completeness",
            "dimension": "completeness",
            "operator": "<",
            "threshold": 95.0,
            "severity": "warning",
            "description": "Test warning",
        }
        res_rule = client.post("/api/v1/data-quality/alerts/rules", json=rule_payload)
        assert res_rule.status_code == 200
