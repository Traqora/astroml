"""Tests for ML cost tracking, budget management and resource optimization.

Covers #647.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from astroml.tracking.budget_manager import (
    AlertSeverity,
    Budget,
    BudgetExceededError,
    BudgetManager,
    BudgetPeriod,
)
from astroml.tracking.cost_tracker import (
    CostCategory,
    CostRecord,
    CostTracker,
    ResourceType,
    utcnow,
)
from astroml.tracking.resource_optimizer import (
    OptimizationKind,
    ResourceOptimizer,
    ResourceUsageSample,
    Severity,
)


@pytest.fixture
def tracker() -> CostTracker:
    """A tracker with deterministic prices."""
    return CostTracker(
        compute_prices={ResourceType.GPU_A100: 3.0, ResourceType.CPU: 0.05},
        api_prices={"openai": 2.0},
    )


# ─── CostTracker ─────────────────────────────────────────────────────────────


class TestCostTracker:
    """Cost ledger behaviour."""

    def test_record_compute_uses_price_table(self, tracker: CostTracker) -> None:
        record = tracker.record_compute(ResourceType.GPU_A100, 2.0, count=2)
        assert record.category is CostCategory.COMPUTE
        assert record.quantity == pytest.approx(4.0)
        assert record.cost_usd == pytest.approx(12.0)

    def test_explicit_cost_overrides_price_table(self, tracker: CostTracker) -> None:
        record = tracker.record_compute(ResourceType.GPU_A100, 2.0, cost_usd=1.23)
        assert record.cost_usd == pytest.approx(1.23)

    def test_record_api_prices_per_thousand_calls(self, tracker: CostTracker) -> None:
        record = tracker.record_api("openai", 2_500)
        assert record.cost_usd == pytest.approx(5.0)
        assert record.unit == "calls"

    def test_unknown_api_service_costs_nothing(self, tracker: CostTracker) -> None:
        assert tracker.record_api("mystery", 1000).cost_usd == 0.0

    def test_record_storage_prorates_by_days(self) -> None:
        tracker = CostTracker(storage_price_usd_per_gb_month=1.0)
        assert tracker.record_storage(10.0, days=15.0).cost_usd == pytest.approx(5.0)

    def test_negative_inputs_rejected(self, tracker: CostTracker) -> None:
        with pytest.raises(ValueError):
            tracker.record_compute(ResourceType.CPU, -1.0)
        with pytest.raises(ValueError):
            tracker.record_api("openai", -1)
        with pytest.raises(ValueError):
            tracker.record_storage(-1.0)

    def test_set_price_overrides_compute_and_api(self, tracker: CostTracker) -> None:
        tracker.set_price(ResourceType.CPU, 1.0)
        tracker.set_price("openai", 10.0)
        assert tracker.compute_price(ResourceType.CPU) == 1.0
        assert tracker.api_price("openai") == 10.0
        with pytest.raises(ValueError):
            tracker.set_price(ResourceType.CPU, -1.0)

    def test_totals_and_breakdowns(self, tracker: CostTracker) -> None:
        tracker.record_compute(ResourceType.GPU_A100, 1.0, project="fraud", team="ml")
        tracker.record_api("openai", 1_000, project="loyalty", team="ml")
        assert tracker.total_cost_usd() == pytest.approx(5.0)
        assert tracker.breakdown("category") == {
            CostCategory.COMPUTE.value: pytest.approx(3.0),
            CostCategory.API.value: pytest.approx(2.0),
        }
        assert tracker.breakdown("project")["fraud"] == pytest.approx(3.0)
        assert tracker.breakdown("team")["ml"] == pytest.approx(5.0)

    def test_unallocated_records_grouped_under_placeholder(self, tracker: CostTracker) -> None:
        tracker.record_compute(ResourceType.CPU, 1.0)
        assert "unallocated" in tracker.breakdown("project")

    def test_unknown_dimension_rejected(self, tracker: CostTracker) -> None:
        with pytest.raises(ValueError, match="unknown dimension"):
            tracker.breakdown("nonsense")

    def test_filtering_by_project_and_time(self, tracker: CostTracker) -> None:
        old = CostRecord(
            category=CostCategory.COMPUTE,
            resource="cpu",
            quantity=1.0,
            unit="hours",
            cost_usd=100.0,
            project="fraud",
            timestamp=utcnow() - timedelta(days=30),
        )
        tracker.record(old)
        tracker.record_compute(ResourceType.GPU_A100, 1.0, project="fraud")
        recent = tracker.total_cost_usd(since=utcnow() - timedelta(days=1))
        assert recent == pytest.approx(3.0)
        assert tracker.total_cost_usd(project="loyalty") == 0.0

    def test_allocation_reports_project_and_team(self, tracker: CostTracker) -> None:
        tracker.record_compute(ResourceType.GPU_A100, 1.0, project="fraud", team="ml")
        allocation = tracker.allocation()
        assert allocation["by_project"]["fraud"] == pytest.approx(3.0)
        assert allocation["by_team"]["ml"] == pytest.approx(3.0)

    def test_daily_costs_keyed_by_iso_date(self, tracker: CostTracker) -> None:
        tracker.record_compute(ResourceType.GPU_A100, 1.0)
        today = utcnow().date().isoformat()
        assert tracker.daily_costs()[today] == pytest.approx(3.0)

    def test_track_compute_context_manager_records_on_exit(self, tracker: CostTracker) -> None:
        with tracker.track_compute("train", ResourceType.GPU_A100, project="fraud"):
            pass
        assert len(tracker) == 1
        assert tracker.records()[0].metadata["label"] == "train"

    def test_track_compute_records_even_when_block_raises(self, tracker: CostTracker) -> None:
        with pytest.raises(RuntimeError), tracker.track_compute("train", ResourceType.GPU_A100):
            raise RuntimeError("boom")
        assert len(tracker) == 1

    def test_forecast_extrapolates_daily_rate(self, tracker: CostTracker) -> None:
        for days_ago in range(4):
            tracker.record(
                CostRecord(
                    category=CostCategory.COMPUTE,
                    resource="cpu",
                    quantity=1.0,
                    unit="hours",
                    cost_usd=10.0,
                    timestamp=utcnow() - timedelta(days=days_ago),
                )
            )
        forecast = tracker.forecast(horizon_days=10, lookback_days=7)
        assert forecast.basis_cost_usd == pytest.approx(40.0)
        assert forecast.projected_cost_usd == pytest.approx(forecast.daily_rate_usd * 10)
        assert forecast.to_dict()["horizon_days"] == 10

    def test_forecast_with_no_history_is_zero(self, tracker: CostTracker) -> None:
        forecast = tracker.forecast()
        assert forecast.projected_cost_usd == 0.0
        assert forecast.observed_days == 0.0

    def test_forecast_rejects_bad_windows(self, tracker: CostTracker) -> None:
        with pytest.raises(ValueError):
            tracker.forecast(horizon_days=0)
        with pytest.raises(ValueError):
            tracker.forecast(lookback_days=0)

    def test_ring_buffer_evicts_but_keeps_total(self) -> None:
        tracker = CostTracker(max_records=2)
        for _ in range(4):
            tracker.record_compute(ResourceType.CPU, 1.0, cost_usd=1.0)
        assert len(tracker) == 2
        assert tracker.total_cost_usd() == pytest.approx(4.0)

    def test_max_records_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            CostTracker(max_records=0)

    def test_summary_and_export(self, tracker: CostTracker, tmp_path) -> None:
        tracker.record_compute(ResourceType.GPU_A100, 1.0, project="fraud")
        summary = tracker.summary()
        assert summary["record_count"] == 1
        assert summary["total_cost_usd"] == pytest.approx(3.0)

        path = tracker.export_json(tmp_path / "nested" / "costs.json")
        assert path.is_file()
        assert '"cost_usd"' in path.read_text(encoding="utf-8")

    def test_extend_and_reset(self, tracker: CostTracker) -> None:
        tracker.extend(
            [
                CostRecord(
                    category=CostCategory.OTHER,
                    resource="misc",
                    quantity=1.0,
                    unit="unit",
                    cost_usd=7.0,
                )
            ]
        )
        assert tracker.total_cost_usd() == pytest.approx(7.0)
        tracker.reset()
        assert len(tracker) == 0
        assert tracker.total_cost_usd() == 0.0


# ─── BudgetManager ───────────────────────────────────────────────────────────


class TestBudget:
    """Budget definition and period arithmetic."""

    def test_invalid_definitions_rejected(self) -> None:
        with pytest.raises(ValueError):
            Budget("", 10.0)
        with pytest.raises(ValueError):
            Budget("b", 0.0)
        with pytest.raises(ValueError):
            Budget("b", 10.0, thresholds=())
        with pytest.raises(ValueError):
            Budget("b", 10.0, thresholds=(-0.5,))

    @pytest.mark.parametrize(
        ("period", "expected"),
        [
            (BudgetPeriod.DAILY, datetime(2026, 5, 20, tzinfo=timezone.utc)),
            (BudgetPeriod.WEEKLY, datetime(2026, 5, 18, tzinfo=timezone.utc)),
            (BudgetPeriod.MONTHLY, datetime(2026, 5, 1, tzinfo=timezone.utc)),
            (BudgetPeriod.QUARTERLY, datetime(2026, 4, 1, tzinfo=timezone.utc)),
        ],
    )
    def test_period_start(self, period: BudgetPeriod, expected: datetime) -> None:
        # 2026-05-20 is a Wednesday, so the week starts on Monday the 18th.
        now = datetime(2026, 5, 20, 13, 45, tzinfo=timezone.utc)
        assert Budget("b", 10.0, period).period_start(now) == expected

    def test_period_key_is_stable_within_a_period(self) -> None:
        budget = Budget("b", 10.0, BudgetPeriod.MONTHLY)
        first = datetime(2026, 5, 1, tzinfo=timezone.utc)
        last = datetime(2026, 5, 31, tzinfo=timezone.utc)
        assert budget.period_key(first) == budget.period_key(last)


class TestBudgetManager:
    """Budget evaluation, alerting and enforcement."""

    def test_status_reflects_spend(self, tracker: CostTracker) -> None:
        manager = BudgetManager(tracker)
        manager.add_budget(Budget("monthly", 10.0))
        tracker.record_compute(ResourceType.GPU_A100, 1.0)

        status = manager.status("monthly")
        assert status.spent_usd == pytest.approx(3.0)
        assert status.remaining_usd == pytest.approx(7.0)
        assert status.utilization == pytest.approx(0.3)
        assert not status.exceeded
        assert status.to_dict()["limit_usd"] == 10.0

    def test_unknown_budget_raises(self, tracker: CostTracker) -> None:
        with pytest.raises(KeyError):
            BudgetManager(tracker).status("nope")

    def test_project_scoped_budget_ignores_other_projects(self, tracker: CostTracker) -> None:
        manager = BudgetManager(tracker)
        manager.add_budget(Budget("fraud", 10.0, project="fraud"))
        tracker.record_compute(ResourceType.GPU_A100, 1.0, project="loyalty")
        assert manager.status("fraud").spent_usd == 0.0

    def test_thresholds_fire_once_each(self, tracker: CostTracker) -> None:
        manager = BudgetManager(tracker)
        manager.add_budget(Budget("monthly", 10.0, thresholds=(0.5, 0.8, 1.0)))

        tracker.record_compute(ResourceType.GPU_A100, 2.0)  # 6.0 USD → 60%
        first = manager.evaluate()
        assert [alert.threshold for alert in first] == [0.5]
        assert first[0].severity is AlertSeverity.INFO
        assert manager.evaluate() == []

        tracker.record_compute(ResourceType.GPU_A100, 2.0)  # 12.0 USD → 120%
        second = manager.evaluate()
        assert [alert.threshold for alert in second] == [0.8, 1.0]
        assert second[-1].severity is AlertSeverity.CRITICAL

    def test_subscribers_receive_alerts(self, tracker: CostTracker) -> None:
        manager = BudgetManager(tracker)
        received: list[str] = []
        manager.on_alert(lambda alert: received.append(alert.budget_name))
        manager.add_budget(Budget("monthly", 1.0, thresholds=(1.0,)))
        tracker.record_compute(ResourceType.GPU_A100, 1.0)

        manager.evaluate()
        assert received == ["monthly"]
        assert len(manager.alerts()) == 1

    def test_failing_subscriber_does_not_break_evaluation(self, tracker: CostTracker) -> None:
        manager = BudgetManager(tracker)

        def explode(_alert: object) -> None:
            raise RuntimeError("subscriber failure")

        manager.on_alert(explode)
        manager.add_budget(Budget("monthly", 1.0, thresholds=(1.0,)))
        tracker.record_compute(ResourceType.GPU_A100, 1.0)
        assert len(manager.evaluate()) == 1

    def test_check_spend_allows_within_limit(self, tracker: CostTracker) -> None:
        manager = BudgetManager(tracker)
        manager.add_budget(Budget("monthly", 10.0, hard_limit=True))
        assert len(manager.check_spend(5.0)) == 1

    def test_check_spend_raises_over_hard_limit(self, tracker: CostTracker) -> None:
        manager = BudgetManager(tracker)
        manager.add_budget(Budget("monthly", 10.0, hard_limit=True))
        tracker.record_compute(ResourceType.GPU_A100, 3.0)  # 9.0 USD
        with pytest.raises(BudgetExceededError, match="would be exceeded"):
            manager.check_spend(5.0)

    def test_check_spend_ignores_soft_budgets(self, tracker: CostTracker) -> None:
        manager = BudgetManager(tracker)
        manager.add_budget(Budget("monthly", 1.0, hard_limit=False))
        assert manager.check_spend(100.0)

    def test_check_spend_rejects_negative(self, tracker: CostTracker) -> None:
        with pytest.raises(ValueError):
            BudgetManager(tracker).check_spend(-1.0)

    def test_remove_budget(self, tracker: CostTracker) -> None:
        manager = BudgetManager(tracker)
        manager.add_budget(Budget("monthly", 10.0))
        assert manager.remove_budget("monthly") is True
        assert manager.remove_budget("monthly") is False
        assert manager.get_budget("monthly") is None

    def test_report_aggregates_budgets(self, tracker: CostTracker) -> None:
        manager = BudgetManager(tracker)
        manager.add_budget(Budget("a", 10.0))
        manager.add_budget(Budget("b", 20.0))
        tracker.record_compute(ResourceType.GPU_A100, 1.0)

        report = manager.report()
        assert report["budget_count"] == 2
        assert report["total_limit_usd"] == pytest.approx(30.0)
        assert report["exceeded_count"] == 0

    def test_alert_history_is_bounded(self, tracker: CostTracker) -> None:
        manager = BudgetManager(tracker, max_alert_history=1)
        manager.add_budget(Budget("a", 1.0, thresholds=(0.5, 1.0)))
        tracker.record_compute(ResourceType.GPU_A100, 1.0)
        manager.evaluate()
        assert len(manager.alerts()) == 1

    def test_invalid_history_size_rejected(self, tracker: CostTracker) -> None:
        with pytest.raises(ValueError):
            BudgetManager(tracker, max_alert_history=0)


# ─── ResourceOptimizer ───────────────────────────────────────────────────────


class TestResourceOptimizer:
    """Utilization analysis and recommendations."""

    def test_idle_resource_recommends_shutdown(self, tracker: CostTracker) -> None:
        optimizer = ResourceOptimizer(tracker)
        optimizer.record_many(
            ResourceUsageSample("gpu-0", ResourceType.GPU_A100, 0.01) for _ in range(5)
        )
        [recommendation] = optimizer.recommend()
        assert recommendation.kind is OptimizationKind.SHUTDOWN_IDLE
        assert recommendation.severity is Severity.HIGH
        assert recommendation.estimated_monthly_saving_usd > 0

    def test_underutilized_resource_recommends_rightsizing(self, tracker: CostTracker) -> None:
        optimizer = ResourceOptimizer(tracker)
        optimizer.record_many(
            ResourceUsageSample("gpu-1", ResourceType.GPU_A100, 0.2) for _ in range(5)
        )
        [recommendation] = optimizer.recommend()
        assert recommendation.kind is OptimizationKind.RIGHTSIZE
        assert recommendation.mean_utilization == pytest.approx(0.2)

    def test_saturated_resource_recommends_scale_up(self, tracker: CostTracker) -> None:
        optimizer = ResourceOptimizer(tracker)
        optimizer.record_many(
            ResourceUsageSample("gpu-2", ResourceType.GPU_A100, 0.95) for _ in range(5)
        )
        [recommendation] = optimizer.recommend()
        assert recommendation.kind is OptimizationKind.SCALE_UP
        assert recommendation.estimated_monthly_saving_usd == 0.0

    def test_interruptible_healthy_resource_recommends_spot(self, tracker: CostTracker) -> None:
        optimizer = ResourceOptimizer(tracker)
        optimizer.record_many(
            ResourceUsageSample("gpu-3", ResourceType.GPU_A100, 0.6, interruptible=True)
            for _ in range(5)
        )
        [recommendation] = optimizer.recommend()
        assert recommendation.kind is OptimizationKind.SPOT_INSTANCE

    def test_healthy_resource_yields_no_recommendation(self, tracker: CostTracker) -> None:
        optimizer = ResourceOptimizer(tracker)
        optimizer.record_many(
            ResourceUsageSample("gpu-4", ResourceType.GPU_A100, 0.6) for _ in range(5)
        )
        assert optimizer.recommend() == []

    def test_too_few_samples_are_ignored(self, tracker: CostTracker) -> None:
        optimizer = ResourceOptimizer(tracker)
        optimizer.record(ResourceUsageSample("gpu-5", ResourceType.GPU_A100, 0.0))
        assert optimizer.recommend() == []

    def test_recommendations_ranked_by_saving(self, tracker: CostTracker) -> None:
        optimizer = ResourceOptimizer(tracker)
        optimizer.record_many(ResourceUsageSample("cpu-0", ResourceType.CPU, 0.0) for _ in range(5))
        optimizer.record_many(
            ResourceUsageSample("gpu-0", ResourceType.GPU_A100, 0.0) for _ in range(5)
        )
        savings = [r.estimated_monthly_saving_usd for r in optimizer.recommend()]
        assert savings == sorted(savings, reverse=True)

    def test_utilization_summary(self, tracker: CostTracker) -> None:
        optimizer = ResourceOptimizer(tracker)
        optimizer.record_many(
            [
                ResourceUsageSample("gpu-0", ResourceType.GPU_A100, 0.2),
                ResourceUsageSample("gpu-0", ResourceType.GPU_A100, 0.8),
            ]
        )
        summary = optimizer.utilization_summary()["gpu-0"]
        assert summary["mean"] == pytest.approx(0.5)
        assert summary["peak"] == pytest.approx(0.8)
        assert summary["samples"] == 2.0

    def test_report_totals_savings(self, tracker: CostTracker) -> None:
        optimizer = ResourceOptimizer(tracker)
        optimizer.record_many(
            ResourceUsageSample("gpu-0", ResourceType.GPU_A100, 0.0) for _ in range(5)
        )
        report = optimizer.report()
        assert report["recommendation_count"] == 1
        assert report["estimated_monthly_saving_usd"] > 0

    def test_sample_bounds_are_validated(self) -> None:
        with pytest.raises(ValueError):
            ResourceUsageSample("gpu-0", ResourceType.CPU, 1.5)
        with pytest.raises(ValueError):
            ResourceUsageSample("gpu-0", ResourceType.CPU, 0.5, memory_utilization=-0.1)

    def test_optimizer_settings_validated(self) -> None:
        with pytest.raises(ValueError):
            ResourceOptimizer(spot_discount=0.0)
        with pytest.raises(ValueError):
            ResourceOptimizer(max_samples_per_resource=0)

    def test_sample_buffer_is_bounded(self, tracker: CostTracker) -> None:
        optimizer = ResourceOptimizer(tracker, max_samples_per_resource=3)
        optimizer.record_many(
            ResourceUsageSample("gpu-0", ResourceType.GPU_A100, 0.0) for _ in range(10)
        )
        assert optimizer.utilization_summary()["gpu-0"]["samples"] == 3.0

    def test_reset_clears_samples(self, tracker: CostTracker) -> None:
        optimizer = ResourceOptimizer(tracker)
        optimizer.record(ResourceUsageSample("gpu-0", ResourceType.GPU_A100, 0.0))
        optimizer.reset()
        assert optimizer.utilization_summary() == {}

    def test_falls_back_to_default_prices_without_tracker(self) -> None:
        optimizer = ResourceOptimizer()
        optimizer.record_many(
            ResourceUsageSample("gpu-0", ResourceType.GPU_H100, 0.0) for _ in range(5)
        )
        assert optimizer.recommend()[0].estimated_monthly_saving_usd > 0
