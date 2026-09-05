"""Tests for the per-epoch training report (issue #738)."""

from __future__ import annotations

import json

import pytest

from astroml.tracking.training_report import EpochRecord, TrainingReport


class FakeObjective:
    """Stands in for a GraphObjective without pulling Torch into these tests."""

    def __init__(self, name="link_prediction", monitor="auc", higher_is_better=True):
        self.name = name
        self.monitor = monitor
        self.higher_is_better = higher_is_better


class FakeRegistry:
    """Records what a real ``ModelRegistry`` would have been asked to persist."""

    def __init__(self, known_version_ids=(1,)):
        self.known = set(known_version_ids)
        self.calls: list[tuple[int, dict]] = []
        self.stored: dict[int, dict] = {}

    def update_model_version_metrics(self, version_id: int, metrics: dict):
        self.calls.append((version_id, metrics))
        if version_id not in self.known:
            return None
        self.stored.setdefault(version_id, {}).update(metrics)
        return {"id": version_id, "metrics": self.stored[version_id]}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_takes_monitor_and_direction_from_the_objective(self):
        report = TrainingReport(FakeObjective(monitor="auc", higher_is_better=True))

        assert report.monitor == "auc"
        assert report.higher_is_better is True
        assert report.objective_name == "link_prediction"

    def test_explicit_arguments_override_the_objective(self):
        report = TrainingReport(
            FakeObjective(monitor="auc", higher_is_better=True),
            monitor="loss",
            higher_is_better=False,
        )

        assert report.monitor == "loss"
        assert report.higher_is_better is False

    def test_works_without_an_objective(self):
        report = TrainingReport()

        assert report.monitor == "loss"
        assert report.higher_is_better is False
        assert report.objective_name == "unknown"

    def test_rejects_a_retention_cap_below_two(self):
        with pytest.raises(ValueError, match="at least 2"):
            TrainingReport(max_epochs_retained=1)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class TestRecording:
    def test_records_train_and_validation_metrics(self):
        report = TrainingReport(FakeObjective())

        record = report.record(1, train={"loss": 0.5}, val={"auc": 0.8}, duration_seconds=1.25)

        assert record == EpochRecord(
            epoch=1, train={"loss": 0.5}, val={"auc": 0.8}, duration_seconds=1.25
        )
        assert len(report) == 1

    def test_validation_metrics_are_optional(self):
        report = TrainingReport(FakeObjective())

        record = report.record(1, train={"loss": 0.5})

        assert record.val == {}

    def test_rejects_a_zero_or_negative_epoch(self):
        report = TrainingReport(FakeObjective())

        with pytest.raises(ValueError, match="1-based"):
            report.record(0, train={"loss": 1.0})

    def test_coerces_numeric_types_to_float(self):
        """Torch and NumPy scalars do not survive JSON serialisation."""
        report = TrainingReport(FakeObjective())

        record = report.record(1, train={"loss": 1}, val={"auc": "0.75"})

        assert record.train["loss"] == 1.0
        assert isinstance(record.train["loss"], float)
        assert record.val["auc"] == 0.75

    def test_drops_values_that_are_not_numbers(self):
        report = TrainingReport(FakeObjective())

        record = report.record(1, train={"loss": 0.5, "note": "diverged"})

        assert record.train == {"loss": 0.5}

    def test_a_torch_scalar_is_stored_as_a_plain_float(self):
        torch = pytest.importorskip("torch")
        report = TrainingReport(FakeObjective())

        record = report.record(1, train={"loss": torch.tensor(0.25)})

        assert record.train["loss"] == pytest.approx(0.25)
        assert isinstance(record.train["loss"], float)


# ---------------------------------------------------------------------------
# Best epoch
# ---------------------------------------------------------------------------


class TestBestEpoch:
    def test_picks_the_highest_value_when_higher_is_better(self):
        report = TrainingReport(FakeObjective(monitor="auc", higher_is_better=True))
        report.record(1, val={"auc": 0.70})
        report.record(2, val={"auc": 0.85})
        report.record(3, val={"auc": 0.80})

        assert report.best.epoch == 2

    def test_picks_the_lowest_value_when_lower_is_better(self):
        report = TrainingReport(FakeObjective(monitor="loss", higher_is_better=False))
        report.record(1, val={"loss": 0.9})
        report.record(2, val={"loss": 0.3})
        report.record(3, val={"loss": 0.5})

        assert report.best.epoch == 2

    def test_prefers_validation_over_training(self):
        """ "Best" must mean best on held-out data when there is any."""
        report = TrainingReport(FakeObjective(monitor="auc", higher_is_better=True))
        report.record(1, train={"auc": 0.99}, val={"auc": 0.60})
        report.record(2, train={"auc": 0.70}, val={"auc": 0.90})

        assert report.best.epoch == 2

    def test_falls_back_to_training_when_there_is_no_validation(self):
        report = TrainingReport(FakeObjective(monitor="auc", higher_is_better=True))
        report.record(1, train={"auc": 0.6})
        report.record(2, train={"auc": 0.8})

        assert report.best.epoch == 2

    def test_ties_resolve_to_the_earlier_epoch(self):
        """The earlier epoch is the checkpoint early stopping would have kept."""
        report = TrainingReport(FakeObjective(monitor="auc", higher_is_better=True))
        report.record(1, val={"auc": 0.8})
        report.record(2, val={"auc": 0.8})

        assert report.best.epoch == 1

    def test_is_none_when_nothing_was_recorded(self):
        assert TrainingReport(FakeObjective()).best is None

    def test_ignores_epochs_missing_the_monitored_metric(self):
        report = TrainingReport(FakeObjective(monitor="auc", higher_is_better=True))
        report.record(1, train={"loss": 0.5})
        report.record(2, val={"auc": 0.7})

        assert report.best.epoch == 2

    def test_is_none_when_no_epoch_has_the_monitored_metric(self):
        report = TrainingReport(FakeObjective(monitor="auc", higher_is_better=True))
        report.record(1, train={"loss": 0.5})

        assert report.best is None

    def test_latest_is_the_most_recent_epoch(self):
        report = TrainingReport(FakeObjective())
        report.record(1, val={"auc": 0.9})
        report.record(2, val={"auc": 0.4})

        assert report.latest.epoch == 2


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


class TestRetention:
    def test_keeps_everything_under_the_cap(self):
        report = TrainingReport(FakeObjective(), max_epochs_retained=10)
        for epoch in range(1, 6):
            report.record(epoch, train={"loss": 1.0 / epoch})

        assert len(report) == 5
        assert report.dropped_epochs == 0

    def test_bounds_what_a_long_run_writes_to_the_registry(self):
        report = TrainingReport(FakeObjective(), max_epochs_retained=5)
        for epoch in range(1, 51):
            report.record(epoch, train={"loss": 1.0 / epoch})

        assert len(report) == 5
        assert report.dropped_epochs == 45

    def test_keeps_the_first_epoch_as_the_baseline(self):
        """Dropping epoch 1 turns a convergence curve into an unanchored tail."""
        report = TrainingReport(FakeObjective(), max_epochs_retained=4)
        for epoch in range(1, 21):
            report.record(epoch, train={"loss": 1.0 / epoch})

        epochs = [record.epoch for record in report.records]
        assert epochs[0] == 1
        assert epochs[-1] == 20

    def test_keeps_the_most_recent_epochs(self):
        report = TrainingReport(FakeObjective(), max_epochs_retained=4)
        for epoch in range(1, 11):
            report.record(epoch, train={"loss": 1.0 / epoch})

        assert [record.epoch for record in report.records] == [1, 8, 9, 10]

    def test_the_recorded_total_survives_pruning(self):
        report = TrainingReport(FakeObjective(), max_epochs_retained=3)
        for epoch in range(1, 21):
            report.record(epoch, train={"loss": 0.1})

        assert report.to_dict()["epochs_recorded"] == 20


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_dict_is_json_serialisable(self):
        report = TrainingReport(FakeObjective())
        report.record(1, train={"loss": 0.5}, val={"auc": 0.7}, duration_seconds=0.5)

        # A report that cannot be written is worse than no report: the failure
        # lands at commit time, long after the run finished.
        assert json.loads(json.dumps(report.to_dict()))

    def test_to_dict_carries_the_objective_and_monitor(self):
        report = TrainingReport(FakeObjective(name="node_classification", monitor="accuracy"))
        report.record(1, val={"accuracy": 0.6})

        payload = report.to_dict()

        assert payload["objective"] == "node_classification"
        assert payload["monitor"] == "accuracy"
        assert payload["best_epoch"] == 1

    def test_history_holds_one_entry_per_retained_epoch(self):
        report = TrainingReport(FakeObjective())
        report.record(1, train={"loss": 0.5}, val={"auc": 0.7})
        report.record(2, train={"loss": 0.4}, val={"auc": 0.8})

        history = report.to_dict()["history"]

        assert [entry["epoch"] for entry in history] == [1, 2]
        assert history[1]["val"] == {"auc": 0.8}

    def test_an_epoch_without_validation_omits_the_key(self):
        report = TrainingReport(FakeObjective())
        report.record(1, train={"loss": 0.5})

        assert "val" not in report.to_dict()["history"][0]

    def test_total_duration_sums_the_epochs(self):
        report = TrainingReport(FakeObjective())
        report.record(1, train={"loss": 1.0}, duration_seconds=1.5)
        report.record(2, train={"loss": 0.9}, duration_seconds=2.5)

        assert report.to_dict()["total_duration_seconds"] == pytest.approx(4.0)


class TestSummaryMetrics:
    def test_prefixes_best_and_final_values(self):
        report = TrainingReport(FakeObjective(monitor="auc", higher_is_better=True))
        report.record(1, train={"loss": 0.9}, val={"auc": 0.60})
        report.record(2, train={"loss": 0.5}, val={"auc": 0.90})
        report.record(3, train={"loss": 0.4}, val={"auc": 0.70})

        summary = report.summary_metrics()

        assert summary["best_epoch"] == 2.0
        assert summary["best_val_auc"] == 0.90
        assert summary["best_train_loss"] == 0.5
        # "final" is the last epoch, which is not the best one here.
        assert summary["final_val_auc"] == 0.70
        assert summary["final_train_loss"] == 0.4

    def test_is_flat_so_versions_can_be_compared_without_parsing(self):
        report = TrainingReport(FakeObjective())
        report.record(1, train={"loss": 0.5}, val={"auc": 0.8})

        assert all(isinstance(value, float) for value in report.summary_metrics().values())

    def test_is_empty_before_anything_is_recorded(self):
        assert TrainingReport(FakeObjective()).summary_metrics() == {}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersist:
    def test_writes_summary_and_history_to_the_registry(self):
        registry = FakeRegistry(known_version_ids=(7,))
        report = TrainingReport(FakeObjective(monitor="auc", higher_is_better=True))
        report.record(1, train={"loss": 0.9}, val={"auc": 0.6}, duration_seconds=1.0)
        report.record(2, train={"loss": 0.4}, val={"auc": 0.9}, duration_seconds=1.0)

        result = report.persist(registry, version_id=7)

        assert result is not None
        version_id, payload = registry.calls[0]
        assert version_id == 7
        assert payload["best_val_auc"] == 0.9
        assert payload["training_report"]["best_epoch"] == 2
        assert len(payload["training_report"]["history"]) == 2

    def test_uses_the_metrics_column_not_the_reserved_metadata_attribute(self):
        """``ModelVersion`` has no ``metadata`` column.

        ``metadata`` is SQLAlchemy's reserved ``MetaData`` on the declarative
        base — the same defect that moved serving lineage to its own column in
        #718 — so ``update_model_version_metadata`` writes nowhere. The report
        must not depend on it.
        """
        registry = FakeRegistry(known_version_ids=(1,))
        report = TrainingReport(FakeObjective())
        report.record(1, val={"auc": 0.5})

        report.persist(registry, version_id=1)

        assert not hasattr(registry, "update_model_version_metadata_called")
        assert len(registry.calls) == 1

    def test_history_is_nested_so_it_cannot_collide_with_caller_metrics(self):
        registry = FakeRegistry(known_version_ids=(1,))
        report = TrainingReport(FakeObjective())
        report.record(1, val={"auc": 0.5})

        report.persist(registry, version_id=1)

        _, payload = registry.calls[0]
        assert isinstance(payload["training_report"], dict)
        assert "history" in payload["training_report"]

    def test_the_payload_is_json_serialisable(self):
        registry = FakeRegistry(known_version_ids=(1,))
        report = TrainingReport(FakeObjective())
        report.record(1, train={"loss": 0.5}, val={"auc": 0.7}, duration_seconds=0.25)

        report.persist(registry, version_id=1)

        _, payload = registry.calls[0]
        assert json.loads(json.dumps(payload))

    def test_returns_none_for_an_unknown_version(self):
        registry = FakeRegistry(known_version_ids=(1,))
        report = TrainingReport(FakeObjective())
        report.record(1, val={"auc": 0.5})

        assert report.persist(registry, version_id=999) is None

    def test_an_empty_report_still_persists_without_error(self):
        registry = FakeRegistry(known_version_ids=(1,))

        TrainingReport(FakeObjective()).persist(registry, version_id=1)

        _, payload = registry.calls[0]
        assert payload["training_report"]["epochs_recorded"] == 0


class TestRegistryIntegration:
    """Against the real registry, on an in-memory database."""

    def test_round_trips_through_the_real_model_registry(self, tmp_path):
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from astroml.db.schema import Base
        from astroml.tracking.model_registry import ModelRegistry

        engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
        Base.metadata.create_all(engine)
        session = sqlalchemy.orm.Session(engine)

        try:
            registry = ModelRegistry(session=session)
            model = registry.create_model(
                name="graph-model", framework="pytorch", task_type="classification"
            )
            version = registry.create_model_version(
                model_id=model.id, artifact_path="/tmp/model.pt"
            )

            report = TrainingReport(FakeObjective(monitor="auc", higher_is_better=True))
            report.record(1, train={"loss": 0.9}, val={"auc": 0.55}, duration_seconds=0.5)
            report.record(2, train={"loss": 0.3}, val={"auc": 0.88}, duration_seconds=0.5)

            report.persist(registry, version_id=version.id)

            stored = registry.get_model_version_by_id(version.id)
            assert stored.metrics["best_val_auc"] == 0.88
            assert stored.metrics["training_report"]["best_epoch"] == 2
            assert len(stored.metrics["training_report"]["history"]) == 2
        finally:
            session.close()
            engine.dispose()


class TestRegistryMetricsPersistence:
    """Regression test for ``update_model_version_metrics`` (issue #738).

    The method mutated ``version.metrics`` in place. ``metrics`` is a plain
    JSON column compared by identity, so the instance was never marked dirty,
    ``commit`` wrote nothing, and the following ``refresh`` reloaded the old
    value over the change — silently, with a version object returned as if it
    had worked.
    """

    @staticmethod
    def _registry(tmp_path):
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from astroml.db.schema import Base
        from astroml.tracking.model_registry import ModelRegistry

        engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'metrics.db'}")
        Base.metadata.create_all(engine)
        session = sqlalchemy.orm.Session(engine)
        return ModelRegistry(session=session), session, engine

    def test_an_update_survives_the_commit(self, tmp_path):
        registry, session, engine = self._registry(tmp_path)
        try:
            model = registry.create_model(name="m", framework="pytorch", task_type="classification")
            version = registry.create_model_version(
                model_id=model.id, artifact_path="/tmp/a.pt", metrics={"seed": 1.0}
            )

            registry.update_model_version_metrics(version.id, {"auc": 0.9})

            stored = registry.get_model_version_by_id(version.id)
            assert stored.metrics["auc"] == 0.9
        finally:
            session.close()
            engine.dispose()

    def test_an_update_merges_rather_than_replaces(self, tmp_path):
        registry, session, engine = self._registry(tmp_path)
        try:
            model = registry.create_model(name="m", framework="pytorch", task_type="classification")
            version = registry.create_model_version(
                model_id=model.id, artifact_path="/tmp/a.pt", metrics={"seed": 1.0}
            )

            registry.update_model_version_metrics(version.id, {"auc": 0.9})

            stored = registry.get_model_version_by_id(version.id)
            assert stored.metrics == {"seed": 1.0, "auc": 0.9}
        finally:
            session.close()
            engine.dispose()

    def test_successive_updates_accumulate(self, tmp_path):
        registry, session, engine = self._registry(tmp_path)
        try:
            model = registry.create_model(name="m", framework="pytorch", task_type="classification")
            version = registry.create_model_version(model_id=model.id, artifact_path="/tmp/a.pt")

            registry.update_model_version_metrics(version.id, {"first": 1.0})
            registry.update_model_version_metrics(version.id, {"second": 2.0})

            stored = registry.get_model_version_by_id(version.id)
            assert stored.metrics["first"] == 1.0
            assert stored.metrics["second"] == 2.0
        finally:
            session.close()
            engine.dispose()

    def test_the_change_is_visible_from_a_fresh_session(self, tmp_path):
        """Reading through the identity map could mask a missing write."""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from astroml.db.schema import Base, ModelVersion
        from astroml.tracking.model_registry import ModelRegistry

        engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
        Base.metadata.create_all(engine)

        writer = sqlalchemy.orm.Session(engine)
        try:
            registry = ModelRegistry(session=writer)
            model = registry.create_model(name="m", framework="pytorch", task_type="classification")
            version = registry.create_model_version(model_id=model.id, artifact_path="/tmp/a.pt")
            registry.update_model_version_metrics(version.id, {"auc": 0.75})
            version_id = version.id
        finally:
            writer.close()

        reader = sqlalchemy.orm.Session(engine)
        try:
            reloaded = reader.get(ModelVersion, version_id)
            assert reloaded.metrics["auc"] == 0.75
        finally:
            reader.close()
            engine.dispose()
