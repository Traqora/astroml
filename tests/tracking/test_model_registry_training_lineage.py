"""Model lineage tracking on the DB-backed registry.

Verifies that ``ModelRegistry`` records training provenance and parent/child
version links into the ``ModelVersion.lineage`` JSON column, reconstructs them
as :class:`ModelLineage`, and keeps serving-transition history intact.

The registry is driven against an in-memory SQLite database so the behaviour
under test is the real ORM write path, not a mock of it.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from astroml.db.models import Base, DbModel, ModelVersion
from astroml.tracking.model_registry import ModelRegistry


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture()
def registry(session: Session) -> ModelRegistry:
    return ModelRegistry(session=session)


def _make_model(session: Session, name: str = "fraud-detector") -> DbModel:
    model = DbModel(name=name, framework="pytorch", task_type="anomaly_detection")
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


def _make_version(
    session: Session,
    model: DbModel,
    version: str,
    status: str = "trained",
) -> ModelVersion:
    mv = ModelVersion(
        model_id=model.id,
        version=version,
        artifact_path=f"s3://models/{model.name}/{version}",
        status=status,
    )
    session.add(mv)
    session.commit()
    session.refresh(mv)
    return mv


class TestRecordTrainingLineage:
    def test_records_training_event_on_the_version(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        updated = registry.record_training_lineage(
            model.id,
            "1.0.0",
            dataset_id="stellar_tx_2026_q1",
            dataset_version="v2.1",
            commit_hash="a1b2c3d",
            hyperparameters={"batch_size": 64, "lr": 0.001},
        )

        assert updated is not None
        events = updated.lineage["events"]
        assert len(events) == 1
        assert events[0]["type"] == "training"
        assert events[0]["dataset_id"] == "stellar_tx_2026_q1"
        assert events[0]["commit_hash"] == "a1b2c3d"
        assert events[0]["hyperparameters"] == {"batch_size": 64, "lr": 0.001}
        assert updated.lineage["latest"]["dataset_version"] == "v2.1"

    def test_lineage_survives_reload(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        registry.record_training_lineage(model.id, "1.0.0", dataset_id="d1", commit_hash="abc")

        session.expire_all()
        reloaded = registry.get_model_version(model.id, "1.0.0")
        assert reloaded.lineage["events"][0]["dataset_id"] == "d1"

    def test_events_are_append_only_and_latest_wins(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        registry.record_training_lineage(model.id, "1.0.0", dataset_id="d1", commit_hash="abc")
        registry.record_training_lineage(model.id, "1.0.0", dataset_id="d2", commit_hash="def")

        updated = registry.get_model_version(model.id, "1.0.0")
        events = updated.lineage["events"]
        assert [e["dataset_id"] for e in events] == ["d1", "d2"]
        assert updated.lineage["latest"]["dataset_id"] == "d2"

    def test_unknown_version_returns_none(self, registry):
        assert registry.record_training_lineage(999, "1.0.0", dataset_id="d1") is None


class TestGetModelLineage:
    def test_reconstructs_model_lineage_after_training_record(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        registry.record_training_lineage(
            model.id,
            "1.0.0",
            dataset_id="stellar_tx_2026_q1",
            dataset_version="v2.1",
            commit_hash="a1b2c3d",
            hyperparameters={"batch_size": 64},
            environment={"python": "3.11"},
        )

        lineage = registry.get_model_lineage(model.id, "1.0.0")
        assert lineage is not None
        assert lineage.model_name == model.name
        assert lineage.version == "1.0.0"
        assert lineage.training_lineage.dataset_id == "stellar_tx_2026_q1"
        assert lineage.training_lineage.commit_hash == "a1b2c3d"
        assert lineage.training_lineage.hyperparameters == {"batch_size": 64}
        assert "stellar_tx_2026_q1" in lineage.upstream_nodes

    def test_returns_none_when_no_training_recorded(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        assert registry.get_model_lineage(model.id, "1.0.0") is None

    def test_returns_none_for_unknown_version(self, registry):
        assert registry.get_model_lineage(999, "1.0.0") is None

    def test_upstream_includes_parent_when_derived(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        registry.record_training_lineage(
            model.id,
            "1.0.0",
            dataset_id="d1",
            parent_model_id=model.id,
            parent_version="0.9.0",
        )

        lineage = registry.get_model_lineage(model.id, "1.0.0")
        assert f"{model.id}:0.9.0" in lineage.upstream_nodes

    def test_does_not_disturb_serving_transition_events(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        registry.activate(model.id, "1.0.0", reason="go live")
        registry.record_training_lineage(model.id, "1.0.0", dataset_id="d1")

        version = registry.get_model_version(model.id, "1.0.0")
        events = version.lineage["events"]
        assert events[0]["transition"] == "activate"
        assert events[1]["type"] == "training"
        lineage = registry.get_model_lineage(model.id, "1.0.0")
        assert lineage.training_lineage.dataset_id == "d1"


class TestChildVersions:
    def test_lists_children_recorded_via_training_lineage(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")
        _make_version(session, model, "2.0.0")

        registry.record_training_lineage(
            model.id,
            "2.0.0",
            dataset_id="d2",
            parent_model_id=model.id,
            parent_version="1.0.0",
        )

        children = registry.list_child_versions(model.id, "1.0.0")
        assert [c.version for c in children] == ["2.0.0"]

    def test_respects_the_parent_version_field(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")
        _make_version(session, model, "1.1.0")
        _make_version(session, model, "2.0.0")

        registry.record_training_lineage(
            model.id,
            "2.0.0",
            dataset_id="d2",
            parent_model_id=model.id,
            parent_version="1.1.0",
        )

        assert [c.version for c in registry.list_child_versions(model.id, "1.0.0")] == []
        assert [c.version for c in registry.list_child_versions(model.id, "1.1.0")] == ["2.0.0"]


class TestCreateVersionParentLink:
    def test_create_version_records_fork_event(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        new_version = registry.create_model_version(
            model_id=model.id,
            artifact_path="/tmp/model.pth",
            status="trained",
            parent_model_id=model.id,
            parent_version="1.0.0",
        )

        fork_events = [e for e in new_version.lineage["events"] if e["type"] == "fork"]
        assert len(fork_events) == 1
        assert fork_events[0]["parent_model_id"] == model.id
        assert fork_events[0]["parent_version"] == "1.0.0"

        children = registry.list_child_versions(model.id, "1.0.0")
        assert [c.version for c in children] == [new_version.version]

    def test_create_version_without_parent_has_no_fork_event(self, session, registry):
        model = _make_model(session)

        new_version = registry.create_model_version(
            model_id=model.id,
            artifact_path="/tmp/model.pth",
            status="trained",
        )

        assert new_version.lineage is None

    def test_rejects_partial_parent_reference(self, session, registry):
        model = _make_model(session)

        with pytest.raises(ValueError, match="Both parent_model_id and parent_version"):
            registry.create_model_version(
                model_id=model.id,
                artifact_path="/tmp/model.pth",
                status="trained",
                parent_model_id=model.id,
            )

    def test_rejects_unknown_parent_version(self, session, registry):
        model = _make_model(session)

        with pytest.raises(ValueError, match="Parent version '9.9.9' not found"):
            registry.create_model_version(
                model_id=model.id,
                artifact_path="/tmp/model.pth",
                status="trained",
                parent_model_id=model.id,
                parent_version="9.9.9",
            )

    def test_downstream_nodes_wired_in_model_lineage(self, session, registry):
        model = _make_model(session)
        parent = registry.create_model_version(
            model_id=model.id,
            artifact_path="/tmp/v1.pth",
            status="trained",
        )
        registry.record_training_lineage(model.id, parent.version, dataset_id="d1")

        child = registry.create_model_version(
            model_id=model.id,
            artifact_path="/tmp/v2.pth",
            status="trained",
            parent_model_id=model.id,
            parent_version=parent.version,
        )
        registry.record_training_lineage(
            model.id,
            child.version,
            dataset_id="d2",
            parent_model_id=model.id,
            parent_version=parent.version,
        )

        lineage = registry.get_model_lineage(model.id, parent.version)
        assert lineage.downstream_nodes == [f"{model.name}:{child.version}"]


class TestVersionHistoryExposesLineage:
    def test_history_includes_lineage_column(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")
        registry.record_training_lineage(model.id, "1.0.0", dataset_id="d1")

        entry = registry.get_version_history(model.id)[0]
        assert entry["lineage"]["latest"]["dataset_id"] == "d1"
