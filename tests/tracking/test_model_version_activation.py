"""Model version activation and rollback (issue #718).

Verifies that ``activate()`` and ``rollback_to_version()`` switch serving to the
expected version and record the transition atomically with lineage.

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


class TestActivate:
    def test_activation_switches_serving_to_the_expected_version(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0", status="deployed")
        _make_version(session, model, "2.0.0")

        activated, previous = registry.activate(model.id, "2.0.0")

        assert activated.version == "2.0.0"
        assert activated.status == "deployed"
        assert previous is not None and previous.version == "1.0.0"
        # Exactly one version serves at a time.
        assert registry.get_latest_deployed_version(model.id).version == "2.0.0"

    def test_the_previous_version_stops_serving(self, session, registry):
        model = _make_model(session)
        old = _make_version(session, model, "1.0.0", status="deployed")
        _make_version(session, model, "2.0.0")

        registry.activate(model.id, "2.0.0")

        session.refresh(old)
        assert old.status != "deployed"

    def test_activation_stamps_deployed_at(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        activated, _ = registry.activate(model.id, "1.0.0")

        assert activated.deployed_at is not None

    def test_activating_the_first_version_has_no_predecessor(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        activated, previous = registry.activate(model.id, "1.0.0")

        assert previous is None
        assert activated.status == "deployed"

    def test_activating_an_unknown_version_raises(self, session, registry):
        model = _make_model(session)

        with pytest.raises(ValueError, match="not found"):
            registry.activate(model.id, "9.9.9")

    def test_activating_the_already_serving_version_raises(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0", status="deployed")

        with pytest.raises(ValueError, match="already deployed"):
            registry.activate(model.id, "1.0.0")


class TestLineage:
    def test_activation_records_lineage_on_the_new_version(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0", status="deployed")
        _make_version(session, model, "2.0.0")

        activated, _ = registry.activate(model.id, "2.0.0", reason="better recall", actor="alice")

        latest = activated.lineage["latest"]
        assert latest["transition"] == "activate"
        assert latest["from_version"] == "1.0.0"
        assert latest["to_version"] == "2.0.0"
        assert latest["reason"] == "better recall"
        assert latest["actor"] == "alice"
        assert latest["at"]

    def test_the_superseded_version_records_the_same_transition(self, session, registry):
        model = _make_model(session)
        old = _make_version(session, model, "1.0.0", status="deployed")
        _make_version(session, model, "2.0.0")

        registry.activate(model.id, "2.0.0", reason="better recall")

        session.refresh(old)
        assert old.lineage["latest"]["role"] == "superseded"
        assert old.lineage["latest"]["to_version"] == "2.0.0"

    def test_lineage_is_persisted_not_just_set_in_memory(self, session, registry):
        # Regression: the registry used to assign to ``version.metadata``, which
        # is SQLAlchemy's reserved MetaData attribute rather than a column, so
        # nothing reached the database.
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        registry.activate(model.id, "1.0.0", reason="first deploy")

        session.expire_all()
        reloaded = registry.get_model_version(model.id, "1.0.0")
        assert reloaded.lineage is not None
        assert reloaded.lineage["latest"]["reason"] == "first deploy"

    def test_lineage_accumulates_across_transitions(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")
        _make_version(session, model, "2.0.0")

        registry.activate(model.id, "1.0.0", reason="first")
        registry.activate(model.id, "2.0.0", reason="second")
        rolled_back, _ = registry.rollback_to_version(model.id, "1.0.0", reason="regression")

        events = rolled_back.lineage["events"]
        assert [e["transition"] for e in events] == ["activate", "activate", "rollback"]
        assert events[-1]["reason"] == "regression"


class TestRollback:
    def test_rollback_switches_serving_back(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")
        _make_version(session, model, "2.0.0")

        registry.activate(model.id, "1.0.0")
        registry.activate(model.id, "2.0.0")

        target, previous = registry.rollback_to_version(model.id, "1.0.0", reason="bad metrics")

        assert target.version == "1.0.0"
        assert target.status == "deployed"
        assert previous.version == "2.0.0"
        assert registry.get_latest_deployed_version(model.id).version == "1.0.0"

    def test_rollback_records_its_own_transition_type(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")
        _make_version(session, model, "2.0.0")
        registry.activate(model.id, "2.0.0")

        target, _ = registry.rollback_to_version(model.id, "1.0.0", reason="bad metrics")

        assert target.lineage["latest"]["transition"] == "rollback"
        assert target.lineage["latest"]["reason"] == "bad metrics"

    def test_rollback_to_unknown_version_raises(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0", status="deployed")

        with pytest.raises(ValueError, match="not found"):
            registry.rollback_to_version(model.id, "0.9.0")

    def test_rollback_to_the_serving_version_raises(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0", status="deployed")

        with pytest.raises(ValueError, match="already deployed"):
            registry.rollback_to_version(model.id, "1.0.0")

    def test_rollback_with_no_previously_deployed_version(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        target, previous = registry.rollback_to_version(model.id, "1.0.0")

        assert previous is None
        assert target.status == "deployed"


class TestAtomicity:
    def test_a_failed_transition_leaves_the_original_version_serving(
        self, session, registry, monkeypatch
    ):
        model = _make_model(session)
        old = _make_version(session, model, "1.0.0", status="deployed")
        new = _make_version(session, model, "2.0.0")

        # Fail at the point where the transition is made durable. Previously the
        # outgoing version was committed *before* the incoming one, so a failure
        # here left the model with no deployed version at all.
        def boom():
            raise RuntimeError("database went away")

        monkeypatch.setattr(session, "commit", boom)

        with pytest.raises(RuntimeError):
            registry.activate(model.id, "2.0.0")

        monkeypatch.undo()
        session.expire_all()

        still_serving = registry.get_latest_deployed_version(model.id)
        assert still_serving is not None, "a failed switch must not leave the model unserved"
        assert still_serving.version == "1.0.0"
        assert registry.get_model_version(model.id, "2.0.0").status != "deployed"

    def test_exactly_one_version_serves_after_a_sequence_of_switches(self, session, registry):
        model = _make_model(session)
        for v in ("1.0.0", "2.0.0", "3.0.0"):
            _make_version(session, model, v)

        registry.activate(model.id, "1.0.0")
        registry.activate(model.id, "2.0.0")
        registry.activate(model.id, "3.0.0")
        registry.rollback_to_version(model.id, "2.0.0")

        deployed = [
            v for v in registry.list_model_versions(model.id) if v.status == "deployed"
        ]
        assert len(deployed) == 1
        assert deployed[0].version == "2.0.0"


class TestIsolationBetweenModels:
    def test_activating_one_model_does_not_touch_another(self, session, registry):
        model_a = _make_model(session, "model-a")
        model_b = _make_model(session, "model-b")
        _make_version(session, model_a, "1.0.0", status="deployed")
        b_v1 = _make_version(session, model_b, "1.0.0", status="deployed")
        _make_version(session, model_a, "2.0.0")

        registry.activate(model_a.id, "2.0.0")

        session.refresh(b_v1)
        assert b_v1.status == "deployed", "another model's serving version must be untouched"
