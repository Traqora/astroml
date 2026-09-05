"""MLflow run-to-registry-version linkage (issue #764).

Verifies that:
- ``mlflow_run_id`` is stored and retrieved on ``ModelVersion``
- ``get_mlflow_run_details()`` returns the linked run data
- Version history and comparison include the run ID
- The API endpoint serves MLflow run details
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
    mlflow_run_id: str | None = None,
) -> ModelVersion:
    mv = ModelVersion(
        model_id=model.id,
        version=version,
        artifact_path=f"s3://models/{model.name}/{version}",
        status=status,
        mlflow_run_id=mlflow_run_id,
    )
    session.add(mv)
    session.commit()
    session.refresh(mv)
    return mv


class TestMlflowRunIdStorage:
    def test_mlflow_run_id_is_persisted_on_version(self, session, registry):
        model = _make_model(session)
        mv = _make_version(session, model, "1.0.0", mlflow_run_id="run-abc-123")

        assert mv.mlflow_run_id == "run-abc-123"

        session.expire_all()
        reloaded = registry.get_model_version(model.id, "1.0.0")
        assert reloaded.mlflow_run_id == "run-abc-123"

    def test_mlflow_run_id_defaults_to_none(self, session):
        model = _make_model(session)
        mv = _make_version(session, model, "1.0.0")

        assert mv.mlflow_run_id is None

    def test_create_model_version_with_mlflow_run_id(self, session, registry):
        model = _make_model(session)
        mv = registry.create_model_version(
            model_id=model.id,
            artifact_path="/tmp/model.pth",
            mlflow_run_id="run-xyz-789",
            status="trained",
        )

        assert mv.mlflow_run_id == "run-xyz-789"
        reloaded = registry.get_model_version(model.id, mv.version)
        assert reloaded.mlflow_run_id == "run-xyz-789"

    def test_mlflow_run_id_survives_status_transition(self, session, registry):
        model = _make_model(session)
        mv = _make_version(session, model, "1.0.0", mlflow_run_id="run-abc")

        registry.activate(model.id, "1.0.0")

        session.expire_all()
        reloaded = registry.get_model_version(model.id, "1.0.0")
        assert reloaded.mlflow_run_id == "run-abc"
        assert reloaded.status == "deployed"


class TestGetMlflowRunDetails:
    def test_returns_none_when_no_run_id(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        result = registry.get_mlflow_run_details(model.id, "1.0.0")
        assert result is None

    def test_returns_none_for_unknown_version(self, session, registry):
        model = _make_model(session)

        result = registry.get_mlflow_run_details(model.id, "9.9.9")
        assert result is None

    @patch("astroml.tracking.model_registry.mlflow", create=True)
    def test_returns_run_data_when_mlflow_available(self, mock_mlflow, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0", mlflow_run_id="run-abc-123")

        fake_run = MagicMock()
        fake_run.info.run_id = "run-abc-123"
        fake_run.info.experiment_id = "exp-1"
        fake_run.info.status = "FINISHED"
        fake_run.info.start_time = 1000
        fake_run.info.end_time = 2000
        fake_run.info.artifact_uri = "mlflow/artifacts"
        fake_run.data.metrics = {"auc": 0.95}
        fake_run.data.params = {"lr": 0.001}
        fake_run.data.tags = {"user": "alice"}

        mock_mlflow.get_run.return_value = fake_run

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            result = registry.get_mlflow_run_details(model.id, "1.0.0")

        assert result is not None
        assert result["run_id"] == "run-abc-123"
        assert result["experiment_id"] == "exp-1"
        assert result["status"] == "FINISHED"
        assert result["metrics"] == {"auc": 0.95}
        assert result["params"] == {"lr": 0.001}
        assert result["tags"] == {"user": "alice"}

    def test_returns_none_when_mlflow_not_installed(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0", mlflow_run_id="run-abc")

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mlflow":
                raise ImportError("No module named 'mlflow'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = registry.get_mlflow_run_details(model.id, "1.0.0")

        assert result is None

    def test_returns_none_when_mlflow_run_not_found(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0", mlflow_run_id="run-nonexistent")

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mlflow":
                mock = MagicMock()
                mock.get_run.side_effect = Exception("Run not found")
                return mock
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = registry.get_mlflow_run_details(model.id, "1.0.0")

        assert result is None


class TestVersionHistoryIncludesRunId:
    def test_version_history_includes_mlflow_run_id(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0", mlflow_run_id="run-aaa")
        _make_version(session, model, "2.0.0", mlflow_run_id="run-bbb")

        history = registry.get_version_history(model.id)

        run_ids = {v["version"]: v["mlflow_run_id"] for v in history}
        assert run_ids["1.0.0"] == "run-aaa"
        assert run_ids["2.0.0"] == "run-bbb"

    def test_version_history_shows_none_when_no_run_id(self, session, registry):
        model = _make_model(session)
        _make_version(session, model, "1.0.0")

        history = registry.get_version_history(model.id)
        assert history[0]["mlflow_run_id"] is None


class TestCompareVersionsIncludesRunId:
    def test_comparison_includes_mlflow_run_id(self, session, registry):
        model = _make_model(session)
        v1 = _make_version(session, model, "1.0.0", mlflow_run_id="run-1")
        v2 = _make_version(session, model, "2.0.0", mlflow_run_id="run-2")

        comparison = registry.compare_versions([v1.id, v2.id])

        version_info = {v["version"]: v for v in comparison["versions"]}
        assert version_info["1.0.0"]["mlflow_run_id"] == "run-1"
        assert version_info["2.0.0"]["mlflow_run_id"] == "run-2"


class TestUpdateVersionStatusWithRunId:
    def test_set_mlflow_run_id_via_update_kwargs(self, session, registry):
        model = _make_model(session)
        mv = _make_version(session, model, "1.0.0")

        updated = registry.update_model_version_status(
            mv.id,
            "trained",
            validate_transition=True,
            mlflow_run_id="run-late-link",
        )

        assert updated.mlflow_run_id == "run-late-link"
        reloaded = registry.get_model_version(model.id, "1.0.0")
        assert reloaded.mlflow_run_id == "run-late-link"
