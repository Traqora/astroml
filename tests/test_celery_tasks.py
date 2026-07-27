"""Tests for Celery tasks — issue #296.

Tasks are executed in *eager* (synchronous) mode so no broker is required
during CI.  We set ``CELERY_TASK_ALWAYS_EAGER=True`` via the Celery app's
``conf.update`` before each test, which makes ``.delay()`` / ``.apply_async()``
run inline and return an ``EagerResult``.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def celery_eager_mode():
    """Force all Celery tasks to execute synchronously (no broker needed)."""
    from astroml.tasks.celery_app import app

    original = app.conf.task_always_eager
    app.conf.update(task_always_eager=True, task_eager_propagates=True)
    yield
    app.conf.update(task_always_eager=original)


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

class TestBuildGraphTask:
    def test_returns_node_and_edge_count(self):
        from astroml.tasks.graph_build_task import build_graph

        account_ids = [
            "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            "GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
            "GCKFBEIYV2U22IO2BJ4KVJOIP7XPWQGQFKKWXR6DOSJBV5SG3B3ORJF",
        ]
        result = build_graph.apply(args=[account_ids, 1000, 2000]).get()

        assert "node_count" in result
        assert "edge_count" in result
        assert result["node_count"] == 3
        assert result["edge_count"] == 2

    def test_echoes_ledger_range(self):
        from astroml.tasks.graph_build_task import build_graph

        result = build_graph.apply(args=[["ACC1", "ACC2"], 500, 999]).get()

        assert result["ledger_from"] == 500
        assert result["ledger_to"] == 999

    def test_empty_account_list(self):
        from astroml.tasks.graph_build_task import build_graph

        result = build_graph.apply(args=[[], 0, 100]).get()

        assert result["node_count"] == 0
        assert result["edge_count"] == 0

    def test_single_account(self):
        from astroml.tasks.graph_build_task import build_graph

        result = build_graph.apply(
            args=[["GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"], 1, 10]
        ).get()

        assert result["node_count"] == 1
        assert result["edge_count"] == 0  # no adjacent pair with single node

    def test_result_is_dict(self):
        from astroml.tasks.graph_build_task import build_graph

        result = build_graph.apply(args=[["A", "B"], 1, 2]).get()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModelTask:
    def test_returns_trained_status(self):
        from astroml.tasks.model_train_task import train_model

        result = train_model.apply(
            args=["link_predictor", {"epochs": 5}]
        ).get()

        assert result["status"] == "trained"

    def test_returns_correct_model_name(self):
        from astroml.tasks.model_train_task import train_model

        result = train_model.apply(
            args=["fraud_gnn", {"epochs": 3}]
        ).get()

        assert result["model_name"] == "fraud_gnn"

    def test_epochs_from_config(self):
        from astroml.tasks.model_train_task import train_model

        result = train_model.apply(
            args=["my_model", {"epochs": 7}]
        ).get()

        assert result["epochs"] == 7

    def test_default_epochs_when_not_provided(self):
        from astroml.tasks.model_train_task import train_model

        result = train_model.apply(args=["default_model", {}]).get()

        assert result["epochs"] == 10

    def test_result_shape(self):
        from astroml.tasks.model_train_task import train_model

        result = train_model.apply(args=["shape_check", {"epochs": 1}]).get()

        assert set(result.keys()) >= {"model_name", "status", "epochs"}

    def test_via_delay(self):
        """Verify .delay() also works in eager mode."""
        from astroml.tasks.model_train_task import train_model

        ar = train_model.delay("delay_test", {"epochs": 2})
        result = ar.get()
        assert result["status"] == "trained"
        assert result["model_name"] == "delay_test"
