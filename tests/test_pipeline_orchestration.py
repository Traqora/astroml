"""Unit tests for pipeline orchestration, DAG execution, sensors, and Airflow/Dagster integration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from astroml.pipeline.dags import (
    create_deployment_dag,
    create_evaluation_dag,
    create_training_dag,
)
from astroml.pipeline.orchestrator import (
    AirflowOrchestrator,
    DAGDefinition,
    DagsterOrchestrator,
    LocalPipelineOrchestrator,
    PipelineRunStatus,
    PipelineTask,
    RetryPolicy,
    TaskStatus,
)
from astroml.pipeline.sensors import (
    DataArrivalSensor,
    DriftSensor,
    FileArrivalSensor,
    ModelPerformanceSensor,
)


class TestDAGAndOrchestrator:
    def test_topological_sort_and_execution(self):
        dag = DAGDefinition("test_dag")
        execution_order = []

        t1 = PipelineTask("step1", lambda ctx: execution_order.append("step1"))
        t2 = PipelineTask(
            "step2", lambda ctx: execution_order.append("step2"), upstream_task_ids={"step1"}
        )
        t3 = PipelineTask(
            "step3", lambda ctx: execution_order.append("step3"), upstream_task_ids={"step2"}
        )

        dag.add_task(t1)
        dag.add_task(t2)
        dag.add_task(t3)

        assert dag.topological_sort() == ["step1", "step2", "step3"]

        orch = LocalPipelineOrchestrator()
        run = orch.run_dag(dag)
        assert run.status == PipelineRunStatus.COMPLETED
        assert execution_order == ["step1", "step2", "step3"]

    def test_cycle_detection(self):
        dag = DAGDefinition("cycle_dag")
        t1 = PipelineTask("task_a", lambda ctx: None, upstream_task_ids={"task_b"})
        t2 = PipelineTask("task_b", lambda ctx: None, upstream_task_ids={"task_a"})
        dag.add_task(t1)
        dag.add_task(t2)

        with pytest.raises(ValueError, match="Cycle detected"):
            dag.topological_sort()

    def test_retry_mechanism(self):
        attempts = 0

        def flaky_action(ctx):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Transient error")
            return "recovered"

        dag = DAGDefinition("retry_dag")
        task = PipelineTask(
            "flaky_step",
            flaky_action,
            retry_policy=RetryPolicy(max_retries=3, delay_seconds=0.01, backoff_factor=1.5),
        )
        dag.add_task(task)

        orch = LocalPipelineOrchestrator()
        run = orch.run_dag(dag)

        assert run.status == PipelineRunStatus.COMPLETED
        assert run.task_results["flaky_step"].status == TaskStatus.SUCCESS
        assert run.task_results["flaky_step"].attempts == 3
        assert run.context.get("flaky_step_output") == "recovered"

    def test_upstream_failure_skips_downstream(self):
        dag = DAGDefinition("failure_dag")
        t1 = PipelineTask(
            "failing_task", lambda ctx: 1 / 0, retry_policy=RetryPolicy(max_retries=0)
        )
        t2 = PipelineTask("downstream_task", lambda ctx: "done", upstream_task_ids={"failing_task"})

        dag.add_task(t1)
        dag.add_task(t2)

        orch = LocalPipelineOrchestrator()
        run = orch.run_dag(dag)

        assert run.status == PipelineRunStatus.FAILED
        assert run.task_results["failing_task"].status == TaskStatus.FAILED
        assert run.task_results["downstream_task"].status == TaskStatus.SKIPPED

    def test_operator_overloading_chaining(self):
        t1 = PipelineTask("t1", lambda ctx: None)
        t2 = PipelineTask("t2", lambda ctx: None)
        t3 = PipelineTask("t3", lambda ctx: None)

        t1 >> t2 >> t3
        assert "t1" in t2.upstream_task_ids
        assert "t2" in t3.upstream_task_ids


class TestPipelineSensors:
    def test_data_arrival_sensor(self):
        sensor = DataArrivalSensor(
            check_fn=lambda: 1500,
            min_records=1000,
            poke_interval_seconds=0.01,
            timeout_seconds=0.1,
        )
        res = sensor.poke()
        assert res.triggered is True
        assert res.payload["record_count"] == 1500

    def test_model_performance_sensor(self):
        # Triggers retrain if F1 score < 0.85
        sensor = ModelPerformanceSensor(
            metric_eval_fn=lambda: 0.78,
            min_threshold=0.85,
        )
        res = sensor.poke()
        assert res.triggered is True
        assert res.payload["value"] == 0.78

    def test_drift_sensor(self):
        sensor = DriftSensor(
            drift_check_fn=lambda: (True, 0.01),
            p_value_threshold=0.05,
        )
        res = sensor.poke()
        assert res.triggered is True

    def test_file_arrival_sensor(self):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(b"sample data content")
            tf.flush()
            temp_path = tf.name

        sensor = FileArrivalSensor(filepath=temp_path, min_size_bytes=5)
        res = sensor.poke()
        assert res.triggered is True
        assert res.payload["size"] > 0
        Path(temp_path).unlink(missing_ok=True)


class TestDAGFactoriesAndExporters:
    def test_create_dags_and_mermaid(self):
        train_dag = create_training_dag()
        assert "ingest_data" in train_dag._tasks
        assert "train_model" in train_dag._tasks
        mermaid = train_dag.to_mermaid()
        assert "graph TD" in mermaid

        eval_dag = create_evaluation_dag()
        assert len(eval_dag._tasks) == 4

        deploy_dag = create_deployment_dag()
        assert len(deploy_dag._tasks) == 5

    def test_airflow_and_dagster_exporters(self):
        train_dag = create_training_dag()
        airflow_code = AirflowOrchestrator.export_to_airflow_py(train_dag)
        assert "from airflow import DAG" in airflow_code
        assert "ingest_data >> validate_data" in airflow_code

        dagster_code = DagsterOrchestrator.export_to_dagster_py(train_dag)
        assert "from dagster import job, op" in dagster_code
        assert "@job" in dagster_code
