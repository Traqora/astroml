from astroml.utils.exceptions import AstroMLError
"""Automated pipeline orchestration engine supporting Airflow and Dagster patterns.

Provides DAG graph definition, task dependency management, topological resolution,
retry handling with backoff, SLA monitoring, and execution tracking.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Execution status of a pipeline task."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class PipelineRunStatus(Enum):
    """Overall status of a pipeline run."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class RetryPolicy:
    """Retry policy for pipeline tasks."""

    max_retries: int = 3
    delay_seconds: float = 1.0
    backoff_factor: float = 2.0
    retry_on_exceptions: tuple[type[BaseException], ...] = (Exception,)


@dataclass
class PipelineTask:
    """A unit of work within a pipeline DAG."""

    task_id: str
    action: Callable[..., Any]
    upstream_task_ids: set[str] = field(default_factory=set)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_seconds: float | None = None
    on_success_callback: Callable[[str, Any], None] | None = None
    on_failure_callback: Callable[[str, Exception], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_upstream(self, *tasks: PipelineTask) -> PipelineTask:
        """Set upstream prerequisite tasks."""
        for t in tasks:
            self.upstream_task_ids.add(t.task_id)
        return self

    def set_downstream(self, *tasks: PipelineTask) -> PipelineTask:
        """Set downstream dependent tasks."""
        for t in tasks:
            t.upstream_task_ids.add(self.task_id)
        return self

    def __rshift__(self, other: PipelineTask) -> PipelineTask:
        """Airflow-style >> operator to chain tasks."""
        self.set_downstream(other)
        return other

    def __lshift__(self, other: PipelineTask) -> PipelineTask:
        """Airflow-style << operator to chain tasks."""
        self.set_upstream(other)
        return other


@dataclass
class TaskExecutionResult:
    """Record of a task execution."""

    task_id: str
    status: TaskStatus
    result: Any = None
    error: str | None = None
    attempts: int = 1
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_seconds: float = 0.0


@dataclass
class PipelineRun:
    """State and history of a pipeline DAG execution."""

    run_id: str
    pipeline_name: str
    status: PipelineRunStatus = PipelineRunStatus.NOT_STARTED
    task_results: dict[str, TaskExecutionResult] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)


class DAGDefinition:
    """Directed Acyclic Graph definition for ML pipelines."""

    def __init__(
        self,
        dag_id: str,
        description: str = "",
        schedule_interval: str | None = None,
        sla_seconds: float | None = None,
        default_retry_policy: RetryPolicy | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialize a DAG definition."""
        self.dag_id = dag_id
        self.description = description
        self.schedule_interval = schedule_interval
        self.sla_seconds = sla_seconds
        self.default_retry_policy = default_retry_policy or RetryPolicy()
        self.tags = tags or []
        self._tasks: dict[str, PipelineTask] = {}

    def add_task(self, task: PipelineTask) -> PipelineTask:
        """Add a task to the DAG."""
        if task.task_id in self._tasks:
            raise ValueError(f"Task with ID '{task.task_id}' already exists in DAG '{self.dag_id}'")
        self._tasks[task.task_id] = task
        return task

    def task(
        self,
        task_id: str,
        upstream_task_ids: Sequence[str] | None = None,
        retries: int = 3,
        timeout: float | None = None,
    ) -> Callable[[Callable[..., Any]], PipelineTask]:
        """Decorator to create and register a task."""

        def decorator(fn: Callable[..., Any]) -> PipelineTask:
            t = PipelineTask(
                task_id=task_id,
                action=fn,
                upstream_task_ids=set(upstream_task_ids or []),
                retry_policy=RetryPolicy(max_retries=retries),
                timeout_seconds=timeout,
            )
            self.add_task(t)
            return t

        return decorator

    def get_task(self, task_id: str) -> PipelineTask | None:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def topological_sort(self) -> list[str]:
        """Return task IDs sorted in dependency order, raising ValueError if cycle detected."""
        in_degree: dict[str, int] = {t_id: 0 for t_id in self._tasks}
        graph: dict[str, list[str]] = {t_id: [] for t_id in self._tasks}

        for task_id, task in self._tasks.items():
            for up in task.upstream_task_ids:
                if up not in self._tasks:
                    raise ValueError(f"Upstream task '{up}' not found in DAG '{self.dag_id}'")
                graph[up].append(task_id)
                in_degree[task_id] += 1

        queue = [t_id for t_id, deg in in_degree.items() if deg == 0]
        sorted_tasks = []

        while queue:
            node = queue.pop(0)
            sorted_tasks.append(node)
            for downstream in graph[node]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)

        if len(sorted_tasks) != len(self._tasks):
            raise ValueError(f"Cycle detected in DAG '{self.dag_id}'")

        return sorted_tasks

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram representation of the DAG."""
        lines = ["graph TD"]
        for task_id, task in self._tasks.items():
            for up in task.upstream_task_ids:
                lines.append(f"    {up} --> {task_id}")
            if not task.upstream_task_ids:
                lines.append(f"    {task_id}")
        return "\n".join(lines)


class LocalPipelineOrchestrator:
    """Executes DAGs locally with retries, timeout, and context passing."""

    def __init__(self) -> None:
        """Initialize orchestrator."""
        self._runs: dict[str, PipelineRun] = {}

    def run_dag(
        self,
        dag: DAGDefinition,
        initial_context: dict[str, Any] | None = None,
    ) -> PipelineRun:
        """Execute all tasks in the DAG in topological order."""
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run = PipelineRun(
            run_id=run_id,
            pipeline_name=dag.dag_id,
            status=PipelineRunStatus.RUNNING,
            context=initial_context.copy() if initial_context else {},
        )
        self._runs[run_id] = run

        start_time = time.time()
        task_order = dag.topological_sort()

        try:
            for task_id in task_order:
                task = dag._tasks[task_id]
                # Check if all upstream tasks succeeded
                failed_upstreams = [
                    up
                    for up in task.upstream_task_ids
                    if run.task_results.get(up)
                    and run.task_results[up].status != TaskStatus.SUCCESS
                ]
                if failed_upstreams:
                    logger.warning(
                        "Skipping task %s due to failed upstreams: %s", task_id, failed_upstreams
                    )
                    run.task_results[task_id] = TaskExecutionResult(
                        task_id=task_id,
                        status=TaskStatus.SKIPPED,
                        error=f"Upstream tasks failed: {failed_upstreams}",
                    )
                    continue

                # Execute task with retries
                result = self._execute_task_with_retry(task, run.context)
                run.task_results[task_id] = result
                if result.status == TaskStatus.SUCCESS and result.result is not None:
                    run.context[f"{task_id}_output"] = result.result

                if result.status == TaskStatus.FAILED:
                    run.status = PipelineRunStatus.FAILED

            if run.status != PipelineRunStatus.FAILED:
                run.status = PipelineRunStatus.COMPLETED

        except Exception as exc:
            logger.error("Pipeline run %s failed with exception: %s", run_id, exc)
            run.status = PipelineRunStatus.FAILED

        finally:
            run.completed_at = datetime.now(timezone.utc)
            run.duration_seconds = time.time() - start_time

            # SLA Check
            if dag.sla_seconds and run.duration_seconds > dag.sla_seconds:
                logger.warning(
                    "Pipeline run %s exceeded SLA of %.1fs (took %.1fs)",
                    run_id,
                    dag.sla_seconds,
                    run.duration_seconds,
                )

        return run

    def _execute_task_with_retry(
        self,
        task: PipelineTask,
        context: dict[str, Any],
    ) -> TaskExecutionResult:
        """Execute a single task with configured retry backoff."""
        attempts = 0
        max_attempts = max(1, task.retry_policy.max_retries + 1)
        delay = task.retry_policy.delay_seconds
        start_time = time.time()

        while attempts < max_attempts:
            attempts += 1
            try:
                logger.info(
                    "Executing task %s (attempt %d/%d)", task.task_id, attempts, max_attempts
                )
                output = task.action(context)
                if task.on_success_callback:
                    task.on_success_callback(task.task_id, output)
                return TaskExecutionResult(
                    task_id=task.task_id,
                    status=TaskStatus.SUCCESS,
                    result=output,
                    attempts=attempts,
                    completed_at=datetime.now(timezone.utc),
                    duration_seconds=time.time() - start_time,
                )
            except task.retry_policy.retry_on_exceptions as exc:
                logger.warning("Task %s failed on attempt %d: %s", task.task_id, attempts, exc)
                if attempts >= max_attempts:
                    if task.on_failure_callback:
                        task.on_failure_callback(task.task_id, exc)
                    return TaskExecutionResult(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error=str(exc),
                        attempts=attempts,
                        completed_at=datetime.now(timezone.utc),
                        duration_seconds=time.time() - start_time,
                    )
                time.sleep(delay)
                delay *= task.retry_policy.backoff_factor

        return TaskExecutionResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Exceeded max attempts",
            attempts=attempts,
            completed_at=datetime.now(timezone.utc),
            duration_seconds=time.time() - start_time,
        )

    def get_run(self, run_id: str) -> PipelineRun | None:
        """Get pipeline run metadata by ID."""
        return self._runs.get(run_id)


class AirflowOrchestrator:
    """Generates standard Apache Airflow DAG definitions from DAGDefinition objects."""

    @staticmethod
    def export_to_airflow_py(dag: DAGDefinition) -> str:
        """Export DAG to Python source code compatible with Apache Airflow."""
        code = [
            "# Auto-generated Apache Airflow DAG for AstroML",
            "from datetime import datetime, timedelta",
            "from airflow import DAG",
            "from airflow.operators.python import PythonOperator",
            "",
            f"default_args = {{",
            f"    'owner': 'astroml',",
            f"    'retries': {dag.default_retry_policy.max_retries},",
            f"    'retry_delay': timedelta(seconds={dag.default_retry_policy.delay_seconds}),",
            f"}}",
            "",
            f"with DAG(",
            f"    dag_id='{dag.dag_id}',",
            f"    default_args=default_args,",
            f"    description='{dag.description}',",
            f"    schedule_interval={repr(dag.schedule_interval)},",
            f"    start_date=datetime(2025, 1, 1),",
            f"    catchup=False,",
            f"    tags={repr(dag.tags)},",
            f") as dag:",
            "",
        ]

        for task_id, task in dag._tasks.items():
            code.append(f"    {task_id} = PythonOperator(")
            code.append(f"        task_id='{task_id}',")
            code.append(f"        python_callable=lambda **kwargs: print('Running {task_id}'),")
            code.append(f"    )")

        code.append("")
        for task_id, task in dag._tasks.items():
            for up in task.upstream_task_ids:
                code.append(f"    {up} >> {task_id}")

        return "\n".join(code)


class DagsterOrchestrator:
    """Generates Dagster @job and @op definitions from DAGDefinition objects."""

    @staticmethod
    def export_to_dagster_py(dag: DAGDefinition) -> str:
        """Export DAG to Dagster job definition."""
        code = [
            "# Auto-generated Dagster Job for AstroML",
            "from dagster import job, op, In, Out",
            "",
        ]
        for task_id in dag._tasks:
            code.append(f"@op(name='{task_id}')")
            code.append(f"def op_{task_id}(context):")
            code.append(f"    context.log.info('Running {task_id}')")
            code.append(f"    return '{task_id}_done'")
            code.append("")

        code.append(f"@job(name='{dag.dag_id}', description='{dag.description}')")
        code.append("def pipeline_job():")
        for task_id in dag.topological_sort():
            code.append(f"    res_{task_id} = op_{task_id}()")

        return "\n".join(code)
