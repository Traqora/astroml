"""DVC pipeline integration for reproducible ML workflows.

Wraps DVC pipeline stages as callable Python functions and provides
pipeline composition, caching, and CI/CD integration.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from astroml.storage.data_versioning import DataVersionControl, DatasetVersion

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """A single stage in a DVC pipeline.

    Attributes:
        stage_id: Unique stage identifier.
        name: Human-readable stage name.
        command: Shell command to execute.
        dependencies: File paths this stage depends on.
        outputs: File paths this stage produces.
        params: Optional parameter dictionary.
        cache_enabled: Whether to use DVC caching for this stage.
    """

    stage_id: str
    name: str
    command: str
    dependencies: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict[str, Any] | None = None
    cache_enabled: bool = True


@dataclass
class PipelineRun:
    """Records the execution of a pipeline."""

    run_id: str
    pipeline_name: str
    stages: list[str]
    status: str = "pending"  # pending, running, completed, failed
    started_at: str | None = None
    completed_at: str | None = None
    dataset_versions: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineDefinition:
    """Defines a reproducible DVC pipeline.

    Attributes:
        pipeline_id: Unique identifier.
        name: Pipeline name.
        description: Human-readable description.
        stages: Ordered list of pipeline stages.
        schedule: Optional cron expression for scheduled runs.
        tags: Optional tags.
    """

    pipeline_id: str
    name: str
    description: str = ""
    stages: list[PipelineStage] = field(default_factory=list)
    schedule: str | None = None
    tags: list[str] = field(default_factory=list)


class DVCPipelineManager:
    """Manages reproducible ML pipelines with DVC integration.

    Usage::

        mgr = DVCPipelineManager()
        pipe = mgr.create_pipeline("feature-engineering")
        mgr.add_stage(pipe.pipeline_id, "clean",
                       command="python clean.py",
                       dependencies=["data/raw.csv"],
                       outputs=["data/clean.csv"])
        run = mgr.run(pipe.pipeline_id)
    """

    def __init__(
        self,
        repo_root: str | Path = ".",
        dvc: DataVersionControl | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self._dvc = dvc or DataVersionControl(repo_root)
        self._pipelines: dict[str, PipelineDefinition] = {}
        self._runs: dict[str, PipelineRun] = {}

    # ------------------------------------------------------------------
    # Pipeline management
    # ------------------------------------------------------------------

    def create_pipeline(
        self,
        name: str,
        description: str = "",
        schedule: str | None = None,
        tags: list[str] | None = None,
    ) -> PipelineDefinition:
        """Create a new pipeline definition.

        Args:
            name: Pipeline name.
            description: Human-readable description.
            schedule: Optional cron schedule expression.
            tags: Optional tags.

        Returns:
            The created PipelineDefinition.
        """
        pipe = PipelineDefinition(
            pipeline_id=uuid.uuid4().hex[:12],
            name=name,
            description=description,
            schedule=schedule,
            tags=tags or [],
        )
        self._pipelines[pipe.pipeline_id] = pipe
        logger.info("Created pipeline '%s' (id=%s)", name, pipe.pipeline_id)
        return pipe

    def get_pipeline(self, pipeline_id: str) -> PipelineDefinition | None:
        """Get a pipeline by ID.

        Args:
            pipeline_id: Pipeline identifier.

        Returns:
            PipelineDefinition or None.
        """
        return self._pipelines.get(pipeline_id)

    def list_pipelines(
        self, tags: list[str] | None = None
    ) -> list[PipelineDefinition]:
        """List pipelines, optionally filtered by tags.

        Args:
            tags: Optional tag filter.

        Returns:
            Filtered pipeline list.
        """
        pipes = list(self._pipelines.values())
        if tags:
            tag_set = set(tags)
            pipes = [p for p in pipes if tag_set.issubset(set(p.tags))]
        return pipes

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def add_stage(
        self,
        pipeline_id: str,
        name: str,
        command: str,
        dependencies: list[str] | None = None,
        outputs: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> PipelineStage:
        """Add a stage to a pipeline.

        Args:
            pipeline_id: Target pipeline.
            name: Stage name.
            command: Shell command.
            dependencies: File dependencies.
            outputs: File outputs.
            params: Optional parameters.

        Returns:
            The created PipelineStage.
        """
        pipe = self._pipelines.get(pipeline_id)
        if pipe is None:
            raise ValueError(f"Pipeline '{pipeline_id}' not found")

        stage = PipelineStage(
            stage_id=uuid.uuid4().hex[:8],
            name=name,
            command=command,
            dependencies=dependencies or [],
            outputs=outputs or [],
            params=params,
        )
        pipe.stages.append(stage)
        logger.info(
            "Added stage '%s' to pipeline '%s'",
            name,
            pipe.name,
        )
        return stage

    def remove_stage(self, pipeline_id: str, stage_id: str) -> None:
        """Remove a stage from a pipeline.

        Args:
            pipeline_id: Pipeline containing the stage.
            stage_id: Stage to remove.
        """
        pipe = self._pipelines.get(pipeline_id)
        if pipe is None:
            raise ValueError(f"Pipeline '{pipeline_id}' not found")
        pipe.stages = [s for s in pipe.stages if s.stage_id != stage_id]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        pipeline_id: str,
        cache: bool = True,
    ) -> PipelineRun:
        """Run a pipeline sequentially.

        Args:
            pipeline_id: Pipeline to execute.
            cache: Whether to use DVC caching (skip unchanged stages).

        Returns:
            PipelineRun with execution status.
        """
        pipe = self._pipelines.get(pipeline_id)
        if pipe is None:
            raise ValueError(f"Pipeline '{pipeline_id}' not found")

        if not pipe.stages:
            raise ValueError(f"Pipeline '{pipe.name}' has no stages")

        run = PipelineRun(
            run_id=uuid.uuid4().hex[:12],
            pipeline_name=pipe.name,
            stages=[s.name for s in pipe.stages],
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._runs[run.run_id] = run

        logger.info(
            "Starting pipeline '%s' run %s (%d stages)",
            pipe.name,
            run.run_id,
            len(pipe.stages),
        )

        import subprocess

        for stage in pipe.stages:
            logger.info("Executing stage: %s", stage.name)

            if self._dvc.enabled and stage.cache_enabled and cache:
                result = self._dvc.run_pipeline_stage(
                    stage_name=stage.name,
                    command=stage.command,
                    dependencies=stage.dependencies,
                    outputs=stage.outputs,
                    params=stage.params,
                )
                if not result["success"]:
                    run.status = "failed"
                    run.error = result.get("stderr", result.get("error", "Unknown error"))
                    run.completed_at = datetime.now(timezone.utc).isoformat()
                    logger.error(
                        "Stage '%s' failed: %s",
                        stage.name,
                        run.error,
                    )
                    return run
            else:
                try:
                    subprocess.run(
                        stage.command,
                        shell=True,
                        cwd=str(self.repo_root),
                        check=True,
                        timeout=600,
                    )
                except subprocess.CalledProcessError as exc:
                    run.status = "failed"
                    run.error = str(exc)
                    run.completed_at = datetime.now(timezone.utc).isoformat()
                    logger.error("Stage '%s' failed: %s", stage.name, exc)
                    return run

            logger.info("Stage '%s' completed", stage.name)

        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc).isoformat()
        logger.info("Pipeline '%s' finished successfully", pipe.name)
        return run

    def get_run_status(self, run_id: str) -> PipelineRun | None:
        """Get a pipeline run by ID.

        Args:
            run_id: Run identifier.

        Returns:
            PipelineRun or None.
        """
        return self._runs.get(run_id)

    def list_runs(
        self,
        pipeline_id: str | None = None,
        limit: int = 20,
    ) -> list[PipelineRun]:
        """List pipeline runs, optionally filtered.

        Args:
            pipeline_id: Optional pipeline filter.
            limit: Maximum number of runs to return.

        Returns:
            List of runs, newest first.
        """
        runs = list(self._runs.values())
        if pipeline_id:
            runs = [r for r in runs if r.pipeline_name == pipeline_id]
        runs.sort(
            key=lambda r: r.started_at or "",
            reverse=True,
        )
        return runs[:limit]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_definition(
        self, pipeline_id: str, output_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Export a pipeline definition as a JSON-compatible dict.

        Args:
            pipeline_id: Pipeline to export.
            output_path: Optional file path to write to.

        Returns:
            Pipeline definition dict.
        """
        pipe = self._pipelines.get(pipeline_id)
        if pipe is None:
            raise ValueError(f"Pipeline '{pipeline_id}' not found")

        data = {
            "pipeline_id": pipe.pipeline_id,
            "name": pipe.name,
            "description": pipe.description,
            "schedule": pipe.schedule,
            "tags": pipe.tags,
            "stages": [
                {
                    "stage_id": s.stage_id,
                    "name": s.name,
                    "command": s.command,
                    "dependencies": s.dependencies,
                    "outputs": s.outputs,
                    "params": s.params,
                }
                for s in pipe.stages
            ],
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        return data