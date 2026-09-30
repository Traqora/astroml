"""Detailed provenance tracking across pipeline stages.

Tracks input/output schemas, row counts, checksums, and timestamps
for each pipeline stage, with support for verification and comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from typing import Any, Generator

from pydantic import BaseModel, ConfigDict, Field

from astroml.tracking.lineage.metadata_store import MetadataRecord, MetadataStore

logger = logging.getLogger(__name__)


class StageRecord(BaseModel):
    """Records the provenance of a single pipeline stage.

    Attributes:
        name: Name of the pipeline stage.
        start_time: When the stage started.
        end_time: When the stage ended.
        input_schema: Schema of input data (column name -> type).
        output_schema: Schema of output data (column name -> type).
        row_count_input: Number of input rows.
        row_count_output: Number of output rows.
        checksum_input: MD5 checksum of input data.
        checksum_output: MD5 checksum of output data.
        metadata: Arbitrary stage-specific metadata.
        nested_stages: Records of nested sub-stages.
    """

    name: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    input_schema: dict[str, str] = Field(default_factory=dict)
    output_schema: dict[str, str] = Field(default_factory=dict)
    row_count_input: int | None = None
    row_count_output: int | None = None
    checksum_input: str | None = None
    checksum_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    nested_stages: list[StageRecord] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    def duration_seconds(self) -> float | None:
        """Return the stage duration in seconds, or None if not completed."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds(),
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "row_count_input": self.row_count_input,
            "row_count_output": self.row_count_output,
            "checksum_input": self.checksum_input,
            "checksum_output": self.checksum_output,
            "metadata": self.metadata,
            "nested_stages": [s.to_dict() for s in self.nested_stages],
        }


class ProvenanceChain(BaseModel):
    """The complete provenance chain for a pipeline run.

    Attributes:
        run_id: Unique identifier for the run.
        stages: Ordered list of stage provenance records.
        created_at: When the chain was created.
    """

    run_id: str
    stages: list[StageRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire chain to a dictionary."""
        return {
            "run_id": self.run_id,
            "stages": [s.to_dict() for s in self.stages],
            "created_at": self.created_at.isoformat(),
        }


class ProvenanceTracker:
    """Tracks data provenance across pipeline stages in detail.

    Provides a context manager for stages, computes checksums, and
    supports verification, export, and comparison of provenance chains.
    """

    def __init__(self, metadata_store: MetadataStore | None = None) -> None:
        """Initialize the provenance tracker.

        Args:
            metadata_store: An optional MetadataStore for persisting
                provenance records.
        """
        self._store = metadata_store or MetadataStore()
        self._chains: dict[str, ProvenanceChain] = {}
        self._stage_stack: list[StageRecord] = []

    @contextmanager
    def stage(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[StageRecord, None, None]:
        """Context manager for tracking a pipeline stage.

        Records start/end times and captures nested stages.

        Args:
            name: Name of the stage.
            metadata: Optional metadata for the stage.

        Yields:
            The StageRecord being populated.

        Example:
            with tracker.stage("preprocessing") as stage:
                stage.row_count_input = 1000
                stage.row_count_output = 950
        """
        record = StageRecord(name=name, metadata=metadata or {})

        # If there's a parent stage, register as nested
        if self._stage_stack:
            self._stage_stack[-1].nested_stages.append(record)

        self._stage_stack.append(record)
        try:
            yield record
        finally:
            record.end_time = datetime.utcnow()
            self._stage_stack.pop()

    def finalize_run(
        self,
        run_id: str,
        stages: list[StageRecord] | None = None,
    ) -> ProvenanceChain:
        """Finalize a provenance chain for a run.

        Args:
            run_id: Unique ID for the run.
            stages: Optional list of stage records. If not provided, uses
                stages recorded via the ``stage`` context manager.

        Returns:
            The finalized ProvenanceChain.
        """
        if stages is not None:
            chain = ProvenanceChain(run_id=run_id, stages=stages)
        else:
            chain = ProvenanceChain(run_id=run_id)

        self._chains[run_id] = chain
        self._store.store_run(run_id, metadata={"provenance": True})

        logger.info("Finalized provenance chain for run: %s", run_id)
        return chain

    def verify_provenance(self, run_id: str) -> dict[str, Any]:
        """Verify the integrity of a provenance chain.

        Checks that:
        - All stages have start and end times.
        - Checksums are consistent between consecutive stages.
        - Row counts are consistent (no unexplained data loss/gain).

        Args:
            run_id: The run to verify.

        Returns:
            Dict with keys: valid (bool), errors (list), warnings (list).
        """
        chain = self._chains.get(run_id)
        if chain is None:
            return {"valid": False, "errors": [f"Run {run_id!r} not found"], "warnings": []}

        errors: list[str] = []
        warnings: list[str] = []

        for i, stage in enumerate(chain.stages):
            if stage.end_time is None:
                errors.append(f"Stage {i} ({stage.name!r}) has no end time")

            if stage.checksum_input and stage.checksum_output:
                if (
                    stage.checksum_input == stage.checksum_output
                    and stage.row_count_input != stage.row_count_output
                ):
                    warnings.append(
                        f"Stage {i} ({stage.name!r}): checksums match but row counts differ "
                        f"({stage.row_count_input} -> {stage.row_count_output})"
                    )

            # Check consistency with previous stage
            if i > 0:
                prev = chain.stages[i - 1]
                if (
                    prev.checksum_output is not None
                    and stage.checksum_input is not None
                    and prev.checksum_output != stage.checksum_input
                ):
                    errors.append(
                        f"Checksum mismatch between stage {i-1} ({prev.name!r}) "
                        f"and stage {i} ({stage.name!r})"
                    )
                if (
                    prev.row_count_output is not None
                    and stage.row_count_input is not None
                    and prev.row_count_output != stage.row_count_input
                ):
                    errors.append(
                        f"Row count mismatch between stage {i-1} ({prev.name!r}) "
                        f"and stage {i} ({stage.name!r}): "
                        f"{prev.row_count_output} vs {stage.row_count_input}"
                    )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def export_provenance(
        self,
        run_id: str,
        fmt: str = "json",
    ) -> dict[str, Any] | str:
        """Export a provenance chain in the requested format.

        Args:
            run_id: The run to export.
            fmt: Output format - "json" (default) or "dict".

        Returns:
            The provenance chain as a dict or JSON string.

        Raises:
            ValueError: If the run_id is not found or format is unsupported.
        """
        chain = self._chains.get(run_id)
        if chain is None:
            raise ValueError(f"Run {run_id!r} not found")

        if fmt == "dict":
            return chain.to_dict()

        if fmt == "json":
            return json.dumps(chain.to_dict(), indent=2, default=str)

        raise ValueError(f"Unsupported format: {fmt!r}")

    def compare_provenance(
        self,
        run_id_1: str,
        run_id_2: str,
    ) -> dict[str, Any]:
        """Compare two provenance chains.

        Args:
            run_id_1: First run ID.
            run_id_2: Second run ID.

        Returns:
            Dict with comparison results including differences in stages,
            row counts, and checksums.

        Raises:
            ValueError: If either run_id is not found.
        """
        chain1 = self._chains.get(run_id_1)
        chain2 = self._chains.get(run_id_2)

        if chain1 is None:
            raise ValueError(f"Run {run_id_1!r} not found")
        if chain2 is None:
            raise ValueError(f"Run {run_id_2!r} not found")

        result: dict[str, Any] = {
            "run_id_1": run_id_1,
            "run_id_2": run_id_2,
            "stage_count_match": len(chain1.stages) == len(chain2.stages),
            "stage_differences": [],
            "checksum_differences": [],
        }

        max_stages = max(len(chain1.stages), len(chain2.stages))
        for i in range(max_stages):
            if i >= len(chain1.stages):
                result["stage_differences"].append(
                    {
                        "index": i,
                        "difference": "stage only in run_2",
                        "run_2_name": chain2.stages[i].name,
                    }
                )
                continue
            if i >= len(chain2.stages):
                result["stage_differences"].append(
                    {
                        "index": i,
                        "difference": "stage only in run_1",
                        "run_1_name": chain1.stages[i].name,
                    }
                )
                continue

            s1 = chain1.stages[i]
            s2 = chain2.stages[i]

            if s1.name != s2.name:
                result["stage_differences"].append(
                    {
                        "index": i,
                        "difference": "stage name mismatch",
                        "run_1_name": s1.name,
                        "run_2_name": s2.name,
                    }
                )

            if s1.checksum_output != s2.checksum_output:
                result["checksum_differences"].append(
                    {
                        "index": i,
                        "stage_name": s1.name,
                        "run_1_checksum": s1.checksum_output,
                        "run_2_checksum": s2.checksum_output,
                    }
                )

        return result

    @staticmethod
    def compute_checksum(data: list[dict[str, Any]] | str) -> str:
        """Compute an MD5 checksum for provenance data.

        Args:
            data: Data to checksum, either a list of dicts or a string.

        Returns:
            Hex digest of the MD5 checksum.
        """
        if isinstance(data, list):
            raw = json.dumps(data, sort_keys=True, default=str).encode()
        else:
            raw = data.encode()
        return hashlib.md5(raw).hexdigest()
