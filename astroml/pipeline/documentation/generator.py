"""Automated ML pipeline documentation generation.

Resolves part of #646.

The generator extracts :class:`PipelineMetadata` (declared explicitly, or
introspected from a callable / object pipeline), renders it to Markdown using
the packaged templates, versions each generated document by content hash, and
publishes the result into a documentation site directory.

Example::

    metadata = PipelineMetadata(
        name="fraud-scoring",
        description="Nightly anomaly scoring for Stellar accounts.",
        stages=[StageMetadata("ingest", NodeKind.SOURCE), ...],
    )
    generator = PipelineDocGenerator(output_dir="docs/pipelines")
    result = generator.generate(metadata)
    generator.publish(result)
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Callable

from astroml.pipeline.documentation.data_flow import (
    DataFlowDiagram,
    DataFlowNode,
    NodeKind,
)
from astroml.pipeline.documentation.model_card import ModelCard
from astroml.pipeline.documentation.templates import load_template

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentVersion",
    "GeneratedDoc",
    "IOSpec",
    "PipelineDocGenerator",
    "PipelineMetadata",
    "StageMetadata",
    "extract_metadata",
]


def _utcnow_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IOSpec:
    """An input or output of a pipeline."""

    name: str
    kind: str = "dataset"
    location: str = ""
    schema: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the spec."""
        return {
            "name": self.name,
            "kind": self.kind,
            "location": self.location,
            "schema": dict(self.schema),
            "description": self.description,
        }


@dataclass
class StageMetadata:
    """One stage of a pipeline."""

    name: str
    kind: NodeKind = NodeKind.TRANSFORM
    description: str = ""
    implementation: str = ""
    depends_on: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the stage."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "description": self.description,
            "implementation": self.implementation,
            "depends_on": list(self.depends_on),
            "parameters": dict(self.parameters),
        }


@dataclass
class PipelineMetadata:
    """Everything needed to document a pipeline."""

    name: str
    description: str = ""
    version: str = "0.1.0"
    owner: str = ""
    framework: str = ""
    schedule: str = "on-demand"
    stages: list[StageMetadata] = field(default_factory=list)
    inputs: list[IOSpec] = field(default_factory=list)
    outputs: list[IOSpec] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    model_card: ModelCard | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the pipeline name."""
        if not self.name:
            raise ValueError("pipeline name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "owner": self.owner,
            "framework": self.framework,
            "schedule": self.schedule,
            "stages": [stage.to_dict() for stage in self.stages],
            "inputs": [spec.to_dict() for spec in self.inputs],
            "outputs": [spec.to_dict() for spec in self.outputs],
            "parameters": dict(self.parameters),
            "dependencies": list(self.dependencies),
            "model_card": self.model_card.to_dict() if self.model_card else None,
            "extra": dict(self.extra),
        }

    def to_diagram(self) -> DataFlowDiagram:
        """Build the data flow diagram implied by the stage dependencies.

        Stages without a declared ``depends_on`` are chained in declaration
        order, which is the common case for linear pipelines.
        """
        diagram = DataFlowDiagram(name=self.name)
        for stage in self.stages:
            diagram.add_node(
                DataFlowNode(
                    node_id=stage.name,
                    label=stage.name,
                    kind=stage.kind,
                    description=stage.description,
                    metadata={"implementation": stage.implementation},
                )
            )
        declared_any = any(stage.depends_on for stage in self.stages)
        if declared_any:
            for stage in self.stages:
                for upstream in stage.depends_on:
                    diagram.add_edge(upstream, stage.name)
        else:
            for previous, current in zip(self.stages, self.stages[1:]):
                diagram.add_edge(previous.name, current.name)
        return diagram

    def validate(self) -> list[str]:
        """Return documentation-completeness problems; empty means complete."""
        problems: list[str] = []
        if not self.description:
            problems.append("pipeline description is required")
        if not self.owner:
            problems.append("pipeline owner is required")
        if not self.stages:
            problems.append("pipeline must declare at least one stage")
        if not self.inputs:
            problems.append("pipeline must declare at least one input")
        if not self.outputs:
            problems.append("pipeline must declare at least one output")
        known = {stage.name for stage in self.stages}
        for stage in self.stages:
            for upstream in stage.depends_on:
                if upstream not in known:
                    problems.append(f"stage {stage.name!r} depends on unknown stage {upstream!r}")
        return problems


@dataclass(frozen=True)
class DocumentVersion:
    """Immutable version marker for a generated document."""

    pipeline: str
    version: str
    content_hash: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the version."""
        return {
            "pipeline": self.pipeline,
            "version": self.version,
            "content_hash": self.content_hash,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class GeneratedDoc:
    """The rendered artefacts for one pipeline."""

    metadata: PipelineMetadata
    markdown: str
    mermaid: str
    dot: str
    version: DocumentVersion
    validation_problems: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        """Return whether the documentation passed completeness validation."""
        return not self.validation_problems

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the generated doc."""
        return {
            "metadata": self.metadata.to_dict(),
            "markdown": self.markdown,
            "mermaid": self.mermaid,
            "dot": self.dot,
            "version": self.version.to_dict(),
            "validation_problems": list(self.validation_problems),
            "is_complete": self.is_complete,
        }


class PipelineDocGenerator:
    """Renders, versions and publishes pipeline documentation.

    Parameters
    ----------
    output_dir:
        Root directory documentation is published into.  Each pipeline gets its
        own subdirectory containing ``index.md``, ``pipeline.json``,
        ``data_flow.mmd``/``.dot``, an optional model card, and a
        ``versions.json`` history.
    template_name:
        Name of the packaged template used for the pipeline page.
    """

    def __init__(
        self,
        output_dir: str | Path = "docs/pipelines",
        *,
        template_name: str = "pipeline.md.tmpl",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.template_name = template_name

    # ── Generation ───────────────────────────────────────────────────────────

    def generate(self, metadata: PipelineMetadata) -> GeneratedDoc:
        """Render documentation for ``metadata`` without writing to disk."""
        diagram = metadata.to_diagram()
        mermaid = diagram.to_mermaid()
        dot = diagram.to_dot()
        problems = tuple(metadata.validate() + diagram.validate())

        template = Template(load_template(self.template_name))
        markdown = template.safe_substitute(
            name=metadata.name,
            version=metadata.version,
            owner=metadata.owner or "_unassigned_",
            generated_at=_utcnow_iso(),
            description=metadata.description or "_No description provided._",
            stage_count=len(metadata.stages),
            schedule=metadata.schedule,
            framework=metadata.framework or "unspecified",
            input_count=len(metadata.inputs),
            output_count=len(metadata.outputs),
            doc_version=_content_hash(metadata)[:12],
            mermaid=mermaid,
            stages=_render_stages(metadata),
            inputs=_render_io(metadata.inputs),
            outputs=_render_io(metadata.outputs),
            parameters=_render_mapping(metadata.parameters),
            dependencies=_render_bullets(metadata.dependencies),
            validation=_render_validation(problems),
            changelog=_render_bullets(
                [str(entry) for entry in metadata.extra.get("changelog", [])]
            ),
        )

        version = DocumentVersion(
            pipeline=metadata.name,
            version=metadata.version,
            content_hash=_content_hash(metadata),
            generated_at=_utcnow_iso(),
        )
        return GeneratedDoc(
            metadata=metadata,
            markdown=markdown,
            mermaid=mermaid,
            dot=dot,
            version=version,
            validation_problems=problems,
        )

    # ── Publishing ───────────────────────────────────────────────────────────

    def publish(self, doc: GeneratedDoc) -> Path:
        """Write ``doc`` into the output directory and return that directory.

        Publishing is idempotent: re-publishing unchanged content leaves the
        version history untouched, so a docs build in CI produces no churn.
        """
        target = self.output_dir / _slugify(doc.metadata.name)
        target.mkdir(parents=True, exist_ok=True)

        (target / "index.md").write_text(doc.markdown, encoding="utf-8")
        (target / "pipeline.json").write_text(
            json.dumps(doc.metadata.to_dict(), indent=2), encoding="utf-8"
        )
        (target / "data_flow.mmd").write_text(doc.mermaid, encoding="utf-8")
        (target / "data_flow.dot").write_text(doc.dot, encoding="utf-8")
        if doc.metadata.model_card is not None:
            doc.metadata.model_card.write(target)

        self._append_version(target, doc.version)
        logger.info("published pipeline documentation to %s", target)
        return target

    def generate_and_publish(self, metadata: PipelineMetadata) -> tuple[GeneratedDoc, Path]:
        """Generate documentation for ``metadata`` and publish it."""
        doc = self.generate(metadata)
        return doc, self.publish(doc)

    def versions(self, pipeline_name: str) -> list[DocumentVersion]:
        """Return the published version history of a pipeline, oldest first."""
        history_file = self.output_dir / _slugify(pipeline_name) / "versions.json"
        if not history_file.is_file():
            return []
        raw = json.loads(history_file.read_text(encoding="utf-8"))
        return [DocumentVersion(**entry) for entry in raw]

    def _append_version(self, target: Path, version: DocumentVersion) -> None:
        """Append ``version`` to the pipeline's history unless unchanged."""
        history_file = target / "versions.json"
        history: list[dict[str, Any]] = []
        if history_file.is_file():
            history = json.loads(history_file.read_text(encoding="utf-8"))
        if history and history[-1]["content_hash"] == version.content_hash:
            return
        history.append(version.to_dict())
        history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")

    # ── Site index ───────────────────────────────────────────────────────────

    def build_index(self) -> Path:
        """Write an ``index.md`` listing every published pipeline."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            "# Pipelines",
            "",
            "| Pipeline | Version | Stages | Last generated |",
            "| --- | --- | --- | --- |",
        ]
        for manifest in sorted(self.output_dir.glob("*/pipeline.json")):
            data = json.loads(manifest.read_text(encoding="utf-8"))
            versions = self.versions(data["name"])
            last = versions[-1].generated_at if versions else "—"
            rows.append(
                f"| [{data['name']}]({manifest.parent.name}/index.md) "
                f"| {data['version']} | {len(data['stages'])} | {last} |"
            )
        index = self.output_dir / "index.md"
        index.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return index


# ─── Introspection ───────────────────────────────────────────────────────────


def extract_metadata(
    pipeline: Callable[..., Any] | object,
    *,
    name: str | None = None,
    stage_attribute: str = "stages",
) -> PipelineMetadata:
    """Derive :class:`PipelineMetadata` from a callable or object pipeline.

    The docstring becomes the description, the signature becomes the parameter
    set, and ``pipeline.stages`` (name configurable) becomes the stage list when
    present.  The result is a starting point — hand-authored metadata should
    override it where richer detail is available.
    """
    resolved_name = name or getattr(pipeline, "__name__", None) or type(pipeline).__name__
    description = inspect.getdoc(pipeline) or ""

    parameters: dict[str, Any] = {}
    target = pipeline if callable(pipeline) else getattr(pipeline, "run", None)
    if target is not None:
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            signature = None
        if signature is not None:
            for param_name, param in signature.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                parameters[param_name] = (
                    None if param.default is inspect.Parameter.empty else param.default
                )

    stages: list[StageMetadata] = []
    for raw in getattr(pipeline, stage_attribute, []) or []:
        if isinstance(raw, StageMetadata):
            stages.append(raw)
        elif isinstance(raw, str):
            stages.append(StageMetadata(name=raw))
        else:
            stages.append(
                StageMetadata(
                    name=getattr(raw, "name", type(raw).__name__),
                    description=inspect.getdoc(raw) or "",
                    implementation=f"{type(raw).__module__}.{type(raw).__name__}",
                )
            )

    return PipelineMetadata(
        name=resolved_name,
        description=description.split("\n\n")[0] if description else "",
        parameters=parameters,
        stages=stages,
        framework=getattr(pipeline, "framework", ""),
        owner=getattr(pipeline, "owner", ""),
    )


# ─── Rendering helpers ───────────────────────────────────────────────────────


def _content_hash(metadata: PipelineMetadata) -> str:
    """Return a stable SHA-256 hash of the pipeline's documented content."""
    payload = json.dumps(metadata.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_stages(metadata: PipelineMetadata) -> str:
    """Render the stage table."""
    if not metadata.stages:
        return "_No stages documented._"
    rows = [
        "| # | Stage | Kind | Depends on | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, stage in enumerate(metadata.stages, start=1):
        depends = ", ".join(stage.depends_on) or "—"
        rows.append(
            f"| {index} | `{stage.name}` | {stage.kind.value} | {depends} "
            f"| {stage.description or '—'} |"
        )
    return "\n".join(rows)


def _render_io(specs: list[IOSpec]) -> str:
    """Render an input/output table."""
    if not specs:
        return "_None documented._"
    rows = ["| Name | Kind | Location | Description |", "| --- | --- | --- | --- |"]
    for spec in specs:
        rows.append(
            f"| `{spec.name}` | {spec.kind} | {spec.location or '—'} "
            f"| {spec.description or '—'} |"
        )
    return "\n".join(rows)


def _render_mapping(mapping: dict[str, Any]) -> str:
    """Render a parameter mapping as a table."""
    if not mapping:
        return "_No parameters documented._"
    rows = ["| Parameter | Default |", "| --- | --- |"]
    for key, value in mapping.items():
        rows.append(f"| `{key}` | `{value!r}` |")
    return "\n".join(rows)


def _render_bullets(items: list[str]) -> str:
    """Render a list as Markdown bullets."""
    if not items:
        return "_None documented._"
    return "\n".join(f"- {item}" for item in items)


def _render_validation(problems: tuple[str, ...]) -> str:
    """Render the validation section."""
    if not problems:
        return "Documentation is complete — no missing sections detected."
    return "\n".join(f"- ⚠️ {problem}" for problem in problems)


def _slugify(name: str) -> str:
    """Return a filesystem-safe slug for ``name``."""
    slug = "".join(char if char.isalnum() else "-" for char in name.lower())
    return "-".join(part for part in slug.split("-") if part) or "pipeline"
