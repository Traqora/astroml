"""Model card generation following the Google Model Cards specification.

Resolves part of #646.

The dataclasses mirror the sections of the Model Cards paper (Mitchell et al.,
2019) — model details, intended use, factors, metrics, evaluation and training
data, ethical considerations, and caveats — and render to either a JSON
document or Markdown suitable for publishing alongside the model artifact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any

from astroml.pipeline.documentation.templates import load_template

__all__ = [
    "ConsiderationSection",
    "EvaluationData",
    "IntendedUse",
    "MetricEntry",
    "ModelCard",
    "ModelCardBuilder",
    "ModelDetails",
    "QuantitativeAnalysis",
    "TrainingData",
]


def _utcnow_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelDetails:
    """ "Model Details" section: what the model is and who owns it."""

    name: str
    overview: str = ""
    version: str = "0.1.0"
    owners: list[str] = field(default_factory=list)
    licenses: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    model_type: str = ""
    training_framework: str = ""
    date: str = field(default_factory=_utcnow_iso)


@dataclass
class IntendedUse:
    """ "Intended Use" section: primary uses, users and out-of-scope uses."""

    primary_uses: list[str] = field(default_factory=list)
    primary_users: list[str] = field(default_factory=list)
    out_of_scope_uses: list[str] = field(default_factory=list)


@dataclass
class MetricEntry:
    """A single evaluation metric, optionally sliced by a factor."""

    name: str
    value: float
    slice_name: str = "overall"
    threshold: float | None = None
    higher_is_better: bool = True

    def passes(self) -> bool | None:
        """Return whether the metric meets its threshold, or ``None`` if unset."""
        if self.threshold is None:
            return None
        return (
            self.value >= self.threshold if self.higher_is_better else self.value <= self.threshold
        )


@dataclass
class QuantitativeAnalysis:
    """ "Quantitative Analysis" section: performance and fairness metrics."""

    performance_metrics: list[MetricEntry] = field(default_factory=list)
    fairness_metrics: list[MetricEntry] = field(default_factory=list)
    graphics: list[str] = field(default_factory=list)


@dataclass
class TrainingData:
    """ "Training Data" section."""

    description: str = ""
    sources: list[str] = field(default_factory=list)
    size: int | None = None
    preprocessing: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)


@dataclass
class EvaluationData:
    """ "Evaluation Data" section."""

    description: str = ""
    sources: list[str] = field(default_factory=list)
    size: int | None = None
    split_strategy: str = ""


@dataclass
class ConsiderationSection:
    """ "Considerations" section: ethics, risks, limitations and mitigations."""

    ethical_considerations: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)
    risks_and_mitigations: list[str] = field(default_factory=list)
    caveats_and_recommendations: list[str] = field(default_factory=list)


@dataclass
class ModelCard:
    """A complete model card.

    Use :meth:`to_dict` for the machine-readable form, :meth:`to_markdown` for
    the published document, and :meth:`validate` to check the card contains the
    sections the Model Cards specification requires.
    """

    model_details: ModelDetails
    intended_use: IntendedUse = field(default_factory=IntendedUse)
    factors: list[str] = field(default_factory=list)
    quantitative_analysis: QuantitativeAnalysis = field(default_factory=QuantitativeAnalysis)
    training_data: TrainingData = field(default_factory=TrainingData)
    evaluation_data: EvaluationData = field(default_factory=EvaluationData)
    considerations: ConsiderationSection = field(default_factory=ConsiderationSection)
    schema_version: str = "0.0.2"
    generated_at: str = field(default_factory=_utcnow_iso)

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return the card as a JSON-serialisable dictionary."""
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Return the card as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def to_markdown(self) -> str:
        """Render the card to Markdown using the packaged template."""
        template = Template(load_template("model_card.md.tmpl"))
        return template.safe_substitute(
            name=self.model_details.name,
            version=self.model_details.version,
            generated_at=self.generated_at,
            overview=self.model_details.overview or "_Not documented._",
            model_type=self.model_details.model_type or "unspecified",
            framework=self.model_details.training_framework or "unspecified",
            owners=_bullets(self.model_details.owners),
            licenses=_bullets(self.model_details.licenses),
            references=_bullets(self.model_details.references),
            primary_uses=_bullets(self.intended_use.primary_uses),
            primary_users=_bullets(self.intended_use.primary_users),
            out_of_scope=_bullets(self.intended_use.out_of_scope_uses),
            factors=_bullets(self.factors),
            performance_metrics=_metric_table(self.quantitative_analysis.performance_metrics),
            fairness_metrics=_metric_table(self.quantitative_analysis.fairness_metrics),
            training_description=self.training_data.description or "_Not documented._",
            training_sources=_bullets(self.training_data.sources),
            training_size=_optional_number(self.training_data.size),
            preprocessing=_bullets(self.training_data.preprocessing),
            evaluation_description=self.evaluation_data.description or "_Not documented._",
            evaluation_sources=_bullets(self.evaluation_data.sources),
            evaluation_size=_optional_number(self.evaluation_data.size),
            split_strategy=self.evaluation_data.split_strategy or "unspecified",
            ethical_considerations=_bullets(self.considerations.ethical_considerations),
            limitations=_bullets(self.considerations.limitations),
            tradeoffs=_bullets(self.considerations.tradeoffs),
            risks=_bullets(self.considerations.risks_and_mitigations),
            caveats=_bullets(self.considerations.caveats_and_recommendations),
        )

    def write(
        self, directory: str | Path, *, formats: tuple[str, ...] = ("md", "json")
    ) -> list[Path]:
        """Write the card to ``directory`` in the requested formats."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        stem = _slugify(self.model_details.name)
        written: list[Path] = []
        for fmt in formats:
            if fmt == "md":
                path = target / f"{stem}_model_card.md"
                path.write_text(self.to_markdown(), encoding="utf-8")
            elif fmt == "json":
                path = target / f"{stem}_model_card.json"
                path.write_text(self.to_json(), encoding="utf-8")
            else:
                raise ValueError(f"unsupported format {fmt!r}; expected 'md' or 'json'")
            written.append(path)
        return written

    # ── Validation ───────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Return a list of missing required sections; empty means valid.

        The Model Cards specification treats model details, intended use,
        metrics, and ethical considerations as the minimum publishable set.
        """
        problems: list[str] = []
        if not self.model_details.name:
            problems.append("model_details.name is required")
        if not self.model_details.overview:
            problems.append("model_details.overview is required")
        if not self.model_details.owners:
            problems.append("model_details.owners must list at least one owner")
        if not self.intended_use.primary_uses:
            problems.append("intended_use.primary_uses must list at least one use")
        if not self.quantitative_analysis.performance_metrics:
            problems.append("quantitative_analysis.performance_metrics must not be empty")
        if not self.considerations.ethical_considerations:
            problems.append("considerations.ethical_considerations must not be empty")
        return problems

    def is_valid(self) -> bool:
        """Return whether the card satisfies :meth:`validate`."""
        return not self.validate()


class ModelCardBuilder:
    """Fluent builder for :class:`ModelCard`.

    Example::

        card = (
            ModelCardBuilder("fraud-gnn")
            .with_overview("GraphSAGE anomaly scorer for Stellar accounts.")
            .with_owners(["ml-platform@example.com"])
            .with_intended_use(primary_uses=["Flag suspicious accounts for review"])
            .with_metric("roc_auc", 0.94, threshold=0.9)
            .with_ethical_considerations(["Scores must not be used to auto-freeze funds."])
            .build()
        )
    """

    def __init__(self, name: str, *, version: str = "0.1.0") -> None:
        if not name:
            raise ValueError("model name must not be empty")
        self._card = ModelCard(model_details=ModelDetails(name=name, version=version))

    def with_overview(self, overview: str) -> ModelCardBuilder:
        """Set the model overview."""
        self._card.model_details.overview = overview
        return self

    def with_owners(self, owners: list[str]) -> ModelCardBuilder:
        """Set the owning individuals or teams."""
        self._card.model_details.owners = list(owners)
        return self

    def with_model_type(self, model_type: str, *, framework: str = "") -> ModelCardBuilder:
        """Set the model architecture and training framework."""
        self._card.model_details.model_type = model_type
        self._card.model_details.training_framework = framework
        return self

    def with_licenses(self, licenses: list[str]) -> ModelCardBuilder:
        """Set the licences the model is released under."""
        self._card.model_details.licenses = list(licenses)
        return self

    def with_references(self, references: list[str]) -> ModelCardBuilder:
        """Set supporting references."""
        self._card.model_details.references = list(references)
        return self

    def with_intended_use(
        self,
        *,
        primary_uses: list[str] | None = None,
        primary_users: list[str] | None = None,
        out_of_scope_uses: list[str] | None = None,
    ) -> ModelCardBuilder:
        """Populate the intended-use section."""
        use = self._card.intended_use
        if primary_uses is not None:
            use.primary_uses = list(primary_uses)
        if primary_users is not None:
            use.primary_users = list(primary_users)
        if out_of_scope_uses is not None:
            use.out_of_scope_uses = list(out_of_scope_uses)
        return self

    def with_factors(self, factors: list[str]) -> ModelCardBuilder:
        """Set the relevant evaluation factors (population slices, instrumentation)."""
        self._card.factors = list(factors)
        return self

    def with_metric(
        self,
        name: str,
        value: float,
        *,
        slice_name: str = "overall",
        threshold: float | None = None,
        higher_is_better: bool = True,
        fairness: bool = False,
    ) -> ModelCardBuilder:
        """Add a performance (or, with ``fairness=True``, a fairness) metric."""
        entry = MetricEntry(
            name=name,
            value=value,
            slice_name=slice_name,
            threshold=threshold,
            higher_is_better=higher_is_better,
        )
        bucket = (
            self._card.quantitative_analysis.fairness_metrics
            if fairness
            else self._card.quantitative_analysis.performance_metrics
        )
        bucket.append(entry)
        return self

    def with_metrics(
        self, metrics: dict[str, float], *, fairness: bool = False
    ) -> ModelCardBuilder:
        """Add several metrics from a ``{name: value}`` mapping."""
        for name, value in metrics.items():
            self.with_metric(name, value, fairness=fairness)
        return self

    def with_training_data(
        self,
        description: str,
        *,
        sources: list[str] | None = None,
        size: int | None = None,
        preprocessing: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> ModelCardBuilder:
        """Populate the training-data section."""
        self._card.training_data = TrainingData(
            description=description,
            sources=list(sources or []),
            size=size,
            preprocessing=list(preprocessing or []),
            feature_names=list(feature_names or []),
        )
        return self

    def with_evaluation_data(
        self,
        description: str,
        *,
        sources: list[str] | None = None,
        size: int | None = None,
        split_strategy: str = "",
    ) -> ModelCardBuilder:
        """Populate the evaluation-data section."""
        self._card.evaluation_data = EvaluationData(
            description=description,
            sources=list(sources or []),
            size=size,
            split_strategy=split_strategy,
        )
        return self

    def with_ethical_considerations(self, considerations: list[str]) -> ModelCardBuilder:
        """Set the ethical considerations."""
        self._card.considerations.ethical_considerations = list(considerations)
        return self

    def with_limitations(self, limitations: list[str]) -> ModelCardBuilder:
        """Set the known limitations."""
        self._card.considerations.limitations = list(limitations)
        return self

    def with_caveats(self, caveats: list[str]) -> ModelCardBuilder:
        """Set the caveats and recommendations."""
        self._card.considerations.caveats_and_recommendations = list(caveats)
        return self

    def build(self) -> ModelCard:
        """Return the assembled card."""
        return self._card


# ─── Rendering helpers ───────────────────────────────────────────────────────


def _bullets(items: list[str]) -> str:
    """Render a list as Markdown bullets, or a placeholder when empty."""
    if not items:
        return "_None documented._"
    return "\n".join(f"- {item}" for item in items)


def _metric_table(metrics: list[MetricEntry]) -> str:
    """Render metrics as a Markdown table, or a placeholder when empty."""
    if not metrics:
        return "_No metrics recorded._"
    rows = [
        "| Metric | Slice | Value | Threshold | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for metric in metrics:
        passed = metric.passes()
        status = "n/a" if passed is None else ("pass" if passed else "FAIL")
        threshold = "—" if metric.threshold is None else f"{metric.threshold:.4g}"
        rows.append(
            f"| {metric.name} | {metric.slice_name} | {metric.value:.4g} "
            f"| {threshold} | {status} |"
        )
    return "\n".join(rows)


def _optional_number(value: int | None) -> str:
    """Render an optional count."""
    return "unknown" if value is None else f"{value:,}"


def _slugify(name: str) -> str:
    """Return a filesystem-safe slug for ``name``."""
    slug = "".join(char if char.isalnum() else "_" for char in name.lower())
    return "_".join(part for part in slug.split("_") if part) or "model"
