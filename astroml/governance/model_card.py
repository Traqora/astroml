"""Model card generation following Google Model Cards framework.

Issue #637 Step 3: Implements model card generation with structured
documentation of model details, intended use, performance metrics,
and ethical considerations.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Card data structures
# ---------------------------------------------------------------------------


class ModelType(Enum):
    """Types of ML models."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ANOMALY_DETECTION = "anomaly_detection"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    CLUSTERING = "clustering"
    EMBEDDING = "embedding"
    OTHER = "other"


@dataclass
class ModelDetails:
    """Basic model metadata."""

    name: str
    version: str
    model_type: ModelType = ModelType.OTHER
    description: str = ""
    authors: list[str] = field(default_factory=list)
    created_date: datetime | None = None
    last_updated: datetime | None = None
    framework: str = "pytorch"
    framework_version: str = ""
    license: str = ""


@dataclass
class IntendedUse:
    """Intended use and out-of-scope use cases."""

    primary_use: str = ""
    primary_users: list[str] = field(default_factory=list)
    out_of_scope_uses: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


@dataclass
class TrainingData:
    """Training data description."""

    data_sources: list[str] = field(default_factory=list)
    data_size: int = 0
    data_time_range: tuple[str, str] | None = None
    preprocessing_steps: list[str] = field(default_factory=list)
    train_test_split: float = 0.8
    known_biases: list[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Quantitative model performance metrics."""

    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    auc_roc: float | None = None
    rmse: float | None = None
    mae: float | None = None
    additional_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field_name in ("accuracy", "precision", "recall", "f1_score", "auc_roc", "rmse", "mae"):
            val = getattr(self, field_name)
            if val is not None:
                result[field_name] = val
        result.update(self.additional_metrics)
        return result


@dataclass
class FairnessEvaluation:
    """Fairness and bias evaluation results."""

    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    groups_evaluated: list[str] = field(default_factory=list)
    methodology: str = ""
    findings: list[str] = field(default_factory=list)


@dataclass
class EthicalConsiderations:
    """Ethical considerations and risks."""

    biases_identified: list[str] = field(default_factory=list)
    mitigations: list[str] = field(default_factory=list)
    privacy_impacts: list[str] = field(default_factory=list)
    societal_impacts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model Card
# ---------------------------------------------------------------------------


@dataclass
class ModelCard:
    """Complete model card following Google Model Cards framework.

    A Model Card provides structured documentation of a machine learning model,
    including its intended use, performance characteristics, and ethical
    considerations. It serves as a transparency artifact for stakeholders.

    Example:
        card = ModelCard(
            model_details=ModelDetails(
                name="Fraud Detector v2",
                version="2.0.0",
                model_type=ModelType.ANOMALY_DETECTION,
                description="Graph-based fraud detection model",
            ),
            intended_use=IntendedUse(
                primary_use="Detect fraudulent transactions on Stellar network",
                primary_users=["Risk analysts", "Compliance officers"],
            ),
        )
    """

    model_details: ModelDetails
    card_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    version: str = "1.0"
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Intended use
    intended_use: IntendedUse = field(default_factory=IntendedUse)

    # Data
    training_data: TrainingData = field(default_factory=TrainingData)
    evaluation_data: TrainingData = field(default_factory=TrainingData)

    # Performance
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Fairness
    fairness: FairnessEvaluation = field(default_factory=FairnessEvaluation)

    # Ethics
    ethical: EthicalConsiderations = field(default_factory=EthicalConsiderations)

    # Additional sections
    caveats: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    quantitative_analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize model card to a dictionary."""
        return {
            "card_id": self.card_id,
            "version": self.version,
            "generated_at": self.generated_at.isoformat(),
            "model_details": {
                "name": self.model_details.name,
                "version": self.model_details.version,
                "model_type": self.model_details.model_type.value,
                "description": self.model_details.description,
                "authors": self.model_details.authors,
                "created_date": (
                    self.model_details.created_date.isoformat()
                    if self.model_details.created_date
                    else None
                ),
                "last_updated": (
                    self.model_details.last_updated.isoformat()
                    if self.model_details.last_updated
                    else None
                ),
                "framework": self.model_details.framework,
                "framework_version": self.model_details.framework_version,
                "license": self.model_details.license,
            },
            "intended_use": {
                "primary_use": self.intended_use.primary_use,
                "primary_users": self.intended_use.primary_users,
                "out_of_scope_uses": self.intended_use.out_of_scope_uses,
                "limitations": self.intended_use.limitations,
            },
            "training_data": {
                "data_sources": self.training_data.data_sources,
                "data_size": self.training_data.data_size,
                "data_time_range": self.training_data.data_time_range,
                "preprocessing_steps": self.training_data.preprocessing_steps,
                "train_test_split": self.training_data.train_test_split,
                "known_biases": self.training_data.known_biases,
            },
            "performance": self.performance.to_dict(),
            "fairness": {
                "metrics": self.fairness.metrics,
                "groups_evaluated": self.fairness.groups_evaluated,
                "methodology": self.fairness.methodology,
                "findings": self.fairness.findings,
            },
            "ethical_considerations": {
                "biases_identified": self.ethical.biases_identified,
                "mitigations": self.ethical.mitigations,
                "privacy_impacts": self.ethical.privacy_impacts,
                "societal_impacts": self.ethical.societal_impacts,
            },
            "caveats": self.caveats,
            "references": self.references,
            "quantitative_analysis": self.quantitative_analysis,
        }

    def to_json(self, output_path: str | Path) -> None:
        """Write model card to a JSON file.

        Args:
            output_path: Path for the JSON output.
        """
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Model card written to {output_path}")

    def to_markdown(self, output_path: str | Path | None = None) -> str:
        """Generate a Markdown representation of the model card.

        Args:
            output_path: Optional path to write the Markdown to.

        Returns:
            Markdown string.
        """
        lines: list[str] = []

        # Header
        lines.append(f"# Model Card: {self.model_details.name} v{self.model_details.version}")
        lines.append("")
        lines.append(f"**Card Version**: {self.version}")
        lines.append(f"**Generated**: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")

        # Model Details
        lines.append("## Model Details")
        lines.append("")
        lines.append(f"- **Name**: {self.model_details.name}")
        lines.append(f"- **Version**: {self.model_details.version}")
        lines.append(f"- **Type**: {self.model_details.model_type.value}")
        lines.append(f"- **Description**: {self.model_details.description}")
        lines.append(f"- **Authors**: {', '.join(self.model_details.authors) if self.model_details.authors else 'N/A'}")
        lines.append(f"- **Framework**: {self.model_details.framework} {self.model_details.framework_version}")
        lines.append(f"- **License**: {self.model_details.license or 'N/A'}")
        lines.append("")

        # Intended Use
        lines.append("## Intended Use")
        lines.append("")
        lines.append(f"**Primary Use**: {self.intended_use.primary_use or 'Not specified'}")
        lines.append(f"**Primary Users**: {', '.join(self.intended_use.primary_users) if self.intended_use.primary_users else 'Not specified'}")
        if self.intended_use.out_of_scope_uses:
            lines.append("**Out-of-Scope Uses**:")
            for use in self.intended_use.out_of_scope_uses:
                lines.append(f"- {use}")
        if self.intended_use.limitations:
            lines.append("**Limitations**:")
            for lim in self.intended_use.limitations:
                lines.append(f"- {lim}")
        lines.append("")

        # Performance
        lines.append("## Performance Metrics")
        lines.append("")
        for k, v in self.performance.to_dict().items():
            lines.append(f"- **{k}**: {v:.4f}")
        lines.append("")

        # Fairness
        lines.append("## Fairness Evaluation")
        lines.append("")
        lines.append(f"**Methodology**: {self.fairness.methodology or 'Not specified'}")
        if self.fairness.findings:
            for finding in self.fairness.findings:
                lines.append(f"- {finding}")
        lines.append("")

        # Ethical
        lines.append("## Ethical Considerations")
        lines.append("")
        if self.ethical.biases_identified:
            lines.append("**Identified Biases**:")
            for b in self.ethical.biases_identified:
                lines.append(f"- {b}")
        if self.ethical.mitigations:
            lines.append("**Mitigations**:")
            for m in self.ethical.mitigations:
                lines.append(f"- {m}")
        lines.append("")

        # Caveats
        if self.caveats:
            lines.append("## Caveats and Recommendations")
            lines.append("")
            for c in self.caveats:
                lines.append(f"- {c}")
            lines.append("")

        # References
        if self.references:
            lines.append("## References")
            lines.append("")
            for ref in self.references:
                lines.append(f"- {ref}")
            lines.append("")

        content = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(content)
            logger.info(f"Model card Markdown written to {output_path}")

        return content


# ---------------------------------------------------------------------------
# Model Card Generator
# ---------------------------------------------------------------------------


class ModelCardGenerator:
    """Generates model cards from model metadata and evaluation results.

    Provides convenience methods to create model cards with sensible defaults
    and integration with the rest of the governance system.

    Example:
        gen = ModelCardGenerator()
        card = gen.generate(
            model_name="Fraud Detector v2",
            model_version="2.0.0",
            model_type=ModelType.ANOMALY_DETECTION,
            description="Graph-based fraud detection on Stellar network",
            authors=["ML Team"],
            performance={"accuracy": 0.95, "f1_score": 0.93},
        )
        card.to_markdown("fraud_detector_v2_card.md")
    """

    def generate(
        self,
        model_name: str,
        model_version: str,
        model_type: ModelType = ModelType.OTHER,
        description: str = "",
        authors: Sequence[str] | None = None,
        framework: str = "pytorch",
        framework_version: str = "",
        license: str = "",
        primary_use: str = "",
        primary_users: Sequence[str] | None = None,
        out_of_scope: Sequence[str] | None = None,
        limitations: Sequence[str] | None = None,
        performance: dict[str, float] | None = None,
        training_data_size: int = 0,
        training_data_sources: Sequence[str] | None = None,
        fairness_metrics: dict[str, dict[str, float]] | None = None,
        biases: Sequence[str] | None = None,
        mitigations: Sequence[str] | None = None,
        caveats: Sequence[str] | None = None,
        references: Sequence[str] | None = None,
    ) -> ModelCard:
        """Generate a model card with the provided metadata.

        Args:
            model_name: Name of the model.
            model_version: Version string.
            model_type: Type of model.
            description: Human-readable description.
            authors: List of author names/teams.
            framework: ML framework used.
            framework_version: Framework version.
            license: License identifier.
            primary_use: Primary intended use case.
            primary_users: Target users.
            out_of_scope: Out-of-scope use cases.
            limitations: Known limitations.
            performance: Dict of metric_name -> value.
            training_data_size: Number of training samples.
            training_data_sources: Data source descriptions.
            fairness_metrics: Group-level fairness metrics.
            biases: Identified biases.
            mitigations: Bias mitigations applied.
            caveats: Additional caveats.
            references: Reference links/papers.

        Returns:
            A complete ModelCard instance.
        """
        # Build model details
        details = ModelDetails(
            name=model_name,
            version=model_version,
            model_type=model_type,
            description=description,
            authors=list(authors or []),
            created_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            framework=framework,
            framework_version=framework_version,
            license=license,
        )

        # Build intended use
        intended = IntendedUse(
            primary_use=primary_use,
            primary_users=list(primary_users or []),
            out_of_scope_uses=list(out_of_scope or []),
            limitations=list(limitations or []),
        )

        # Build training data
        training = TrainingData(
            data_sources=list(training_data_sources or []),
            data_size=training_data_size,
        )

        # Build performance
        perf = PerformanceMetrics()
        if performance:
            for k, v in performance.items():
                if hasattr(perf, k):
                    setattr(perf, k, v)
                else:
                    perf.additional_metrics[k] = v

        # Build fairness
        fairness = FairnessEvaluation(
            metrics=fairness_metrics or {},
            findings=list(biases or []),
        )

        # Build ethical
        ethical = EthicalConsiderations(
            biases_identified=list(biases or []),
            mitigations=list(mitigations or []),
        )

        return ModelCard(
            model_details=details,
            intended_use=intended,
            training_data=training,
            performance=perf,
            fairness=fairness,
            ethical=ethical,
            caveats=list(caveats or []),
            references=list(references or []),
        )

    def generate_from_risk_assessment(
        self,
        model_name: str,
        model_version: str,
        risk_assessment,  # RiskAssessment
        model_type: ModelType = ModelType.OTHER,
    ) -> ModelCard:
        """Generate a model card pre-populated from a risk assessment.

        Args:
            model_name: Model name.
            model_version: Model version.
            risk_assessment: RiskAssessment instance.
            model_type: Model type.

        Returns:
            ModelCard with risk findings as caveats and ethical considerations.
        """
        biases = []
        mitigations = []
        caveats = []

        for finding in risk_assessment.findings:
            if finding.category.value in ("fairness", "bias"):
                biases.append(finding.description)
                if finding.mitigation:
                    mitigations.append(finding.mitigation)
            else:
                caveats.append(f"[{finding.level.value.upper()}] {finding.description}")

        return self.generate(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            biases=biases,
            mitigations=mitigations,
            caveats=caveats,
        )


__all__ = [
    "ModelType",
    "ModelDetails",
    "IntendedUse",
    "TrainingData",
    "PerformanceMetrics",
    "FairnessEvaluation",
    "EthicalConsiderations",
    "ModelCard",
    "ModelCardGenerator",
]