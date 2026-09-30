"""Model security scoring, test pipeline and report generation.

Resolves part of #645 (procedure steps 5-7).

:class:`SecurityTestPipeline` runs the adversarial, extraction and poisoning
checks in one pass and condenses them into a single 0-100
:class:`SecurityScore` with a letter grade, plus a Markdown report suitable for
attaching to a model release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroml.security.adversarial.attacks import (
    AttackConfig,
    AttackType,
    GradientFn,
    PredictProbaFn,
)
from astroml.security.adversarial.defenses import RobustnessEvaluator, RobustnessReport
from astroml.security.model_extraction import ExtractionVerdict, ModelExtractionDetector
from astroml.security.poisoning_detection import PoisoningDetector, PoisoningReport

__all__ = ["SecurityScore", "SecurityTestPipeline", "SecurityTestResult"]

#: Weight of each check in the composite score.
_WEIGHTS: dict[str, float] = {
    "robustness": 0.5,
    "poisoning": 0.3,
    "extraction": 0.2,
}


def _utcnow_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SecurityScore:
    """Composite security score for one model."""

    overall: float
    grade: str
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the score."""
        return {
            "overall": self.overall,
            "grade": self.grade,
            "components": dict(self.components),
        }


@dataclass(frozen=True)
class SecurityTestResult:
    """Everything :class:`SecurityTestPipeline` produced for one model."""

    model_name: str
    score: SecurityScore
    robustness: RobustnessReport | None = None
    poisoning: PoisoningReport | None = None
    extraction: tuple[ExtractionVerdict, ...] = ()
    generated_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the result."""
        return {
            "model_name": self.model_name,
            "generated_at": self.generated_at,
            "score": self.score.to_dict(),
            "robustness": self.robustness.to_dict() if self.robustness else None,
            "poisoning": self.poisoning.to_dict() if self.poisoning else None,
            "extraction": [verdict.to_dict() for verdict in self.extraction],
        }

    def to_markdown(self) -> str:
        """Render a human-readable security report."""
        lines = [
            f"# Model Security Report — {self.model_name}",
            "",
            f"**Generated:** {self.generated_at}",
            "",
            f"## Score: {self.score.overall:.1f}/100 (grade {self.score.grade})",
            "",
            "| Component | Score |",
            "| --- | --- |",
        ]
        for name, value in self.score.components.items():
            lines.append(f"| {name} | {value:.1f} |")

        if self.robustness is not None:
            lines += ["", "## Adversarial robustness", ""]
            lines.append(f"- Clean accuracy: {self.robustness.clean_accuracy:.3f}")
            for name, result in self.robustness.attack_results.items():
                lines.append(
                    f"- `{name}`: attack success {result.success_rate:.1%}, "
                    f"robust accuracy {1.0 - result.success_rate:.1%}, "
                    f"mean L2 distortion {result.mean_l2_distortion:.4g}"
                )

        if self.poisoning is not None:
            lines += ["", "## Training data poisoning", ""]
            lines.append(
                f"- Contamination rate: {self.poisoning.contamination_rate:.2%} "
                f"({len(self.poisoning.suspicious_indices)} of "
                f"{self.poisoning.sample_count} samples)"
            )
            for finding in self.poisoning.findings:
                lines.append(
                    f"- `{finding.poisoning_type.value}`: "
                    f"{finding.suspicious_count} suspicious samples"
                )

        if self.extraction:
            lines += ["", "## Model extraction", ""]
            for verdict in self.extraction:
                signals = ", ".join(s.value for s in verdict.signals) or "none"
                lines.append(
                    f"- `{verdict.client_id}`: risk {verdict.risk.value} "
                    f"(score {verdict.score:.2f}); signals: {signals}"
                )

        return "\n".join(lines) + "\n"


class SecurityTestPipeline:
    """Runs the full model security suite and scores the outcome.

    Every check is optional: pass only the data you have, and the composite
    score is renormalised over the checks that actually ran.
    """

    def __init__(
        self,
        model_name: str,
        predict_proba: PredictProbaFn,
        *,
        gradient: GradientFn | None = None,
        attack_config: AttackConfig | None = None,
        attacks: tuple[AttackType | str, ...] = (AttackType.FGSM, AttackType.PGD),
        poisoning_detector: PoisoningDetector | None = None,
        extraction_detector: ModelExtractionDetector | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        self.model_name = model_name
        self.predict_proba = predict_proba
        self.gradient = gradient
        self.attack_config = attack_config or AttackConfig()
        self.attacks = attacks
        self.poisoning_detector = poisoning_detector or PoisoningDetector()
        self.extraction_detector = extraction_detector

    def run(
        self,
        *,
        x_eval: NDArray[np.float64] | None = None,
        y_eval: NDArray[np.int_] | None = None,
        x_train: NDArray[np.float64] | None = None,
        y_train: NDArray[np.int_] | None = None,
    ) -> SecurityTestResult:
        """Run every applicable check and return the scored result."""
        robustness: RobustnessReport | None = None
        if x_eval is not None and y_eval is not None:
            evaluator = RobustnessEvaluator(
                self.predict_proba,
                gradient=self.gradient,
                config=self.attack_config,
                attacks=self.attacks,
            )
            robustness = evaluator.evaluate(x_eval, y_eval)

        poisoning: PoisoningReport | None = None
        if x_train is not None and y_train is not None:
            poisoning = self.poisoning_detector.detect(x_train, y_train)

        extraction: tuple[ExtractionVerdict, ...] = ()
        if self.extraction_detector is not None:
            extraction = tuple(self.extraction_detector.assess_all())

        return SecurityTestResult(
            model_name=self.model_name,
            score=self._score(robustness, poisoning, extraction),
            robustness=robustness,
            poisoning=poisoning,
            extraction=extraction,
        )

    def _score(
        self,
        robustness: RobustnessReport | None,
        poisoning: PoisoningReport | None,
        extraction: tuple[ExtractionVerdict, ...],
    ) -> SecurityScore:
        """Condense the individual checks into a weighted 0-100 score."""
        components: dict[str, float] = {}
        if robustness is not None:
            components["robustness"] = 100.0 * robustness.robustness_score
        if poisoning is not None:
            components["poisoning"] = 100.0 * (1.0 - min(poisoning.contamination_rate, 1.0))
        if extraction:
            worst = max(verdict.score for verdict in extraction)
            components["extraction"] = 100.0 * (1.0 - worst)

        if not components:
            return SecurityScore(overall=0.0, grade="N/A", components={})

        total_weight = sum(_WEIGHTS[name] for name in components)
        overall = sum(_WEIGHTS[name] * value for name, value in components.items())
        overall /= total_weight
        return SecurityScore(overall=overall, grade=_grade(overall), components=components)


def _grade(score: float) -> str:
    """Map a 0-100 score onto a letter grade."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"
