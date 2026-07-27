"""Post-tuning evaluation for fine-tuned models.

Provides automated evaluation against holdout sets, baseline
comparison, and A/B testing capabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of a fine-tuned model evaluation."""
    model_id: str
    metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    improvement_pct: Dict[str, float]
    num_test_samples: int
    duration_seconds: float
    evaluated_at: datetime = field(default_factory=datetime.utcnow)


class FineTuneEvaluator:
    """Evaluates fine-tuned models against baseline and holdout sets."""

    def __init__(
        self,
        model_id: str,
        model: object,
        baseline_model_name: str = "gpt-3.5-turbo",
    ):
        self.model_id = model_id
        self.model = model
        self.baseline_model_name = baseline_model_name

    def evaluate(
        self,
        test_data: List[Dict[str, str]],
        baseline_model: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate the fine-tuned model against test data.

        Args:
            test_data: List of test records with 'input' and 'output' keys
            baseline_model: Baseline model name for comparison

        Returns:
            EvaluationResult with metrics and comparison
        """
        import time

        baseline = baseline_model or self.baseline_model_name
        start = time.time()

        ft_predictions = self._generate_predictions(test_data)
        ft_metrics = self._compute_metrics(test_data, ft_predictions)

        baseline_predictions = self._generate_baseline_predictions(
            test_data, baseline
        )
        baseline_metrics = self._compute_metrics(test_data, baseline_predictions)

        improvement = {}
        for metric in ft_metrics:
            if metric in baseline_metrics and baseline_metrics[metric] != 0:
                improvement[metric] = (
                    (ft_metrics[metric] - baseline_metrics[metric])
                    / abs(baseline_metrics[metric])
                    * 100
                )
            else:
                improvement[metric] = 0.0

        duration = time.time() - start
        logger.info(
            f"Evaluation complete: {len(test_data)} samples in {duration:.1f}s"
        )

        return EvaluationResult(
            model_id=self.model_id,
            metrics=ft_metrics,
            baseline_metrics=baseline_metrics,
            improvement_pct=improvement,
            num_test_samples=len(test_data),
            duration_seconds=duration,
        )

    def _generate_predictions(
        self,
        test_data: List[Dict[str, str]],
    ) -> List[str]:
        """Generate predictions using the fine-tuned model."""
        predictions = []
        trainer = getattr(self.model, "_predict", None)
        if trainer:
            for record in test_data:
                try:
                    pred = trainer(record["input"])
                    predictions.append(str(pred))
                except Exception as e:
                    logger.warning(f"Prediction failed: {e}")
                    predictions.append("")
        else:
            for record in test_data:
                predictions.append(f"<prediction_for: {record['input'][:50]}>")
        return predictions

    def _generate_baseline_predictions(
        self,
        test_data: List[Dict[str, str]],
        baseline_model: str,
    ) -> List[str]:
        """Generate predictions using the baseline model."""
        predictions = []
        for record in test_data:
            try:
                from astroml.llm.providers.factory import get_llm_provider
                provider = get_llm_provider("openai")
                response = provider.generate(record["input"], model=baseline_model)
                predictions.append(response)
            except Exception as e:
                logger.warning(f"Baseline prediction failed: {e}")
                predictions.append("")
        return predictions

    def _compute_metrics(
        self,
        test_data: List[Dict[str, str]],
        predictions: List[str],
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        exact_matches = 0
        partial_scores = []
        total = len(test_data)

        for i, record in enumerate(test_data):
            expected = record.get("output", "")
            predicted = predictions[i] if i < len(predictions) else ""

            if predicted == expected:
                exact_matches += 1

            score = self._compute_similarity(expected, predicted)
            partial_scores.append(score)

        accuracy = exact_matches / max(total, 1)
        avg_similarity = float(np.mean(partial_scores)) if partial_scores else 0.0
        std_similarity = float(np.std(partial_scores)) if len(partial_scores) > 1 else 0.0

        return {
            "exact_match_accuracy": accuracy,
            "average_similarity": avg_similarity,
            "similarity_std": std_similarity,
            "num_samples": total,
        }

    def _compute_similarity(self, expected: str, predicted: str) -> float:
        """Compute similarity between expected and predicted outputs."""
        if not expected and not predicted:
            return 1.0
        if not expected or not predicted:
            return 0.0

        set_a = set(expected.lower().split())
        set_b = set(predicted.lower().split())
        intersection = set_a & set_b
        union = set_a | set_b
        if not union:
            return 1.0
        return len(intersection) / len(union)
