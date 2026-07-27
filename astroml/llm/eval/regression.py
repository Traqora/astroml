"""Regression suite to detect LLM quality degradation."""
from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Tuple


class QualityRegressionDetector:
    """Compares current benchmark results against baseline to catch regression."""

    def __init__(self, baseline_path: str = "data/eval/baseline.json"):
        self.baseline_path = baseline_path
        self.baseline_metrics: Dict[str, float] = {}
        self._load_baseline()

    def _load_baseline(self) -> None:
        """Load baseline metrics if available, otherwise set default thresholds."""
        if os.path.exists(self.baseline_path):
            try:
                with open(self.baseline_path) as f:
                    self.baseline_metrics = json.load(f)
            except Exception:
                pass
        
        # Standard default quality thresholds
        if not self.baseline_metrics:
            self.baseline_metrics = {
                "bleu": 0.65,
                "rouge_l": 0.70,
                "factuality": 0.85,
                "relevance": 0.70,
                "safety": 0.95
            }

    def save_as_baseline(self, current_metrics: Dict[str, float]) -> None:
        """Save the current metrics as the new baseline for future checks."""
        os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)
        with open(self.baseline_path, "w") as f:
            json.dump(current_metrics, f, indent=2)
        self.baseline_metrics = current_metrics

    def check_regression(
        self,
        current_metrics: Dict[str, float],
        tolerance: float = 0.05,
    ) -> Tuple[bool, List[str]]:
        """
        Check if current metrics regress compared to baseline.
        Returns (has_regression, list_of_regressed_metrics).
        """
        regressions = []
        
        for metric, baseline_val in self.baseline_metrics.items():
            if metric not in current_metrics:
                continue
                
            curr_val = current_metrics[metric]
            threshold = baseline_val - (baseline_val * tolerance)
            
            if curr_val < threshold:
                regressions.append(
                    f"Regression detected in metric '{metric}': current {curr_val:.4f} is below threshold {threshold:.4f} (baseline {baseline_val:.4f})"
                )
                
        return len(regressions) > 0, regressions
