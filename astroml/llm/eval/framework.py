"""LLM Evaluation Orchestration Framework."""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable

from astroml.llm.eval.datasets import EvalDataset, get_default_dataset
from astroml.llm.eval.benchmarks import BenchmarkRunner
from astroml.llm.eval.regression import QualityRegressionDetector
from astroml.llm.eval.human import HumanEvaluator
from astroml.llm.eval.reporting import generate_eval_report

logger = logging.getLogger(__name__)


class LLMEvalFramework:
    """Orchestrates LLM evaluation benchmarks, regression checks, and reporting."""

    def __init__(self, model_name: str, generation_fn: Callable[[str], Awaitable[str]]):
        self.model_name = model_name
        self.runner = BenchmarkRunner(model_name, generation_fn)
        self.regression_detector = QualityRegressionDetector()
        self.human_evaluator = HumanEvaluator()

    async def run_full_evaluation(
        self,
        dataset: Optional[EvalDataset] = None,
        save_baseline: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full evaluation workflow: run benchmarks, check regressions,
        generate markdown report.
        """
        if dataset is None:
            dataset = get_default_dataset()
            
        logger.info("Starting LLM evaluation run on dataset '%s' with model '%s'", dataset.name, self.model_name)
        
        # 1. Run benchmarks
        results = await self.runner.run_benchmark(dataset)
        
        # Calculate averages for current run
        metrics_sum: Dict[str, float] = {}
        successful_runs = sum(1 for r in results if r["status"] == "success")
        
        for r in results:
            if r["status"] == "success":
                for k, v in r.get("metrics", {}).items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                    
        avg_metrics = {k: v / successful_runs for k, v in metrics_sum.items()} if successful_runs > 0 else {}
        
        # 2. Check for regression
        has_regression, regression_details = self.regression_detector.check_regression(avg_metrics)
        
        # Save baseline if requested
        if save_baseline and not has_regression:
            self.regression_detector.save_as_baseline(avg_metrics)
            
        # 3. Generate report
        report_path = generate_eval_report(self.model_name, results)
        
        return {
            "model_name": self.model_name,
            "dataset_name": dataset.name,
            "metrics": avg_metrics,
            "has_regression": has_regression,
            "regressions": regression_details,
            "report_path": report_path,
            "results": results
        }
