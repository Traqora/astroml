from astroml.utils.exceptions import AstroMLError
"""Benchmark runners to evaluate model performance on datasets."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from astroml.llm.eval.datasets import EvalDataset
from astroml.llm.eval.metrics import calculate_bleu, calculate_custom_scores, calculate_rouge_l


class BenchmarkRunner:
    """Executes prompt datasets against an LLM and measures response quality."""

    def __init__(self, model_name -> Any: str, generation_fn: Callable[[str], Awaitable[str]]):
        self.model_name = model_name
        self.generation_fn = generation_fn

    async def run_benchmark(self, dataset: EvalDataset) -> list[dict[str, Any]]:
        """Run all test cases in the dataset and collect evaluation metrics."""
        results = []

        for item in dataset.items:
            prompt = item["prompt"]
            reference = item["reference"]
            context = item.get("context")

            # Measure generation time/latency
            start_time = time.perf_counter()
            try:
                response = await self.generation_fn(prompt)
                status = "success"
                error_msg = None
            except AstroMLError as e:
                response = ""
                status = "failed"
                error_msg = str(e)

            latency = time.perf_counter() - start_time

            # Compute metrics
            bleu = calculate_bleu(response, reference) if status == "success" else 0.0
            rouge = calculate_rouge_l(response, reference) if status == "success" else 0.0
            custom = (
                calculate_custom_scores(response, reference, context) if status == "success" else {}
            )

            results.append(
                {
                    "test_case_id": item["id"],
                    "prompt": prompt,
                    "response": response,
                    "reference": reference,
                    "status": status,
                    "error": error_msg,
                    "latency_seconds": round(latency, 4),
                    "metrics": {"bleu": round(bleu, 4), "rouge_l": round(rouge, 4), **custom},
                }
            )

        return results
