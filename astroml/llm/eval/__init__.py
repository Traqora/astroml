"""LLM Evaluation and Benchmarking Framework Package."""
from __future__ import annotations

from astroml.llm.eval.framework import LLMEvalFramework
from astroml.llm.eval.datasets import EvalDataset, get_default_dataset
from astroml.llm.eval.metrics import calculate_bleu, calculate_rouge_l, calculate_custom_scores
from astroml.llm.eval.regression import QualityRegressionDetector
from astroml.llm.eval.human import HumanEvaluator

__all__ = [
    "LLMEvalFramework",
    "EvalDataset",
    "get_default_dataset",
    "calculate_bleu",
    "calculate_rouge_l",
    "calculate_custom_scores",
    "QualityRegressionDetector",
    "HumanEvaluator",
]
