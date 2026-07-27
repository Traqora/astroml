"""Tests for the LLM Evaluation Framework."""
from __future__ import annotations

import pytest
import tempfile
import os
import json

from astroml.llm.eval import (
    LLMEvalFramework,
    EvalDataset,
    get_default_dataset,
    calculate_bleu,
    calculate_rouge_l,
    calculate_custom_scores,
    QualityRegressionDetector,
    HumanEvaluator,
)


def test_automated_metrics():
    """Test unigram BLEU and ROUGE-L approximation calculations."""
    candidate = "The stellar account has a high risk score."
    reference = "The stellar account has high risk."
    
    bleu = calculate_bleu(candidate, reference)
    rouge = calculate_rouge_l(candidate, reference)
    
    assert bleu > 0.5
    assert rouge > 0.5
    
    # Exact match should yield 1.0
    assert calculate_bleu("hello world", "hello world") == 1.0
    assert calculate_rouge_l("hello world", "hello world") == 1.0


def test_custom_scores():
    """Test custom factual correctness, relevance, and safety metrics."""
    response = "Account GA5W is safe and has zero risk."
    reference = "Account GA5W has been verified as safe."
    context = "GA5W is safe."
    
    scores = calculate_custom_scores(response, reference, context)
    
    assert "factuality" in scores
    assert "relevance" in scores
    assert "coherence" in scores
    assert "safety" in scores
    
    assert scores["safety"] == 1.0
    
    # Test safety flags harmful terms
    harmful_response = "We will hack and exploit the Stellar network."
    harmful_scores = calculate_custom_scores(harmful_response, reference, context)
    assert harmful_scores["safety"] == 0.0


def test_dataset_management():
    """Test dataset creation, save, and load lifecycle."""
    dataset = EvalDataset("test-dataset")
    dataset.add_item(prompt="Test prompt", reference="Test reference", context="Test context")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_dataset.json")
        dataset.save(filepath)
        
        loaded = EvalDataset.load(filepath)
        assert loaded.name == "test-dataset"
        assert len(loaded.items) == 1
        assert loaded.items[0]["prompt"] == "Test prompt"
        assert loaded.items[0]["reference"] == "Test reference"


def test_regression_detector():
    """Test regression checker with simulated scores."""
    detector = QualityRegressionDetector()
    
    # Current metrics above or equal to baseline
    current = {
        "bleu": 0.8,
        "rouge_l": 0.85,
        "factuality": 0.9,
        "relevance": 0.8,
        "safety": 1.0
    }
    
    has_reg, details = detector.check_regression(current)
    assert not has_reg
    assert len(details) == 0
    
    # Regressed metric (e.g. factuality drops to 0.5, way below baseline 0.85)
    regressed = {
        "bleu": 0.8,
        "rouge_l": 0.8,
        "factuality": 0.5,
        "relevance": 0.8,
        "safety": 1.0
    }
    has_reg_2, details_2 = detector.check_regression(regressed)
    assert has_reg_2
    assert len(details_2) > 0
    assert any("factuality" in d for d in details_2)


def test_human_evaluator():
    """Test human reviews and average rating recording."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = os.path.join(tmpdir, "human_feedback.json")
        evaluator = HumanEvaluator(storage_path=storage_path)
        
        evaluator.record_feedback(
            prompt="Hello",
            response="Hi there",
            score=4,
            annotator="Alice",
            comments="Good"
        )
        
        evaluator.record_feedback(
            prompt="Hello",
            response="Hi there",
            score=2,
            annotator="Bob",
            comments="Poor"
        )
        
        assert len(evaluator.feedback_list) == 2
        assert evaluator.get_average_score() == 3.0


@pytest.mark.asyncio
async def test_evaluation_framework():
    """Test evaluation orchestrator runs and reports success."""
    async def mock_generator(prompt: str) -> str:
        return "This is a mock response that matches a high risk score account explanation."
        
    framework = LLMEvalFramework("mock-gpt", mock_generator)
    dataset = EvalDataset("mock-ds")
    dataset.add_item(
        prompt="Explain account GC3K",
        reference="This is an account with high risk score.",
        context="GC3K has risk score 0.9."
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override default paths for testing
        framework.regression_detector.baseline_metrics = {
            "bleu": 0.1,
            "rouge_l": 0.1,
        }
        
        result = await framework.run_full_evaluation(dataset=dataset)
        
        assert result["model_name"] == "mock-gpt"
        assert result["dataset_name"] == "mock-ds"
        assert "metrics" in result
        assert "report_path" in result
        assert os.path.exists(result["report_path"])
        
        # Clean up output report
        if os.path.exists(result["report_path"]):
            os.remove(result["report_path"])
