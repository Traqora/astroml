"""Feature computation functions for LLM-generated features.

Provides the actual computation logic that calls LLM providers
to generate embeddings, scores, confidence, and uncertainty estimates.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from astroml.llm.providers.factory import get_llm_provider
from astroml.llm.providers.embedding_router import build_default_router

logger = logging.getLogger(__name__)


def compute_embeddings(
    texts: List[str],
    provider: str = "openai",
    model: str = "text-embedding-ada-002",
) -> List[List[float]]:
    """Compute embeddings for a list of texts using the configured provider."""
    if not texts:
        return []
    try:
        router = build_default_router()
        embeddings = router.embed(texts, provider=provider, model=model)
        return embeddings
    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        raise


def compute_fraud_scores(
    data: pd.DataFrame,
    entity_col: str,
    model: str = "gpt-4",
    prompt_version: str = "v1",
) -> List[float]:
    """Compute fraud probability scores for transactions using LLM."""
    scores = []
    provider = get_llm_provider("openai")

    for _, row in data.iterrows():
        prompt = _build_fraud_prompt(row, prompt_version)
        try:
            start = time.time()
            response = provider.generate_detailed(prompt, model=model)
            score = _parse_score(response.text)
            scores.append(score)
        except Exception as e:
            logger.warning(f"Fraud scoring failed for {row.get(entity_col)}: {e}")
            scores.append(0.5)

    return scores


def compute_confidence_scores(
    data: pd.DataFrame,
    entity_col: str,
    model: str = "gpt-4",
    prompt_version: str = "v1",
) -> List[float]:
    """Compute confidence scores for LLM explanations."""
    confidence = []
    provider = get_llm_provider("openai")

    for _, row in data.iterrows():
        prompt = _build_confidence_prompt(row, prompt_version)
        try:
            start = time.time()
            response = provider.generate_detailed(prompt, model=model)
            score = _parse_score(response.text)
            confidence.append(score)
        except Exception as e:
            logger.warning(f"Confidence scoring failed for {row.get(entity_col)}: {e}")
            confidence.append(0.0)

    return confidence


def compute_uncertainty(
    data: pd.DataFrame,
    entity_col: str,
    model: str = "gpt-4",
    num_samples: int = 5,
) -> pd.DataFrame:
    """Compute uncertainty estimates via Monte Carlo sampling of LLM outputs."""
    results: List[Dict[str, float]] = []
    provider = get_llm_provider("openai")

    for _, row in data.iterrows():
        samples = []
        for _ in range(num_samples):
            prompt = _build_uncertainty_prompt(row)
            try:
                response = provider.generate_detailed(prompt, model=model)
                score = _parse_score(response.text)
                samples.append(score)
            except Exception:
                continue

        if samples:
            results.append({
                "uncertainty_mean": float(np.mean(samples)),
                "uncertainty_std": float(np.std(samples)),
                "uncertainty_min": float(np.min(samples)),
                "uncertainty_max": float(np.max(samples)),
            })
        else:
            results.append({
                "uncertainty_mean": 0.0,
                "uncertainty_std": 1.0,
                "uncertainty_min": 0.0,
                "uncertainty_max": 1.0,
            })

    return pd.DataFrame(results, index=data[entity_col].values)


def _build_fraud_prompt(row: pd.Series, version: str = "v1") -> str:
    """Build prompt for fraud probability scoring."""
    return (
        f"Analyze this transaction for fraud risk (version {version}):\n"
        f"Amount: {row.get('amount', 'N/A')}\n"
        f"Description: {row.get('description', row.get('memo', 'N/A'))}\n"
        f"Return a single number between 0 and 1 indicating fraud probability."
    )


def _build_confidence_prompt(row: pd.Series, version: str = "v1") -> str:
    """Build prompt for explanation confidence scoring."""
    return (
        f"Rate your confidence in this explanation (version {version}):\n"
        f"Explanation: {row.get('explanation', 'N/A')}\n"
        f"Return a single number between 0 and 1 indicating confidence."
    )


def _build_uncertainty_prompt(row: pd.Series) -> str:
    """Build prompt for uncertainty estimation."""
    return (
        f"Given this input, provide a fraud risk score between 0 and 1:\n"
        f"Input: {row.get('input_text', row.get('description', 'N/A'))}\n"
        f"Score:"
    )


def _parse_score(text: str) -> float:
    """Parse a numerical score from LLM response text."""
    import re
    matches = re.findall(r'0\.\d+|1\.0|0|1', text.strip())
    if matches:
        score = float(matches[0])
        return max(0.0, min(1.0, score))
    return 0.5
