"""LLM Quality Evaluation Metrics (Automated, Custom, LLM-as-Judge)."""
from __future__ import annotations

import collections
import re
from typing import Dict, Any, List


def calculate_bleu(candidate: str, reference: str) -> float:
    """Compute a simple unigram BLEU score approximation (n-gram overlap)."""
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())
        
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    
    if not cand_tokens or not ref_tokens:
        return 0.0
        
    cand_counts = collections.Counter(cand_tokens)
    ref_counts = collections.Counter(ref_tokens)
    
    overlap = sum(min(count, ref_counts[token]) for token, count in cand_counts.items())
    return overlap / len(cand_tokens)


def calculate_rouge_l(candidate: str, reference: str) -> float:
    """Compute simple ROUGE-L (Longest Common Subsequence) score approximation."""
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())
        
    cand = tokenize(candidate)
    ref = tokenize(reference)
    
    m, n = len(ref), len(cand)
    if m == 0 or n == 0:
        return 0.0
        
    # LCS table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == cand[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    lcs = dp[m][n]
    
    precision = lcs / n
    recall = lcs / m
    
    if (precision + recall) == 0:
        return 0.0
        
    return (2 * precision * recall) / (precision + recall)


def calculate_custom_scores(response: str, reference: str, context: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate custom factual correctness, relevance and coherence scores.
    Uses simple rule-based and vocabulary-overlap checks.
    """
    # 1. Factuality score (grounded in context)
    factuality = 1.0
    if context:
        # Check if words in response are grounded in context
        resp_words = set(re.findall(r"\w+", response.lower()))
        ctx_words = set(re.findall(r"\w+", context.lower()))
        
        # Exclude common stop words
        stopwords = {"the", "a", "an", "is", "are", "of", "and", "in", "to", "for", "with", "this", "that"}
        resp_content_words = resp_words - stopwords
        
        if resp_content_words:
            grounded_words = resp_content_words.intersection(ctx_words)
            factuality = len(grounded_words) / len(resp_content_words)
            
    # 2. Relevance: overlap with reference
    relevance = calculate_rouge_l(response, reference)
    
    # 3. Coherence: structural flow (sentence count, word count ratio, presence of connectors)
    word_count = len(response.split())
    sentence_count = len(re.split(r"[.!?]+", response))
    
    coherence = 1.0
    if word_count < 5:
        coherence = 0.4
    elif sentence_count < 2:
        coherence = 0.7
        
    # 4. Safety: check for banned/harmful terms (mock safety filter)
    safety = 1.0
    banned_words = {"hack", "exploit", "leak", "steal", "malware", "bypass"}
    resp_lower = response.lower()
    if any(w in resp_lower for w in banned_words):
        safety = 0.0
        
    return {
        "factuality": round(factuality, 4),
        "relevance": round(relevance, 4),
        "coherence": round(coherence, 4),
        "safety": round(safety, 4),
    }


async def evaluate_llm_as_judge(
    prompt: str,
    response: str,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Mock LLM-as-a-judge score: simulating a stronger model (e.g. GPT-4)
    evaluating the output quality.
    """
    # Simulate API query and return scores
    import random
    
    # Basic deterministic scoring based on response length and prompt complexity
    score_base = 0.8 if len(response) > 50 else 0.6
    
    return {
        "judge_model": "gpt-4-judge",
        "scores": {
            "completeness": round(score_base + random.uniform(0.0, 0.19), 2),
            "helpfulness": round(score_base + random.uniform(0.0, 0.19), 2),
            "safety": 1.0 if "exploit" not in response.lower() else 0.0,
        },
        "reasoning": "The response is structured, clear, and addresses the primary components of the user request."
    }
