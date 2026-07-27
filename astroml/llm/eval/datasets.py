"""Test dataset management for LLM evaluations."""
from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Optional


class EvalDataset:
    """Manages test prompt-response pairs for evaluation."""

    def __init__(self, name: str):
        self.name = name
        self.items: List[Dict[str, Any]] = []

    def add_item(
        self,
        prompt: str,
        reference: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a test case to the evaluation dataset."""
        self.items.append({
            "id": len(self.items) + 1,
            "prompt": prompt,
            "reference": reference,
            "context": context,
            "metadata": metadata or {}
        })

    def save(self, filepath: str) -> None:
        """Save dataset to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump({
                "name": self.name,
                "items": self.items
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> EvalDataset:
        """Load dataset from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file {filepath} not found.")
            
        with open(filepath) as f:
            data = json.load(f)
            
        dataset = cls(name=data.get("name", "unnamed"))
        dataset.items = data.get("items", [])
        return dataset


def get_default_dataset() -> EvalDataset:
    """Return standard AstroML Q&A evaluation dataset."""
    dataset = EvalDataset("astroml-qa-pairs")
    
    # 1. Transaction Explanation
    dataset.add_item(
        prompt="Explain transaction GA5W... with 5 operations",
        reference="This transaction contains 5 operations moving XLM and USDC between Stellar accounts.",
        context="Stellar transaction GA5W closed in ledger 41203 with 5 operations: payment of 10 XLM, path payment of 5 USDC...",
        metadata={"category": "explanations"}
    )
    
    # 2. Fraud detection query
    dataset.add_item(
        prompt="Show fraud alerts for risk > 0.8",
        reference="Here are the accounts marked with fraud risk score greater than 0.8: GC3K (0.91), GBD4 (0.85).",
        context="Account GC3K has risk score 0.91 based on velocity anomalies. Account GBD4 has risk score 0.85.",
        metadata={"category": "fraud_query"}
    )
    
    # 3. Model benchmarking query
    dataset.add_item(
        prompt="Compare model v1.2 and v1.3",
        reference="Model v1.3 has higher accuracy (0.94) compared to v1.2 (0.89) but with 20% higher latency.",
        context="v1.2: accuracy=0.89, latency=120ms. v1.3: accuracy=0.94, latency=145ms.",
        metadata={"category": "benchmarking"}
    )
    
    return dataset
