"""Human evaluation workflows and annotation storage."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional


class HumanEvaluator:
    """Manages human review feedback and annotations for LLM outputs."""

    def __init__(self, storage_path: str = "data/eval/human_feedback.json"):
        self.storage_path = storage_path
        self.feedback_list: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load human feedback database if it exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path) as f:
                    self.feedback_list = json.load(f)
            except Exception:
                self.feedback_list = []

    def save(self) -> None:
        """Save human feedback database."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.feedback_list, f, indent=2)

    def record_feedback(
        self,
        prompt: str,
        response: str,
        score: int,  # 1 to 5
        annotator: str,
        comments: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record human review score and comments for a prompt response pair."""
        feedback = {
            "id": len(self.feedback_list) + 1,
            "prompt": prompt,
            "response": response,
            "score": max(1, min(5, score)),
            "annotator": annotator,
            "comments": comments,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.feedback_list.append(feedback)
        self.save()
        return feedback

    def get_average_score(self) -> float:
        """Calculate average human rating across all reviews."""
        if not self.feedback_list:
            return 0.0
        return sum(item["score"] for item in self.feedback_list) / len(self.feedback_list)
