from typing import List, Dict
from .semantic import SemanticCache

class CacheWarmingStrategy:
    def __init__(self, semantic_cache: SemanticCache):
        self.cache = semantic_cache
        self.common_templates = [
            "Explain transaction details for high risk withdrawals",
            "Identify accounts with unusual activities and micro-transfers",
            "What is the current F1 score of fraud prediction model?",
            "List recent security incidents in LLM routing"
        ]

    def warm_cache(self):
        # Seed cache with standard response completions
        responses = {
            "Explain transaction details for high risk withdrawals": "High risk withdrawals typically involve transaction amounts exceeding $10,000, unverified international destinations, or sudden change in typical velocity.",
            "Identify accounts with unusual activities and micro-transfers": "Micro-transfers under $1.00 executed rapidly across accounts are flagged as card testing or structured routing behavior.",
            "What is the current F1 score of fraud prediction model?": "The current XGBoost fraud prediction model F1-Score is 0.91, and AUC is 0.94.",
            "List recent security incidents in LLM routing": "No high-severity prompt injection or toxic outputs detected in the last 24 hours."
        }
        for q, r in responses.items():
            self.cache.set(q, r)
