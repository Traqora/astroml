import random
from typing import List, Dict, Any

class SyntheticFraudGenerator:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
        # Design prompt templates
        self.prompt_templates = {
            "wash_trading": "Generate a synthetic wash trading pattern involving {n_accounts} accounts with {n_tx} transactions over {duration} hours.",
            "pump_and_dump": "Create a realistic pump and dump scheme sequence for an illiquid asset involving {n_tx} rapid buys followed by massive sells.",
            "phishing_funnel": "Simulate a phishing funnel where funds from {n_accounts} victims are aggregated into a central wallet and dispersed.",
            "layering": "Generate a transaction layering pattern mimicking money laundering across {n_accounts} hops.",
            "rug_pull": "Simulate a liquidity pool rug pull scenario with initial liquidity provision followed by a drain transaction."
        }
        
    def generate_pattern(self, fraud_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement synthetic generator
        """
        if fraud_type not in self.prompt_templates:
            raise ValueError(f"Unknown fraud type: {fraud_type}")
            
        template = self.prompt_templates[fraud_type]
        prompt = template.format(**params)
        
        if self.llm_client:
            response = self.llm_client.generate(prompt)
            pattern_data = response.get("json", {"transactions": []})
        else:
            # Mock pattern generation
            pattern_data = {
                "fraud_type": fraud_type,
                "transactions": [
                    {"tx_id": f"mock_{i}", "amount": random.uniform(100, 10000)}
                    for i in range(params.get("n_tx", 5))
                ],
                "metadata": {"generated_with": "mock"}
            }
            
        return pattern_data

    def validate_realism(self, pattern: Dict[str, Any]) -> float:
        """
        Implement realism validator.
        Returns a score between 0.0 and 1.0 indicating how realistic the pattern is.
        """
        transactions = pattern.get("transactions", [])
        if not transactions:
            return 0.0
            
        # Basic heuristic validation
        score = 1.0
        
        # Penalize identical amounts if not typical for the fraud type
        amounts = [tx.get("amount") for tx in transactions if "amount" in tx]
        if len(amounts) > 1 and len(set(amounts)) == 1:
            score -= 0.3
            
        # Add pattern diversity check
        if pattern.get("fraud_type") == "wash_trading" and len(transactions) < 3:
            score -= 0.5
            
        return max(0.0, score)
        
    def generate_diverse_dataset(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Add pattern diversity by generating a mix of different fraud types.
        """
        fraud_types = list(self.prompt_templates.keys())
        dataset = []
        
        for _ in range(num_samples):
            f_type = random.choice(fraud_types)
            params = {
                "n_accounts": random.randint(2, 20),
                "n_tx": random.randint(5, 50),
                "duration": random.randint(1, 48)
            }
            
            pattern = self.generate_pattern(f_type, params)
            realism_score = self.validate_realism(pattern)
            
            if realism_score > 0.85: # Acceptance criteria
                dataset.append({
                    "pattern": pattern,
                    "score": realism_score
                })
                
        return dataset
