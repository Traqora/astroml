from typing import List, Dict, Any
from api.schemas import CostMetric, BudgetAlert, CostDashboardResponse

class CostMonitoringService:
    def __init__(self):
        self.budget_limit = 1000.0 # $1000 limit
        self.optimization_active = True
        
        # Mock usage data
        self.provider_usage = {
            "OpenAI": {"tokens": 15000000, "cost": 450.0},
            "Anthropic": {"tokens": 5000000, "cost": 150.0},
            "Local_Llama": {"tokens": 20000000, "cost": 50.0}
        }
        
    def _calculate_alerts(self, total_cost: float) -> List[BudgetAlert]:
        alerts = []
        percent_used = (total_cost / self.budget_limit) * 100
        
        for threshold in [80, 90, 100]:
            alerts.append(
                BudgetAlert(
                    threshold_percent=threshold,
                    is_triggered=(percent_used >= threshold)
                )
            )
        return alerts

    def get_dashboard(self) -> CostDashboardResponse:
        metrics = []
        total_cost = 0.0
        
        for provider, usage in self.provider_usage.items():
            metrics.append(
                CostMetric(
                    provider=provider,
                    model="mixed",
                    total_cost=usage["cost"],
                    total_tokens=usage["tokens"]
                )
            )
            total_cost += usage["cost"]
            
        alerts = self._calculate_alerts(total_cost)
        
        return CostDashboardResponse(
            metrics=metrics,
            total_cost=total_cost,
            budget_limit=self.budget_limit,
            alerts=alerts,
            optimization_active=self.optimization_active
        )
        
    def optimize_provider(self, required_capability: str) -> str:
        """
        Automatic optimization: choose cheapest provider
        that meets requirements. (Mocked implementation)
        """
        if self.optimization_active:
            # Simple mock logic: prefer local if we are close to budget
            total_cost = sum(v["cost"] for v in self.provider_usage.values())
            if total_cost > self.budget_limit * 0.8:
                return "Local_Llama"
            return "Anthropic" # cheaper than OpenAI for mock
        return "OpenAI"
