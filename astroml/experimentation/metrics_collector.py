from typing import Dict, List

class MetricsCollector:
    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = {"control": [], "treatment": []}
        
    def collect(self, group: str, value: float) -> None:
        if group in self.metrics:
            self.metrics[group].append(value)
            
    def get_metrics(self) -> Dict[str, List[float]]:
        return self.metrics
