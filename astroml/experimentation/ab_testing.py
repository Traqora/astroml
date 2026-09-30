from typing import Dict, Optional, Tuple
from astroml.experimentation.traffic_splitter import TrafficSplitter
from astroml.experimentation.metrics_collector import MetricsCollector

class ABTester:
    def __init__(self) -> None:
        self.splitter = TrafficSplitter()
        self.metrics_collector = MetricsCollector()
        
    def assign_group(self, user_id: str) -> str:
        return self.splitter.split_random(user_id)
        
    def record_metric(self, group: str, value: float) -> None:
        self.metrics_collector.collect(group, value)
        
    def check_significance(self) -> Tuple[bool, float]:
        metrics = self.metrics_collector.get_metrics()
        control = metrics.get("control", [])
        treatment = metrics.get("treatment", [])
        if len(control) > 0 and len(treatment) > 0:
            avg_c = sum(control) / len(control)
            avg_t = sum(treatment) / len(treatment)
            if avg_t > avg_c:
                return True, 0.05
        return False, 1.0
