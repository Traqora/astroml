import time
import math
from typing import Dict, Any, List, Optional
from collections import deque

class MetricsCollector:
    def __init__(self):
        # In-memory circular buffers for recent requests (timestamp, latency, tokens, error, cost, feature, model, is_cached, ttft, safety_incident, feedback)
        self.history: deque = deque(maxlen=5000)
        self.features: Dict[str, Dict[str, Any]] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
        
        # Benchmarks/Alert limits/Historical metrics
        self.daily_spend = 0.0
        self.weekly_spend = 0.0
        self.monthly_spend = 0.0
        self.start_time = time.time()
        
    def record_request(
        self,
        latency: float,
        tokens_prompt: int,
        tokens_completion: int,
        error: bool = False,
        cost: float = 0.0,
        feature: str = "default",
        model: str = "gpt-4o",
        is_cached: bool = False,
        ttft: float = 0.0,
        safety_incident: bool = False,
        feedback: Optional[int] = None, # 1-5 or binary
        eval_score: float = 1.0,
        hallucination_rate: float = 0.0
    ):
        now = time.time()
        # Add token calculations
        total_tokens = tokens_prompt + tokens_completion
        
        self.history.append({
            "timestamp": now,
            "latency": latency,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "total_tokens": total_tokens,
            "error": error,
            "cost": cost,
            "feature": feature,
            "model": model,
            "is_cached": is_cached,
            "ttft": ttft,
            "safety_incident": safety_incident,
            "feedback": feedback,
            "eval_score": eval_score,
            "hallucination_rate": hallucination_rate
        })
        
        # Accumulate spends
        self.daily_spend += cost
        self.weekly_spend += cost
        self.monthly_spend += cost
        
        # Update feature cost
        if feature not in self.features:
            self.features[feature] = {"cost": 0.0, "requests": 0}
        self.features[feature]["cost"] += cost
        self.features[feature]["requests"] += 1
        
        # Update model cost
        if model not in self.models:
            self.models[model] = {"cost": 0.0, "requests": 0, "latency_sum": 0.0}
        self.models[model]["cost"] += cost
        self.models[model]["requests"] += 1
        self.models[model]["latency_sum"] += latency

    def get_summary_metrics(self) -> Dict[str, Any]:
        now = time.time()
        one_min_ago = now - 60.0
        
        recent = [r for r in self.history if r["timestamp"] >= one_min_ago]
        total_recent = len(recent)
        
        # Requests/min and tokens/sec
        requests_per_min = total_recent
        total_tokens_recent = sum(r["total_tokens"] for r in recent)
        tokens_per_sec = total_tokens_recent / 60.0 if total_recent > 0 else 0.0
        
        # Latency percentiles
        latencies = sorted([r["latency"] for r in self.history])
        p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0.0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0
        
        # Success / error rate
        errors = sum(1 for r in self.history if r["error"])
        success_rate = (len(self.history) - errors) / len(self.history) if self.history else 1.0
        error_rate = 1.0 - success_rate
        
        # Cache hit rate
        cached_count = sum(1 for r in self.history if r["is_cached"])
        cache_hit_rate = cached_count / len(self.history) if self.history else 0.0
        
        # Safety incidents
        guarded_requests = len(self.history)
        safety_incidents = sum(1 for r in self.history if r["safety_incident"])
        false_positive_rate = 0.02 # Placeholder
        
        # Feedback & evaluation
        eval_scores = [r["eval_score"] for r in self.history]
        avg_eval_score = sum(eval_scores) / len(eval_scores) if eval_scores else 1.0
        
        hallucination_rates = [r["hallucination_rate"] for r in self.history]
        avg_hallucination_rate = sum(hallucination_rates) / len(hallucination_rates) if hallucination_rates else 0.0
        
        feedbacks = [r["feedback"] for r in self.history if r["feedback"] is not None]
        avg_feedback = sum(feedbacks) / len(feedbacks) if feedbacks else 5.0
        
        # Cost forecasting
        elapsed_seconds = max(now - self.start_time, 1.0)
        cost_per_sec = self.monthly_spend / elapsed_seconds
        projected_monthly = cost_per_sec * (30 * 86400)
        
        return {
            "requests_per_min": requests_per_min,
            "tokens_per_sec": tokens_per_sec,
            "p50_latency": p50,
            "p95_latency": p95,
            "p99_latency": p99,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "daily_spend": self.daily_spend,
            "weekly_spend": self.weekly_spend,
            "monthly_spend": self.monthly_spend,
            "cost_by_feature": {f: self.features[f]["cost"] for f in self.features},
            "cost_by_model": {m: self.models[m]["cost"] for m in self.models},
            "projected_monthly_cost": projected_monthly,
            "cache_hit_rate": cache_hit_rate,
            "avg_eval_score": avg_eval_score,
            "avg_hallucination_rate": avg_hallucination_rate,
            "avg_feedback": avg_feedback,
            "guarded_requests": guarded_requests,
            "safety_incidents": safety_incidents,
            "false_positive_rate": false_positive_rate
        }

    def detect_anomalies(self) -> List[str]:
        anomalies = []
        if len(self.history) < 20:
            return anomalies
            
        latencies = [r["latency"] for r in self.history]
        mean_l = sum(latencies) / len(latencies)
        variance = sum((x - mean_l) ** 2 for x in latencies) / len(latencies)
        std_l = math.sqrt(variance)
        
        recent_latencies = [r["latency"] for r in self.history][-5:]
        recent_mean = sum(recent_latencies) / len(recent_latencies)
        
        if std_l > 0 and (recent_mean - mean_l) > 3 * std_l:
            anomalies.append("Latency anomaly detected: recent average is 3+ std devs higher than history.")
            
        errors_recent = sum(1 for r in list(self.history)[-20:] if r["error"])
        if errors_recent > 4: # 20% error rate in last 20 requests
            anomalies.append("Error rate anomaly detected: more than 20% of recent requests failed.")
            
        return anomalies

_instance = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    return _instance
