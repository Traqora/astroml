try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    # Minimal mock for environment without prometheus_client
    class Counter:
        def __init__(self, name, desc, labelnames=()): self.name = name
        def labels(self, *args, **kwargs): return self
        def inc(self, amount=1): pass
    class Histogram:
        def __init__(self, name, desc, labelnames=(), buckets=None): self.name = name
        def labels(self, *args, **kwargs): return self
        def observe(self, val): pass
    class Gauge:
        def __init__(self, name, desc, labelnames=()): self.name = name
        def labels(self, *args, **kwargs): return self
        def set(self, val): pass
        def inc(self, amount=1): pass
    def generate_latest(): return b""
    CONTENT_TYPE_LATEST = "text/plain"

from .collector import get_metrics_collector

class PrometheusExporter:
    def __init__(self):
        self.request_counter = Counter("llm_requests_total", "Total LLM requests", ["model", "feature", "status"])
        self.token_counter = Counter("llm_tokens_total", "Total tokens used", ["model", "type"])
        self.latency_histogram = Histogram("llm_latency_seconds", "LLM request latency", ["model"], buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0))
        self.cost_gauge = Gauge("llm_spend_usd_total", "Total cumulative spend in USD", ["model"])
        self.cache_counter = Counter("llm_cache_hits_total", "LLM cache hits and misses", ["status"])
        self.safety_incident_counter = Counter("llm_safety_incidents_total", "Total LLM safety incidents")

    def update_metrics(self):
        collector = get_metrics_collector()
        summary = collector.get_summary_metrics()
        
        # In a real environment, we'd update gauges or increment counters as events happen.
        # For simplicity, we expose these metrics.
        # We can also scrape recent items from history to populate counters
        for record in list(collector.history)[-100:]: # sync last 100
            status = "error" if record["error"] else "success"
            self.request_counter.labels(model=record["model"], feature=record["feature"], status=status).inc(0)
            self.token_counter.labels(model=record["model"], type="prompt").inc(0)
            self.token_counter.labels(model=record["model"], type="completion").inc(0)
            self.latency_histogram.labels(model=record["model"]).observe(record["latency"])
            self.cost_gauge.labels(model=record["model"]).set(record["cost"])
            
    def export(self) -> bytes:
        self.update_metrics()
        return generate_latest()

_exporter_instance = PrometheusExporter()

def get_prometheus_exporter() -> PrometheusExporter:
    return _exporter_instance
