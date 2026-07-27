from prometheus_client import Counter, Gauge, Histogram

LLM_REQUESTS_TOTAL = Counter(
    "astroml_llm_requests_total",
    "Total LLM API requests",
    ["provider", "status"],
)

LLM_REQUEST_LATENCY_SECONDS = Histogram(
    "astroml_llm_request_latency_seconds",
    "LLM API request latency in seconds",
    ["provider"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60],
)

LLM_COST_USD_TOTAL = Counter(
    "astroml_llm_cost_usd_total",
    "Total LLM API cost in USD",
    ["provider"],
)

LLM_TOKENS_TOTAL = Counter(
    "astroml_llm_tokens_total",
    "Total LLM tokens processed",
    ["provider", "token_type"],
)

LLM_PROVIDER_HEALTH = Gauge(
    "astroml_llm_provider_health",
    "LLM provider health status (1=healthy, 0=unhealthy)",
    ["provider"],
)
