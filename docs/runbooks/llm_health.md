# LLM Infrastructure Runbook

## Overview

This runbook covers health checks, monitoring, alerting, and incident response for LLM providers (OpenAI, Anthropic, HuggingFace).

## Health Check Architecture

- **Health endpoints**: `GET /api/v1/llm/health` and `GET /api/v1/llm/health/{provider}`
- **Polling interval**: 60 seconds via Prometheus or external monitor
- **Metrics endpoint**: `GET /metrics` (Prometheus text format)
- **Grafana dashboard**: `monitoring/grafana/llm_health_dashboard.json`

## Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `astroml_llm_provider_health` | Gauge | 1 = healthy, 0 = unhealthy |
| `astroml_llm_request_latency_seconds` | Histogram | Per-provider latency |
| `astroml_llm_requests_total` | Counter | Request count by provider and status |
| `astroml_llm_cost_usd_total` | Counter | Accumulated cost USD |
| `astroml_llm_tokens_total` | Counter | Token count by provider and token_type |

## Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| `LLMProviderDown` | Provider health == 0 for > 2m | Critical |
| `LLMHighErrorRate` | Error rate > 0.1 req/s for > 2m | Warning |
| `LLMCostThreshold` | Cost > $10 in 1h window | Warning |
| `LLMHighLatency` | P95 latency > 5s for > 3m | Warning |

## Cost Tracking

- **Threshold**: $100 (logged)
- **Granularity**: Per-request cost calculated using mock rates in `astroml/llm/tracker.py`
- **Alerting**: Prometheus `LLMCostThreshold` rule triggers on spikes (>$10/hour)
- **Dashboard**: Cost panel in Grafana shows 1-hour rolling sums

## Incident Response

### Provider Down
1. Check `LLMProviderDown` alert in Alertmanager
2. Verify API keys are configured (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HUGGINGFACE_API_KEY`)
3. Check network connectivity from container to provider API
4. Review provider status pages: OpenAI, Anthropic, HuggingFace
5. Rotate API keys if suspected exposure
6. Failover: update `LLM_PROVIDER` env var to alternate provider

### High Error Rate
1. Check `LLMHighErrorRate` alert
2. Correlate with latency spikes in Grafana dashboard
3. Review application logs for stack traces
4. Check for rate limits or quota exhaustion
5. Consider switching providers or reducing request rate

### Cost Spike
1. Check `LLMCostThreshold` alert
2. Correlate with traffic volume in Grafana
3. Review recent deployments for prompt regression
4. If legitimate growth, update budget thresholds
5. If anomaly, audit prompt caching (`SemanticCache`) and consider tightening limits

## Runbook Verification

```bash
# Verify health endpoint
curl -s http://localhost:8000/api/v1/llm/health | jq

# Verify metrics exposition
curl -s http://localhost:8000/metrics | grep astroml_llm_

# Run monitoring stack
docker compose --profile monitoring up -d

# Check Prometheus targets
open http://localhost:9090/targets

# Open Grafana
open http://localhost:3000
Default login: admin/admin
```

## Maintenance

- **Dashboard refresh**: Import `monitoring/grafana/llm_health_dashboard.json` into Grafana
- **Alert review**: Review rules in `monitoring/prometheus/alert_rules.yml`
- **Rate updates**: Update mock cost rates in `astroml/llm/tracker.py` and `COST_RATES` from provider pricing pages
