# Alerting Configuration (Issue #570)

## Overview

Comprehensive alerting system for critical failures with defined SLOs, error budgets, and automated routing.

## Service Level Objectives (SLOs)

| Metric | SLO | Error Budget | Alert Threshold |
|--------|-----|-------------|----------------|
| API Availability | 99.9% | 0.1% | Error rate > 0.1% |
| Database Query Latency | P99 < 500ms | - | P99 > 500ms |
| Ingestion Error Rate | <1% | 1% | Error rate > 1% |
| Ingestion Lag | < 1000 ledgers | - | Lag > 1000 ledgers |
| Disk Space | > 20% | - | Space < 20% |

## Alert Routing

### Critical Alerts
**Severity**: Critical  
**Routing**: PagerDuty/Opsgenie + Slack (#astroml-critical)  
**Response Time**: Immediate (< 5 minutes)

**Examples**:
- API availability below 99.9%
- Database pool exhausted
- Database query latency > 500ms
- Ingestion lag > 1000 ledgers
- Disk space < 20%
- Memory < 10%

### Warning Alerts
**Severity**: Warning  
**Routing**: Slack (#astroml-alerts)  
**Response Time**: Within 30 minutes

**Examples**:
- Database pool > 80% utilization
- API P95 latency > 1s
- LLM high latency
- CPU usage > 90%
- Rate limiting

### Info Alerts
**Severity**: Info  
**Routing**: Email digest  
**Response Time**: Next business day

**Examples**:
- Deployment completions
- Scheduled maintenance
- Performance improvements

## Alert Suppression

### Maintenance Windows

Suppress alerts during planned maintenance:

```bash
# Label pods for maintenance
kubectl label -n astroml pod <pod-name> maintenance="true"

# Remove maintenance label
kubectl label -n astroml pod <pod-name> maintenance-
```

### Alertmanager Configuration

Add suppression rules to `monitoring/prometheus/alertmanager.yml`:

```yaml
inhibit_rules:
  # Suppress warnings if critical is firing
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname']
```

## Runbooks

Each alert includes a runbook reference in the `runbook` label:

- `docs/runbooks/api_availability.md` - API availability issues
- `docs/runbooks/ingestion_lag.md` - Ingestion lag issues
- `docs/runbooks/db_query_latency.md` - Database query latency
- `docs/runbooks/db_pool_exhaustion.md` - Database pool exhaustion
- `docs/runbooks/disk_space_low.md` - Disk space issues
- `docs/runbooks/memory_high.md` - Memory issues
- `docs/runbooks/cpu_high.md` - CPU issues

## Synthetic Monitoring

External monitoring configured via UptimeRobot or similar:

### Monitored Endpoints
- `https://api.astroml.example.com/healthz`
- `https://api.astroml.example.com/api/v1/models`
- `https://api.astroml.example.com/api/v1/healthz/db`

### Check Frequency
- Every 1 minute for critical endpoints
- Every 5 minutes for non-critical endpoints

### Alert Thresholds
- 2 consecutive failures → Warning
- 5 consecutive failures → Critical

## Testing Alerts

### Test Alert Firing

```bash
# Trigger test alert
kubectl exec -n monitoring prometheus -- promtool check rules monitoring/prometheus/alert_rules.yml

# Test Alertmanager config
kubectl exec -n monitoring alertmanager -- amtool config check
```

### Test Routing

```bash
# Send test alert
curl -XPOST http://alertmanager:9093/api/v1/alerts -d '[
  {
    "labels": {
      "alertname": "TestAlert",
      "severity": "critical"
    },
    "annotations": {
      "description": "Test alert for routing verification"
    }
  }
]'
```

## Configuration

### Environment Variables

```bash
# Slack webhook for alerts
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# PagerDuty service key
export PAGERDUTY_SERVICE_KEY="..."

# Opsgenie API key (alternative to PagerDuty)
export OPSGENIE_API_KEY="..."
```

### Alertmanager Secrets

Store sensitive configuration in Kubernetes secrets:

```bash
kubectl create secret generic alertmanager-secrets \
  --from-literal=slack-webhook-url="$SLACK_WEBHOOK_URL" \
  --from-literal=pagerduty-service-key="$PAGERDUTY_SERVICE_KEY" \
  -n monitoring
```

## Monitoring Alert Health

### Check Alertmanager Status

```bash
# Check Alertmanager health
kubectl exec -n monitoring alertmanager -- wget -qO- http://localhost:9093/-/healthy

# Check active alerts
kubectl exec -n monitoring alertmanager -- wget -qO- http://localhost:9093/api/v1/alerts
```

### Check Prometheus Alert Status

```bash
# Check firing alerts
kubectl exec -n monitoring prometheus -- wget -qO- http://localhost:9090/api/v1/alerts | jq '.data[] | select(.state=="firing")'
```

## Best Practices

1. **Silence Appropriately**: Use silences for known issues, not to ignore problems
2. **Document Everything**: Keep runbooks updated with actual resolution steps
3. **Test Regularly**: Verify alert routing works as expected
4. **Review Thresholds**: Adjust SLOs based on actual requirements
5. **Learn from Incidents**: Update alerts after each incident

## Related Documentation

- [Alert Rules](../monitoring/prometheus/alert_rules.yml)
- [Alertmanager Config](../monitoring/prometheus/alertmanager.yml)
- [Runbooks](../docs/runbooks/)
- [SLO Documentation](#service-level-objectives-slos)
