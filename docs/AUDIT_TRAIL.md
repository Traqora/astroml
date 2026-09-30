# Audit Trail Documentation

## Overview

The AstroML audit trail system provides comprehensive logging of sensitive API operations for security audits and compliance (issues #332, #535).

## Features

### Enhanced Audit Logging (Issue #535)

- **Request Parameter Logging**: Captures and sanitizes request parameters
- **Sensitive Data Redaction**: Automatically redacts sensitive fields (passwords, tokens, API keys)
- **API Key Tracking**: Tracks which API key was used for each request
- **IP Address Logging**: Captures client IP addresses with proxy support
- **User-Agent Logging**: Records client user-agent strings
- **Tamper-Resistant**: Append-only database storage prevents modification
- **90-Day Retention**: Automatic cleanup of logs older than 90 days

### Logged Information

Each audit log entry includes:

- **Timestamp**: ISO 8601 format (UTC)
- **User Identity**: User ID and username
- **Authentication Type**: How the user authenticated (api_key, session, etc.)
- **API Key ID**: Which API key was used (if applicable)
- **Endpoint and Method**: Request path and HTTP method
- **Request Parameters**: Sanitized query and body parameters
- **Response Status**: HTTP status code
- **IP Address**: Client IP address (with proxy support)
- **User-Agent**: Client user-agent string

### Sensitive Fields Redacted

The following fields are automatically redacted from audit logs:

- `password`
- `token`
- `api_key`
- `secret`
- `credit_card`
- `ssn`
- `social_security`
- `auth`
- `authorization`

## API Endpoints

### Search Audit Logs

```http
GET /api/v1/audit/logs
```

**Query Parameters:**
- `user_id`: Filter by user ID
- `action`: Filter by action (create, update, delete, login, logout)
- `resource_type`: Filter by resource type
- `resource_id`: Filter by resource ID
- `start_date`: Filter by start date
- `end_date`: Filter by end date
- `limit`: Maximum results (default: 100, max: 1000)
- `offset`: Pagination offset

**Required Scope:** `audit:read`

### Export Audit Logs

```http
GET /api/v1/audit/export
```

**Query Parameters:**
- `user_id`: Filter by user ID
- `action`: Filter by action
- `resource_type`: Filter by resource type
- `start_date`: Filter by start date
- `end_date`: Filter by end date

**Required Scope:** `audit:export`

### Rotate Audit Logs

```http
POST /api/v1/audit/rotate
```

Manually trigger deletion of logs older than retention period.

**Required Scope:** `audit:admin`

### Get Audit Statistics

```http
GET /api/v1/audit/stats
```

Returns:
- Total log count
- Retention period
- Maximum records

**Required Scope:** `audit:read`

## Access Control

### Scopes

- `audit:read`: Read audit logs and statistics
- `audit:export`: Export audit logs
- `audit:admin`: Rotate logs and administrative functions

### Privacy Considerations

**Data Minimization:**
- Only sensitive operations are logged
- Sensitive fields are automatically redacted
- Request parameters are sanitized before storage

**Access Restrictions:**
- Audit log access requires specific scopes
- All access is logged in the audit trail itself
- Export functionality requires elevated permissions

**Retention Policy:**
- Logs are retained for 90 days by default
- Automatic cleanup removes old logs
- Manual rotation available for immediate cleanup

**Data Protection:**
- Logs stored in append-only database table
- No direct modification of existing log entries
- IP addresses and user-agents captured for security analysis

## Configuration

### Environment Variables

```bash
# Audit retention period in days (default: 90)
AUDIT_RETENTION_DAYS=90

# Maximum audit log records (default: 1000000)
AUDIT_MAX_RECORDS=1000000
```

### Middleware Configuration

The audit middleware is automatically enabled for sensitive operations:

- POST, PUT, PATCH, DELETE requests
- Authentication endpoints (login, logout)
- User management endpoints
- API key management endpoints

## Security Best Practices

1. **Regular Review**: Review audit logs regularly for suspicious activity
2. **Access Monitoring**: Monitor who accesses audit logs
3. **Retention Compliance**: Ensure retention policy meets compliance requirements
4. **Export Security**: Securely handle exported audit data
5. **Alerting**: Set up alerts for unusual patterns in audit logs

## Example Usage

### Search for User Activity

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "https://api.example.com/api/v1/audit/logs?user_id=123&start_date=2024-01-01"
```

### Export Logs for Compliance

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "https://api.example.com/api/v1/audit/export?start_date=2024-01-01&end_date=2024-01-31" \
  -o audit_export.json
```

### Get Statistics

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "https://api.example.com/api/v1/audit/stats"
```

## Troubleshooting

### Missing Audit Logs

1. Check if the operation type is logged (only sensitive operations)
2. Verify middleware is properly configured
3. Check database connection for audit logger

### Sensitive Data in Logs

1. Ensure sensitive field names match the redaction list
2. Check custom parameter sanitization logic
3. Verify request body parsing is working

### Performance Impact

1. Audit logging adds minimal overhead (<5ms per request)
2. Database writes are asynchronous
3. Failed audit logging does not block requests

## Compliance

This audit trail system helps meet compliance requirements for:

- **SOC 2**: Access logging and monitoring
- **PCI DSS**: Access control and audit trails
- **GDPR**: Data access logging
- **HIPAA**: Audit controls for PHI access
- **ISO 27001**: Access logging and review

## Future Enhancements

- Real-time alerting on suspicious patterns
- Machine learning anomaly detection on audit logs
- Immutable log storage (WORM)
- Blockchain-based audit trail verification
- Advanced search and analytics

## Pipeline Audit Logging (Issue #757)

Beyond the API-level audit trail above, AstroML records an **immutable
who/what/when/result trail for critical pipeline operations**: model
activations, configuration changes, and rollbacks.

### Operations recorded

| Operation | Recorded when | Key fields |
| --- | --- | --- |
| `model_activated` | A model version is activated in the pipeline | `version`, `activated_by` |
| `model_deactivated` | A model version is taken out of service | `version` |
| `model_rolled_back` | A rollback to a previous model version | `from_version`, `to_version`, `reason` |
| `config_changed` | A pipeline/model configuration file is edited | per-key `before`/`after` changes |
| `config_rolled_back` | A configuration rollback | `from_version`, `to_version` |
| `feature_config_changed` | Feature-builder YAML definitions change (#742) | changed builders, definition hashes |

Each record contains:

- **Who**: the `actor` (user ID or service identity)
- **What**: the `operation`, `target` (model/config path), and `details`
- **When**: a UTC ISO-8601 `timestamp`
- **Result**: an `outcome` (`success` / `failure`)

### Tamper evidence

Records are **hash-chained**: every entry stores the hash of its predecessor
plus a content hash over its own fields. `PipelineAuditLogger.verify_chain()`
detects any silent edit, deletion, or reordering of history. Records themselves
are frozen dataclasses and are persisted through the append-only
`AuditStore` backends (NDJSON files in production).

### Usage

```python
from astroml.tracking.pipeline_audit import PipelineAuditLogger
from astroml.governance.audit_logger import FileAuditStore

audit = PipelineAuditLogger(
    store=FileAuditStore("audit_logs/pipeline"),
    actor="pipeline-service",
)

audit.log_model_activation("fraud-detector", "v3", actor="alice")
audit.log_config_change(
    "configs/model/thresholds.yaml",
    {"false_positive_weight": {"before": 0.2, "after": 0.5}},
    actor="bob",
)
audit.log_rollback("fraud-detector", "v3", "v2", actor="carol", reason="precision regression")

assert audit.verify_chain()  # raises no error; returns False if history was edited
```

### Querying and verification

```python
from astroml.tracking.pipeline_audit import PipelineOperation

# Newest-first records, filterable by operation / target / actor
records = audit.query(operation=PipelineOperation.CONFIG_CHANGED, limit=50)

# Verify the tamper-evidence chain across stored history
ok = audit.verify_chain()
```

### Storage and retention

Pipeline audit records are written to the same append-only NDJSON stores as
API audit logs (90-day retention applies) and share the sensitive-field
redaction list documented above.
