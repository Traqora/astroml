# API Errors and Operator Actions

## Purpose

This document is the reference for error messages returned by the v1 API and the actions an operator should take. It complements the symptom focused [Troubleshooting Guide](TROUBLESHOOTING.md). Error responses are produced by `api/middleware/errors.py` and conform to RFC 7807.

## Response shape

Every error is served as `application/problem+json` with these fields:

| Field | Meaning |
| --- | --- |
| `type` | Stable problem URI, `https://astroml.dev/problems/<code>` |
| `title` | Short human readable category |
| `status` | HTTP status code |
| `code` | Stable machine readable identifier used in this document |
| `detail` | Occurrence specific message, omitted when not available |
| `instance` | Request path that produced the error |
| `details` | Structured validation or context data, omitted when not available |

`PROBLEM_TYPE_BASE` in `api/schemas/errors.py` is the source of truth for the problem URI prefix. Clients should branch on `code`, not on `title` or `detail`.

## Common codes

| Code | Status | Typical message | Operator action |
| --- | --- | --- | --- |
| `HTTP_400` | 400 | `Invalid request` or a route specific validation message | Check the request body and query parameters against the endpoint contract |
| `HTTP_401` | 401 | `Authentication required` | Confirm the bearer header is present and that `AUTH_ENABLED=true` is intentional |
| `HTTP_401` | 401 | `Invalid or expired token` | Confirm the credential matches an active user or API key, then rotate or reissue it |
| `HTTP_403` | 403 | `Admin privileges required` | Use a principal with the required scope or admin role |
| `HTTP_403` | 403 | `Missing required scopes: ...` | Issue a new API key containing the listed scopes |
| `HTTP_404` | 404 | `Model not found` or `Account not found` | Verify the identifier and confirm the record exists in the target environment |
| `HTTP_409` | 409 | Conflict with current resource state | Re-read the resource and retry only if the new state is acceptable |
| `HTTP_429` | 429 | `Rate limit exceeded` | Back off, respect `Retry-After`, and review the per principal quota |
| `HTTP_429` | 429 | `Too many failed authentication attempts. Please try again later.` | Stop retrying the same credential and investigate possible credential stuffing |
| `HTTP_503` | 503 | `Scoring service temporarily unavailable` | Check model artifact availability and scorer logs before retrying |
| `VALIDATION_ERROR` | 422 | `Request validation failed` | Read the `details.loc` and `details.message` entries and correct the offending fields |
| `INTERNAL_SERVER_ERROR` | 500 | `Internal server error` | Correlate the request with service logs, do not return stack traces to clients |

Anything not caused by client input should be treated as an operational issue rather than retried blindly.

## Authentication and rate limit messages

The auth middleware emits a small fixed set of messages. Their full meaning and client handling are documented in [API Authentication for Deployed Inference](API_AUTHENTICATION.md).

- `Authentication required`: the `Authorization` header is absent or does not use the `Bearer` scheme.
- `Invalid or expired token`: the JWT or API key could not be resolved to an active principal.
- `Rate limit exceeded`: the per principal request budget is exhausted.
- `Too many failed authentication attempts. Please try again later.`: the brute force lockout is active for the client and token fingerprint.

Both 429 responses include `Retry-After` and `X-RateLimit-*` headers.

## Operator actions by symptom

1. A rising 401 rate usually means a stale client credential or a wrong environment. Rotate the key and update the deployment.
2. A rising 429 rate with valid credentials means real load. Increase the documented quota deliberately or scale the service.
3. A rising 429 rate with invalid credentials means abuse. Keep the lockout enabled and block the source if necessary.
4. A sustained 503 from inference means the model artifact or scorer failed to load. Check the model registry and restart after the artifact is restored.
5. A 500 with no client side cause should be investigated with the correlation id. The API intentionally omits exception details from the response.

## Logging safety

Error details must never contain credentials. `sanitize_log_value` redacts bearer tokens, API keys, secret assignments and URL credentials before a log record is emitted. When adding a new error path, log exception messages through the standard logger and rely on the shared redaction filter rather than formatting secrets manually.

Client facing responses use stable messages only. Internal exception text is never returned by `unhandled_exception_handler`.

## Related documentation

- [API Authentication for Deployed Inference](API_AUTHENTICATION.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Security Policy](../SECURITY.md)
- [Health Checks](HEALTH_CHECKS.md)
- Source of truth: `api/middleware/errors.py`, `api/schemas/errors.py`, `api/auth/middleware.py`

## Verification

`tests/docs/test_api_errors_doc.py` asserts that every stable code emitted by the error layer is documented and that both new documents are linked from the documentation index and mkdocs navigation.
