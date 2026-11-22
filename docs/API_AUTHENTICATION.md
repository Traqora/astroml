# API Authentication for Deployed Inference

## Purpose

This document defines the authentication required to call the deployed v1 inference API. The inference endpoint is `POST /api/v1/fraud/score`, which scores up to 50 accounts for anomaly and fraud risk. Authentication is enforced globally by `AuthMiddleware` in `api/auth/middleware.py`, so the same rules apply to every `/api/v1/*` route unless the path is explicitly public.

This is the supported boundary for v1 deployments. No optional service is required beyond the PostgreSQL database the API already uses.

## Supported credentials

Send credentials on every request with the standard HTTP header:

```text
Authorization: Bearer <token>
```

Two credential types are accepted after the `Bearer ` prefix:

| Type | How it is obtained | Token shape | Notes |
| --- | --- | --- | --- |
| JWT | `POST /api/v1/auth/login` with a username and password | Compact JWT (`eyJ...`) | Short lived, expires according to `ACCESS_TOKEN_EXPIRE_HOURS` |
| API key | `POST /api/v1/auth/api-keys` with an authenticated session | Starts with `ak_` | Intended for machine to machine inference clients |

An API key can be issued with a scoped permission set. Rotate and revoke keys using the procedure in [API Key Rotation and Revocation](API_KEY_ROTATION.md).

The inference route itself does not declare an additional scope requirement. Any successfully authenticated principal may call `POST /api/v1/fraud/score`. Other administrative routes may require scopes and return 403 when a scope is missing.

## Public paths

The following paths are deliberately unauthenticated so health checks and documentation work before credentials are available:

- `/health`
- `/docs`
- `/openapi.json`
- `/redoc`
- `/api/v1/auth/login`

`POST /api/v1/fraud/score` is not in this list. A deployed inference request without a bearer token is rejected.

## Deployment boundary

`AUTH_ENABLED` controls enforcement and is read at call time from the environment.

- Production and shared deployments must set `AUTH_ENABLED=true`. This is the default.
- When `AUTH_ENABLED=false`, the middleware treats every caller as an anonymous principal with all scopes. This mode exists for local development and must never be used for a deployed inference endpoint.
- Production startups also require `JWT_SECRET_KEY` or `SECRET_KEY` to be set to a non default value. The application fails fast at import time otherwise.

Setting `AUTH_ENABLED=true` does not add an external dependency. Validation uses the same database session factory as the rest of the API.

## Failure responses

Failures use the RFC 7807 problem details format described in [API Errors and Operator Actions](API_ERRORS.md).

| Status | Message | Cause | Required client action |
| --- | --- | --- | --- |
| 401 | `Authentication required` | Missing `Authorization` header | Add a bearer token |
| 401 | `Invalid or expired token` | Unknown, malformed, revoked or expired credential | Obtain a fresh token or rotate the API key |
| 403 | `Missing required scopes: ...` | Authenticated principal lacks a scope required by the route | Request a key with the needed scope |
| 429 | `Rate limit exceeded` | Per principal request limit reached | Back off and respect the `Retry-After` header |
| 429 | `Too many failed authentication attempts. Please try again later.` | Brute force lockout after repeated bad tokens | Stop retrying with the same credential and check the token |

Clients should treat 401 and 403 as permanent for the current credential and 429 as retryable after the advertised delay.

## Example inference request

```bash
curl -X POST https://api.example.com/api/v1/fraud/score \
  -H "Authorization: Bearer $ASTROML_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"accounts": ["GABC..."], "edges": []}'
```

Keep credentials in environment variables or a secret manager. Do not place tokens in query strings, commit them to local environment files, or include them in log messages and exception reports.

## Operator checklist

- Confirm `AUTH_ENABLED=true` on the deployed service.
- Issue API keys with the minimum scopes needed by the client.
- Verify that the public paths above are the only unauthenticated routes.
- Monitor 401 and 429 rates for credential or abuse issues.
- Rotate keys before their expiration date using [API Key Rotation and Revocation](API_KEY_ROTATION.md).

## Related documentation

- [API Errors and Operator Actions](API_ERRORS.md)
- [API Key Rotation and Revocation](API_KEY_ROTATION.md)
- [Security Policy](../SECURITY.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- Source of truth: `api/auth/config.py`, `api/auth/middleware.py`, `api/auth/dependencies.py`

## Verification

`tests/docs/test_api_authentication_doc.py` asserts that this document covers the credential types, public paths, deployment boundary and failure responses defined in the source modules.
