# API Rate and Size Limits

The v1 API enforces conservative request body size and pagination limits to protect service stability and prevent accidental or malicious overload.

---

## Request body size limit

| Setting | Default | Environment variable |
|---------|---------|----------------------|
| Maximum body size | 10 485 760 bytes (10 MB) | `MAX_REQUEST_BODY_SIZE` |

Every inbound request body is checked before the handler runs.  If the body exceeds the limit the API returns HTTP **413** immediately without processing the payload.

The check uses both the `Content-Length` header (fast path) and actual streaming byte count (slow path), so it applies even when no `Content-Length` is sent.

### Error shape (HTTP 413)

```json
{
  "type": "https://astroml.dev/problems/REQUEST_TOO_LARGE",
  "title": "Request entity too large",
  "status": 413,
  "code": "REQUEST_TOO_LARGE",
  "detail": "Request body exceeds the maximum allowed size.",
  "details": []
}
```

---

## Pagination limits

| Setting | Default | Environment variable |
|---------|---------|----------------------|
| Maximum page size | 100 | `MAX_PAGE_SIZE` |
| Default page size | 20 | `DEFAULT_PAGE_SIZE` |

Endpoints that accept a `limit` or `page_size` query parameter enforce these bounds.  Requesting more items than `MAX_PAGE_SIZE` returns HTTP **400** — the API rejects out-of-range requests rather than silently clamping them, so callers know their request was out of range.

### Error shape (HTTP 400)

```json
{
  "type": "https://astroml.dev/problems/PAGE_SIZE_TOO_LARGE",
  "title": "Requested page size 200 exceeds the maximum allowed page size of 100.",
  "status": 400,
  "code": "PAGE_SIZE_TOO_LARGE",
  "detail": "Requested page size 200 exceeds the maximum allowed page size of 100.",
  "details": []
}
```

---

## Adjusting limits for local development

Add any of the following to your `.env` file (or export them as environment variables before starting the server):

```dotenv
# Raise the body limit to 50 MB locally
MAX_REQUEST_BODY_SIZE=52428800

# Allow larger pages in development
MAX_PAGE_SIZE=500
DEFAULT_PAGE_SIZE=50
```

The server reads these values at startup via `pydantic-settings`; a restart is required after changing them.

---

## Full RFC 7807 error shapes

### 413 — Request entity too large

```json
{
  "type": "https://astroml.dev/problems/REQUEST_TOO_LARGE",
  "title": "Request entity too large",
  "status": 413,
  "code": "REQUEST_TOO_LARGE",
  "detail": "Request body exceeds the maximum allowed size.",
  "details": []
}
```

Returned with `Content-Type: application/problem+json`.

### 400 — Page size too large

```json
{
  "type": "https://astroml.dev/problems/PAGE_SIZE_TOO_LARGE",
  "title": "Requested page size {N} exceeds the maximum allowed page size of {M}.",
  "status": 400,
  "code": "PAGE_SIZE_TOO_LARGE",
  "detail": "Requested page size {N} exceeds the maximum allowed page size of {M}.",
  "details": []
}
```

Returned with `Content-Type: application/problem+json`.
