# CORS Configuration

AstroML's v1 API uses FastAPI's `CORSMiddleware`. This document describes
the environment variables that control allowed origins, the safe defaults,
and how to configure CORS for local development versus production.

## Environment variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `CORS_ORIGINS` | Comma-separated URLs | _(empty — deny all)_ | Allowed cross-origin hosts. Set to your frontend URL(s) in production. |
| `CORS_CREDENTIALS` | `true` / `false` | `false` | Whether to include credentials (cookies, `Authorization` header) in cross-origin requests. Set to `true` to opt in to credential sharing with the listed origins. |

## Safe defaults

The default `CORS_ORIGINS=` (empty) denies all cross-origin requests. This
is the correct production posture when the frontend is served from the same
origin as the API, or when the allowed origins are not yet known at build time.

### Why `*` + credentials is disallowed

Browsers reject `Access-Control-Allow-Origin: *` responses to credentialed
requests (cookies or `Authorization` header). The FastAPI/Starlette
`CORSMiddleware` does not send `*` in that case; instead it echoes the
request `Origin`. This means setting `allow_origins=["*"]` with
`allow_credentials=True` silently degrades to per-origin allow-listing
of _every_ origin that asks — equivalent to removing CORS protection
entirely. The config validator rejects this combination at startup.

## Local development

```dotenv
# .env (development)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
CORS_CREDENTIALS=true
```

This allows the Vite dev server (`5173`) and Create React App (`3000`) to
make credentialed requests to the local API.

## Production

```dotenv
# .env.production
CORS_ORIGINS=https://app.yourdomain.com
CORS_CREDENTIALS=true
```

List only the origins you control. Never use `*` in production when
credentials are involved.

## Allowed methods and headers

The API accepts: `GET POST PUT PATCH DELETE OPTIONS`

The API accepts these request headers: `Authorization`, `Content-Type`, `X-Request-ID`

Expand `allow_headers` in `api/app.py` only if a new endpoint requires
additional headers.
