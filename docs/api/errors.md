# API Error Responses

AstroML's HTTP API returns errors in [RFC 7807](https://www.rfc-editor.org/rfc/rfc7807)
`application/problem+json` format.

## Top-level shape

Every error response body has these members:

| Member     | Type    | Description |
|------------|---------|-------------|
| `type`     | string  | A URI identifying the problem type (`about:blank` when no specific type is defined). |
| `title`    | string  | Short human-readable summary of the problem. |
| `status`   | integer | HTTP status code. |
| `detail`   | string  | Human-readable explanation of this specific occurrence. |
| `instance` | string  | The request path that produced the error. |
| `code`     | string  | Machine-readable error code (e.g. `"VALIDATION_ERROR"`, `"HTTP_404"`). |
| `details`  | array   | Present on 422 responses; one item per field error (see below). |

### Example — generic 404

```json
{
  "type": "about:blank",
  "title": "Account 'GXYZ' not found",
  "status": 404,
  "detail": "Account 'GXYZ' not found",
  "instance": "/api/v1/accounts/GXYZ",
  "code": "HTTP_404"
}
```

## Validation error shape (HTTP 422)

When request input fails Pydantic validation the response has `status: 422`,
`code: "VALIDATION_ERROR"`, and a `details` array.  Each item in `details`
describes one field failure:

| Key       | Type   | Description |
|-----------|--------|-------------|
| `field`   | string | Dot-separated path to the failing field (sentinels stripped — see below). |
| `message` | string | Human-readable explanation of why the value was rejected. |
| `code`    | string | Pydantic error type string (e.g. `"missing"`, `"string_too_short"`). |

### Example — missing required field

Request: `POST /api/v1/errors/report` with body `{}`

```json
{
  "type": "about:blank",
  "title": "Request validation failed",
  "status": 422,
  "detail": "Request validation failed",
  "instance": "/api/v1/errors/report",
  "code": "VALIDATION_ERROR",
  "details": [
    {
      "field": "message",
      "message": "Field required",
      "code": "missing"
    }
  ]
}
```

### `field` construction rules

Pydantic reports errors with a `loc` tuple such as `("body", "address", "zip")`.
The API strips the leading source-sentinel token (`"body"`, `"query"`, `"path"`,
or `"header"`) and joins the remaining segments with `"."`.

| Pydantic `loc`                    | `field` value  |
|-----------------------------------|----------------|
| `("body", "email")`               | `"email"`      |
| `("body", "address", "zip")`      | `"address.zip"` |
| `("body", "items", 0)`            | `"items.0"`    |
| `("query", "page")`               | `"page"`       |
| `("body",)` (root-level error)    | `"__root__"`   |

Integer segments (list indices) are converted to strings.
If no segments remain after stripping, `"__root__"` is used.

### `code` values

The `code` field on each `details` item comes directly from Pydantic's error
`type` string.  Common values include:

| Code                | Meaning |
|---------------------|---------|
| `missing`           | Required field absent from the request. |
| `string_too_short`  | String value is shorter than the minimum length. |
| `string_too_long`   | String value exceeds the maximum length. |
| `int_parsing`       | Value could not be parsed as an integer. |
| `value_error`       | Value failed a custom validator. |

For the full list see the [Pydantic error reference](https://docs.pydantic.dev/latest/errors/validation_errors/).
