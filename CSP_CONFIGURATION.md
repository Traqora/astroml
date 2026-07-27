# Content Security Policy (CSP) Configuration

This document describes the Content Security Policy (CSP) implementation in the AstroML API to prevent XSS attacks.

## Overview

Content Security Policy (CSP) is an added layer of security that helps to detect and mitigate certain types of attacks, including Cross-Site Scripting (XSS) and data injection attacks. The AstroML API implements CSP middleware to automatically add security headers to all HTTP responses.

## Implementation

### CSP Middleware

The CSP middleware is located in `api/middleware/csp.py` and implements:

- **Restrictive default policy**: Only allows resources from the same origin
- **Nonce-based script execution**: Generates cryptographically secure nonces for inline scripts
- **Report-only mode**: Supports development/testing without blocking resources
- **Additional security headers**: Adds X-Content-Type-Options, X-Frame-Options, etc.

### Current Policy

The CSP policy applied to all responses:

```
default-src 'self';
script-src 'self' 'nonce-{nonce}';
style-src 'self' 'unsafe-inline';
img-src 'self' data: https:;
connect-src 'self';
font-src 'self';
object-src 'none';
base-uri 'self';
form-action 'self';
frame-ancestors 'none';
```

### Policy Directives Explained

| Directive | Value | Purpose |
|-----------|-------|---------|
| `default-src` | `'self'` | Default policy for all resource types - only allow from same origin |
| `script-src` | `'self' 'nonce-{nonce}'` | Only allow scripts from same origin or with valid nonce |
| `style-src` | `'self' 'unsafe-inline'` | Allow inline styles for development (can be tightened in production) |
| `img-src` | `'self' data: https:` | Allow images from same origin, data URIs, and HTTPS sources |
| `connect-src` | `'self'` | Only allow fetch/XHR to same origin |
| `font-src` | `'self'` | Only allow fonts from same origin |
| `object-src` | `'none'` | Block plugins (Flash, etc.) |
| `base-uri` | `'self'` | Restrict base tag to same origin |
| `form-action` | `'self'` | Restrict form submissions to same origin |
| `frame-ancestors` | `'none'` | Prevent clickjacking by blocking embedding |

## Configuration

### Environment Variables

Configure CSP behavior via environment variables in `.env`:

```bash
# Content Security Policy
CSP_REPORT_ONLY=true              # Use report-only mode (development)
CSP_REPORT_URI=https://example.com/csp-report  # URI for violation reports
CSP_ENABLE_NONCE=true             # Generate nonce for script-src
```

### Settings

CSP configuration is managed in `api/config.py`:

```python
class Settings(BaseSettings):
    # Content Security Policy
    csp_report_only: bool = True  # Use report-only mode in development
    csp_report_uri: str | None = None  # URI to send CSP violation reports
    csp_enable_nonce: bool = True  # Generate nonce for script-src
```

### CORS Configuration

CORS is configured to work alongside CSP:

```python
# In api/config.py
cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

# In production, set CORS_ORIGINS environment variable:
# CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
```

The CORS middleware now restricts methods to only those needed:
- `GET`, `POST`, `PUT`, `PATCH`, `DELETE`, `OPTIONS`

## Usage

### Development Mode

In development, CSP runs in report-only mode by default:

```bash
ENV=development make run-api
```

This allows you to see CSP violations in the browser console without blocking resources.

### Production Mode

In production, set `CSP_REPORT_ONLY=false` to enforce the policy:

```bash
CSP_REPORT_ONLY=false make run-api
```

### Using Nonces in Frontend

When using inline scripts in your frontend, use the nonce provided in the `X-CSP-Nonce` header:

```javascript
// Get nonce from response headers
const nonce = response.headers.get('X-CSP-Nonce');

// Use nonce in script tag
const script = document.createElement('script');
script.nonce = nonce;
script.src = '/path/to/script.js';
document.head.appendChild(script);
```

### Using Script Hashes

For static scripts, you can use SHA-256 hashes instead of nonces:

```python
from api.middleware.csp import CSPMiddleware

script_content = "console.log('test');"
hash_value = CSPMiddleware.hash_script(script_content)
# Add to CSP: script-src 'sha256-{hash_value}'
```

## Testing

### Running CSP Tests

Run the CSP middleware tests:

```bash
pytest api/tests/test_csp.py -v
```

### Manual Testing with Browser Dev Tools

1. Open the API in your browser
2. Open Developer Tools (F12)
3. Go to the Console tab
4. Look for CSP violation reports
5. Go to the Network tab
6. Check response headers for `Content-Security-Policy`

### Testing CSP Violations

To test CSP violations:

1. Try to load a script from an external domain
2. Try to use inline JavaScript without nonce
3. Check browser console for violation messages

## CSP Violation Reports

### Setting Up Report URI

To collect CSP violation reports:

1. Set `CSP_REPORT_URI` to your report collection endpoint
2. Implement an endpoint to receive POST requests with violation reports
3. Store and analyze reports to identify issues

### Report Format

CSP violation reports are JSON documents sent to your report URI:

```json
{
  "csp-report": {
    "document-uri": "http://example.com/",
    "referrer": "http://example.com/",
    "violated-directive": "script-src",
    "effective-directive": "script-src",
    "original-policy": "...",
    "disposition": "report",
    "blocked-uri": "https://evil.com/script.js",
    "line-number": 10,
    "column-number": 5,
    "source-file": "http://example.com/page.html"
  }
}
```

## Troubleshooting

### Common Issues

#### Inline Scripts Blocked

**Problem**: Inline scripts are being blocked by CSP.

**Solution**: Use the nonce from `X-CSP-Nonce` header in your script tags, or move scripts to external files.

#### External Resources Blocked

**Problem**: External CSS/JS/fonts are being blocked.

**Solution**: Add the domain to the appropriate CSP directive in `api/middleware/csp.py`. For example:

```python
policy_parts = [
    "default-src 'self'",
    "script-src 'self' 'nonce-{nonce}' https://cdn.example.com",
    # ...
]
```

#### Development Difficulties

**Problem**: CSP is too restrictive during development.

**Solution**: Use report-only mode (`CSP_REPORT_ONLY=true`) to see violations without blocking.

## Security Best Practices

1. **Always use CSP in production**: Enable enforcement mode in production environments
2. **Use nonces for dynamic scripts**: Generate nonces for any inline scripts
3. **Minimize inline scripts**: Move JavaScript to external files when possible
4. **Monitor violation reports**: Set up report URI and regularly review violations
5. **Keep policies minimal**: Only add directives you actually need
6. **Test thoroughly**: Test CSP in report-only mode before enforcing
7. **Review CORS origins**: Ensure CORS origins match your CSP policy

## Additional Security Headers

The CSP middleware also adds these security headers:

| Header | Value | Purpose |
|--------|-------|---------|
| `X-Content-Type-Options` | `nosniff` | Prevent MIME type sniffing |
| `X-Frame-Options` | `DENY` | Prevent clickjacking |
| `X-XSS-Protection` | `1; mode=block` | Enable XSS filtering |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Control referrer information |
| `Permissions-Policy` | `geolocation=(), microphone=(), camera=()` | Restrict sensitive features |

## References

- [MDN Web Docs - CSP](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)
- [OWASP - Content Security Policy](https://owasp.org/www-community/attacks/Content_Security_Policy)
- [CSP Evaluator](https://csp-evaluator.withgoogle.com/)
- [CSP Report Viewer](https://report-uri.com/home/tools)

## Support

For questions or issues with CSP configuration:
- Open an issue with the `security` label
- Check browser console for violation details
- Review CSP violation reports if report URI is configured
