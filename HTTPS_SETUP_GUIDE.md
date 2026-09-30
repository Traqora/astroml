# HTTPS Enforcement Setup Guide

This document describes how to configure HTTPS enforcement for the AstroML API to prevent downgrade attacks and ensure secure connections.

## Overview

The AstroML API includes middleware to enforce HTTPS connections:

- **HTTPS Redirect Middleware**: Redirects HTTP requests to HTTPS
- **HSTS Middleware**: Adds HTTP Strict Transport Security headers
- **Proxy SSL Header Configuration**: Support for load balancers and reverse proxies

## Configuration

### Environment Variables

Configure HTTPS behavior via environment variables in `.env`:

```bash
# HTTPS Enforcement
HTTPS_ENABLED=true  # Enable HTTPS redirects (production only)
HTTPS_ALLOWED_HOSTS=example.com,api.example.com  # Allowed hostnames
HSTS_ENABLED=true  # Enable HSTS headers (production only)
HSTS_MAX_AGE=31536000  # HSTS max-age in seconds (1 year)
HSTS_INCLUDE_SUBDOMAINS=true  # Apply HSTS to subdomains
HSTS_PRELOAD=false  # Include in browser preload list
SECURE_PROXY_SSL_HEADER=X-Forwarded-Proto,https  # For load balancers
```

### Settings

HTTPS configuration is managed in `api/config.py`:

```python
class Settings(BaseSettings):
    # HTTPS Enforcement
    https_enabled: bool = False  # Enable HTTPS redirects (production only)
    https_allowed_hosts: list[str] = []  # Allowed hostnames for HTTPS
    hsts_enabled: bool = False  # Enable HSTS headers (production only)
    hsts_max_age: int = 31536000  # HSTS max-age in seconds (1 year)
    hsts_include_subdomains: bool = True  # Apply HSTS to subdomains
    hsts_preload: bool = False  # Include in browser preload list
    secure_proxy_ssl_header: tuple[str, str] | None = None  # For load balancers
```

## Production Setup

### Option 1: Direct HTTPS with Uvicorn

For simple deployments where Uvicorn handles HTTPS directly:

```bash
# Generate SSL certificate (using Let's Encrypt for example)
certbot certonly --standalone -d api.example.com

# Run with SSL
uvicorn api.app:app --host 0.0.0.0 --port 8443 \
    --ssl-keyfile /etc/letsencrypt/live/api.example.com/privkey.pem \
    --ssl-certfile /etc/letsencrypt/live/api.example.com/fullchain.pem
```

Environment variables:
```bash
HTTPS_ENABLED=true
HTTPS_ALLOWED_HOSTS=api.example.com
HSTS_ENABLED=true
```

### Option 2: Nginx Reverse Proxy

For production deployments with Nginx as a reverse proxy:

**Nginx Configuration:**
```nginx
server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Environment variables:
```bash
HTTPS_ENABLED=false  # Nginx handles HTTPS
HSTS_ENABLED=true
SECURE_PROXY_SSL_HEADER=X-Forwarded-Proto,https
```

### Option 3: Docker with Nginx

Update `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: api/Dockerfile
    environment:
      - HTTPS_ENABLED=false
      - HSTS_ENABLED=true
      - SECURE_PROXY_SSL_HEADER=X-Forwarded-Proto,https
    ports:
      - "8000:8000"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - api
```

### Option 4: Kubernetes with Ingress

For Kubernetes deployments:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: astroml-api-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: astroml-api-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: astroml-api
            port:
              number: 8000
```

Environment variables in your deployment:
```yaml
env:
  - name: HTTPS_ENABLED
    value: "false"
  - name: HSTS_ENABLED
    value: "true"
  - name: SECURE_PROXY_SSL_HEADER
    value: "X-Forwarded-Proto,https"
```

## HSTS Configuration

### HSTS Header Format

The HSTS header is formatted as:
```
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
```

### HSTS Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_age` | 31536000 (1 year) | Time in seconds browsers should remember HSTS |
| `include_subdomains` | true | Apply HSTS to all subdomains |
| `preload` | false | Include in browser HSTS preload list |

### HSTS Preload

To add your domain to the HSTS preload list:

1. Set `HSTS_PRELOAD=true` in your environment
2. Ensure your HSTS header includes: `max-age=31536000; includeSubDomains; preload`
3. Submit your domain to [hstspreload.org](https://hstspreload.org/)
4. Wait for inclusion (can take several weeks)

**Warning**: Once preloaded, it's difficult to remove. Only enable if you're committed to HTTPS.

## Testing HTTPS

### Local Testing with Self-Signed Certificates

Generate self-signed certificates for local testing:

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

Run with self-signed certificates:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8443 \
    --ssl-keyfile key.pem \
    --ssl-certfile cert.pem
```

### Testing HTTPS Redirect

```bash
# Test HTTP redirect
curl -I http://api.example.com
# Should return 301 with Location: https://api.example.com

# Test HTTPS access
curl -I https://api.example.com
# Should return 200 with HSTS header
```

### Testing HSTS Header

```bash
curl -I https://api.example.com
# Check for: Strict-Transport-Security: max-age=31536000; includeSubDomains
```

### Running Tests

Run the HTTPS middleware tests:

```bash
pytest api/tests/test_https.py -v
```

## Security Best Practices

### 1. Always Use HTTPS in Production

Never disable HTTPS enforcement in production. Set:
```bash
HTTPS_ENABLED=true
HSTS_ENABLED=true
```

### 2. Use Strong SSL/TLS Configuration

Ensure your reverse proxy uses:
- TLS 1.2 or higher
- Strong cipher suites
- Valid certificates from trusted CAs

### 3. Implement Certificate Rotation

Regularly rotate SSL certificates:
- Use Let's Encrypt for automatic renewal
- Set up monitoring for certificate expiration
- Test certificate renewal process

### 4. Monitor HSTS Compliance

- Monitor for HSTS violations
- Check browser console for warnings
- Review access logs for HTTP traffic

### 5. Use Secure Headers

The HTTPS middleware works alongside other security headers:
- CSP (Content Security Policy)
- X-Frame-Options
- X-Content-Type-Options
- X-XSS-Protection

## Troubleshooting

### HTTPS Redirect Loop

**Problem**: Infinite redirect loop between HTTP and HTTPS.

**Solution**: 
- Check if your reverse proxy is already handling HTTPS
- Set `HTTPS_ENABLED=false` if using a reverse proxy
- Configure `SECURE_PROXY_SSL_HEADER` correctly

### HSTS Not Applied

**Problem**: HSTS header not appearing in responses.

**Solution**:
- Ensure `HSTS_ENABLED=true`
- Verify request is using HTTPS (HSTS only applies to HTTPS)
- Check middleware order (HSTS should be after HTTPS redirect)

### Mixed Content Errors

**Problem**: Browser blocks mixed content (HTTP resources on HTTPS page).

**Solution**:
- Update all resource URLs to use HTTPS
- Ensure CSP allows HTTPS sources
- Check for hardcoded HTTP URLs in frontend code

### Certificate Errors

**Problem**: Browser shows certificate warnings.

**Solution**:
- Use certificates from trusted CAs in production
- Ensure certificate covers all required domains
- Check certificate chain is complete
- Verify certificate hasn't expired

## Load Balancer Configuration

When using a load balancer (AWS ELB, GCP Load Balancer, etc.):

### AWS ELB

```bash
SECURE_PROXY_SSL_HEADER=X-Forwarded-Proto,https
HTTPS_ENABLED=false  # ELB handles HTTPS
HSTS_ENABLED=true
```

### GCP Load Balancer

```bash
SECURE_PROXY_SSL_HEADER=X-Forwarded-Proto,https
HTTPS_ENABLED=false  # Load balancer handles HTTPS
HSTS_ENABLED=true
```

### Cloudflare

```bash
SECURE_PROXY_SSL_HEADER=CF-Visitor,https
HTTPS_ENABLED=false  # Cloudflare handles HTTPS
HSTS_ENABLED=true
```

## Docker Configuration

The API Dockerfile has been updated to support HTTPS:

```dockerfile
# Expose ports
EXPOSE 8000 8443

# Run the application
# For production with HTTPS, set environment variables:
# HTTPS_ENABLED=true
# HTTPS_ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com
# HSTS_ENABLED=true
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run with HTTPS:
```bash
docker build -f api/Dockerfile -t astroml-api .
docker run -p 8443:8443 \
  -e HTTPS_ENABLED=true \
  -e HTTPS_ALLOWED_HOSTS=api.example.com \
  -e HSTS_ENABLED=true \
  -v /path/to/certs:/certs \
  astroml-api \
  uvicorn api.app:app --host 0.0.0.0 --port 8443 \
    --ssl-keyfile /certs/key.pem \
    --ssl-certfile /certs/cert.pem
```

## References

- [OWASP - Transport Layer Protection](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [MDN - HTTP Strict Transport Security](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Strict-Transport-Security)
- [HSTS Preload](https://hstspreload.org/)
- [Let's Encrypt](https://letsencrypt.org/)

## Support

For questions about HTTPS configuration:
- Open an issue with the `security` label
- Check browser console for HTTPS/HSTS errors
- Review Nginx/load balancer logs for SSL errors
