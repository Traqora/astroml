# Post-Deployment Smoke Tests

## Overview

Smoke tests verify that a deployment succeeded by checking critical system components (issue #560). These tests run automatically after deployment and can be manually triggered for re-runs.

## Test Coverage

### Health Check
- **Endpoint**: `GET /healthz`
- **Purpose**: Verify the application is running
- **Expected**: HTTP 200 with status "healthy"

### API Root
- **Endpoint**: `GET /`
- **Purpose**: Verify API is responding with app info
- **Expected**: HTTP 200 with app name and version

### Database Connectivity
- **Endpoint**: `GET /healthz/db`
- **Purpose**: Verify database connection is healthy
- **Expected**: HTTP 200 with status "healthy"

### Cache Connectivity
- **Endpoint**: `GET /healthz/cache`
- **Purpose**: Verify cache connection is healthy
- **Expected**: HTTP 200 with status "healthy"

### API Endpoints Responsive
- **Endpoints**: `/api/v1/healthz`, `/api/v1/models`
- **Purpose**: Verify key API endpoints are accessible
- **Expected**: HTTP 200 or 401 (auth required)

## Running Smoke Tests

### Automated (After Deployment)

Smoke tests run automatically in the CI/CD pipeline after deployment with a 5-minute timeout. If tests fail, the deployment is automatically rolled back.

### Manual Trigger

Use the GitHub Actions workflow to manually trigger smoke tests:

1. Go to Actions tab in GitHub
2. Select "Smoke Tests (Manual Trigger)"
3. Choose environment (production/staging)
4. Optionally provide custom base URL
5. Click "Run workflow"

### Command Line

Run smoke tests directly from the command line:

```bash
# Default URL (localhost:8000)
python tests/e2e/test_deployment_smoke.py

# Custom URL
python tests/e2e/test_deployment_smoke.py https://api.example.com

# Using pytest
pytest tests/e2e/test_deployment_smoke.py -v
```

### Programmatic

```python
from tests.e2e.test_deployment_smoke import run_smoke_tests

success = run_smoke_tests("https://api.example.com")
if success:
    print("All smoke tests passed!")
```

## Timeout

Smoke tests have a 5-minute timeout. If tests exceed this duration, they are marked as failed and the deployment is rolled back.

## Rollback Behavior

If smoke tests fail during automated deployment:
- Kubernetes deployments are rolled back to previous version
- Notification is sent to Slack
- Deployment is marked as failed in GitHub Actions

## Test Results

### Success
- All checks pass
- Deployment proceeds
- Success notification sent

### Failure
- Deployment rolled back
- Failure notification sent with details
- Manual intervention required

## Configuration

### Environment Variables

```bash
# Base URL for smoke tests (optional, defaults to localhost:8000)
SMOKE_TEST_BASE_URL=https://api.example.com

# Timeout in seconds (optional, defaults to 300)
SMOKE_TEST_TIMEOUT=300
```

### Custom Endpoints

To add custom smoke test endpoints, modify `tests/e2e/test_deployment_smoke.py`:

```python
def test_custom_endpoint(self):
    """Test custom endpoint."""
    response = requests.get(f"{self.BASE_URL}/custom/endpoint", timeout=30)
    assert response.status_code == 200
```

## Troubleshooting

### Smoke Tests Timeout

**Cause**: Service not responding within 5 minutes

**Solutions**:
1. Check service logs for startup errors
2. Verify network connectivity
3. Increase timeout if needed (not recommended for production)

### Health Check Fails

**Cause**: Application not starting or unhealthy

**Solutions**:
1. Check application logs
2. Verify database and cache connectivity
3. Check resource limits (CPU, memory)

### Database Connectivity Fails

**Cause**: Database connection issues

**Solutions**:
1. Verify database credentials
2. Check database service status
3. Verify network connectivity to database

### Cache Connectivity Fails

**Cause**: Cache connection issues

**Solutions**:
1. Verify Redis/cache service status
2. Check cache configuration
3. Verify network connectivity to cache

## Best Practices

1. **Run Early**: Smoke tests should run immediately after deployment
2. **Keep Fast**: Tests should complete within 1-2 minutes
3. **Be Specific**: Test only critical paths, not full functionality
4. **Monitor**: Set up alerts for smoke test failures
5. **Document**: Keep test coverage updated as system evolves

## Maintenance

### Adding New Tests

When adding new critical services, add corresponding smoke tests:

1. Add test method to `TestDeploymentSmoke` class
2. Update documentation
3. Test locally before committing

### Removing Tests

When deprecating services, remove corresponding smoke tests:

1. Remove test method from class
2. Update documentation
3. Verify no other tests depend on removed service

## Related Documentation

- [CI/CD Pipeline](../.github/workflows/docker-ci-cd.yml)
- [Health Check Endpoints](../api/routers/healthz.py)
- [Deployment Guide](../DOCKER_PRODUCTION_DEPLOYMENT.md)
