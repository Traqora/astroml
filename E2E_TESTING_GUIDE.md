# End-to-End API Tests

## Overview

Comprehensive E2E test suite for AstroML API covering critical user journeys, error handling, performance, and reliability.

## Test Coverage

### User Journeys
- **Health Check**: API availability
- **Authentication**: Registration and login flows
- **Transactions**: Fetching and filtering transaction data
- **Fraud Detection**: Fraud detection endpoints
- **Accounts**: Account management operations
- **Loyalty**: Points and redemption flows
- **Monitoring**: Metrics and performance data
- **Discussions**: Community forum interactions
- **Contributors**: Contributor dashboard
- **Notifications**: Notification management

### Quality Assurance
- Error handling and validation
- Concurrent request handling
- Response format consistency
- Rate limiting
- CORS headers
- Performance benchmarks

## Running Tests Locally

### Prerequisites
```bash
pip install pytest pytest-asyncio pytest-cov
pip install -r requirements.txt
```

### Start Services
```bash
docker-compose -f docker-compose.e2e.yml up -d
```

### Run Tests
```bash
# All E2E tests
pytest tests/e2e/test_api_e2e.py -v

# Specific test class
pytest tests/e2e/test_api_e2e.py::TestCriticalUserJourneys -v

# With coverage report
pytest tests/e2e/test_api_e2e.py --cov=api --cov-report=html

# Parallel execution
pytest tests/e2e/test_api_e2e.py -n auto
```

### View Reports
- JSON Report: `test-results/e2e-report.json`
- HTML Report: `test-results/e2e-report.html`
- Coverage: `htmlcov/index.html`

## Running in Docker

```bash
# Run full E2E test suite in containers
docker-compose -f docker-compose.e2e.yml run pytest-e2e

# Stop all containers
docker-compose -f docker-compose.e2e.yml down
```

## CI/CD Integration

Tests run automatically on:
- Push to `main` or `develop`
- All pull requests
- Daily schedule (2 AM UTC)

### Environment Variables
```
DATABASE_URL=postgresql://test:test@localhost:5432/astroml_test
REDIS_URL=redis://localhost:6379
API_BASE_URL=http://localhost:8000
GITHUB_TOKEN=<your-github-token>
```

## Flake Detection

The test suite automatically detects flaky tests by analyzing:
- Timeout errors
- Connection failures
- Intermittent errors

Flaky tests are reported in:
- JSON report: `summary.flaky_tests`
- HTML report: "Potentially Flaky Tests" section

## Performance Benchmarks

Tests validate response times:
- Single requests: < 5 seconds
- Concurrent requests (20): < 30 seconds

## Test Results

Test results are published to:
- GitHub PR comments (for pull requests)
- GitHub Actions artifacts
- Test reports directory

## Troubleshooting

### Tests timeout
```bash
# Increase timeout
pytest tests/e2e/test_api_e2e.py --timeout=60
```

### Database connection issues
```bash
# Check PostgreSQL is running
docker-compose -f docker-compose.e2e.yml logs postgres-e2e

# Reset database
docker-compose -f docker-compose.e2e.yml down -v
docker-compose -f docker-compose.e2e.yml up -d
```

### API not responding
```bash
# Check API logs
docker-compose -f docker-compose.e2e.yml logs api-e2e

# Wait for API to be ready
curl http://localhost:8000/health
```

## Contributing

When adding new tests:
1. Follow existing test naming conventions
2. Add docstrings explaining what is tested
3. Use appropriate assertions
4. Organize tests into logical classes
5. Run full suite before submitting PR

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/advanced/testing-dependencies/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
