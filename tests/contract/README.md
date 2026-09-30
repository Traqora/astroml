# Contract Tests

This directory contains contract tests for external APIs that AstroML relies upon, including:
- **Stellar Horizon API**: We verify that the Stellar network endpoints return responses with expected keys and types.
- **MLflow API**: We verify the expected schema of the MLflow tracking service.

## Running Contract Tests

Contract tests can be run using pytest:
```bash
pytest tests/contract/
```

## Failing CI on Contract Breakages

These tests run as part of our CI pipeline. If an external API changes its response schema in a backwards-incompatible way (or if their API goes down and returns 500s), the contract test will fail. This acts as an early warning system.

When a contract test fails, do **not** blindly update the test. Verify if the external API change broke our internal parsers and update the application code and the test simultaneously.
