#!/usr/bin/env bash
set -eo pipefail

echo "Running API load/performance smoke tests..."
# Fails if tests regress beyond pytest-benchmark thresholds (usually handled via --benchmark-fail-fast or historical comparison)
# We can use pytest-benchmark to fail if it's too slow by parsing or relying on assertion limits if we added them.
# Here we just run them and store the JSON output.

python -m pytest tests/performance/test_api_endpoints.py --benchmark-only --benchmark-json=api_benchmark.json

echo "Performance smoke tests completed successfully."
