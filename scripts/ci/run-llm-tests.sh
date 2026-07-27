#!/bin/bash
set -euo pipefail

echo "Running LLM unit tests..."
python3 -m pytest tests/llm/ -v --tb=short --cov=astroml.llm --cov-report=term --cov-report=xml

echo ""
echo "Running LLM integration tests..."
python3 -m pytest tests/llm/ -v --tb=short -m "not gpu"

echo ""
echo "All LLM tests passed."
