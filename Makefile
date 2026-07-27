.PHONY: help quickstart test test-api lint format clean install run-api canary-deploy canary-promote rollback-llm dependency-tree llm-test llm-eval llm-cost-check llm-validate-prompts llm-safety-scan security-audit secrets-scan

help:
	@echo "AstroML Development Commands"
	@echo "============================"
	@echo ""
	@echo "make quickstart          Run quick start: ingestion → graph → train pipeline"
	@echo "make quickstart-verbose  Run quick start with verbose output"
	@echo "make test                Run full test suite"
	@echo "make test-api            Run API integration tests only"
	@echo "make lint                Run linters (flake8, mypy)"
	@echo "make format              Format code (black, isort)"
	@echo "make install             Install development dependencies"
	@echo "make dependency-tree     Print the resolved dependency tree (pipdeptree)"
	@echo "make clean               Clean build artifacts and cache"
	@echo "make run-api             Start the FastAPI dev server on localhost:8000"
	@echo "make canary-deploy       Deploy LLM canary to Kubernetes"
	@echo "make canary-promote      Promote canary to stable"
	@echo "make rollback-llm        Rollback LLM canary deployment"
	@echo "make llm-test             Run LLM test suite (CI)"
	@echo "make llm-eval             Run LLM evaluation benchmarks"
	@echo "make llm-cost-check       Check LLM cost against baseline"
	@echo "make llm-validate-prompts Validate all prompt templates"
	@echo "make llm-safety-scan      Run LLM safety scan"
	@echo "make security-audit       Run pip-audit to check for vulnerable dependencies"
	@echo "make secrets-scan         Run detect-secrets to scan for leaked credentials"
	@echo ""

quickstart:
	python -m astroml.quick_start

quickstart-verbose:
	python -m astroml.quick_start --num-ledgers 200 --num-accounts 100 --epochs 20

test:
	pytest tests/ -v

test-api:
	pytest api/tests/ -v --tb=short

lint:
	flake8 astroml/ tests/
	mypy astroml/ --ignore-missing-imports

format:
	black astroml/ tests/
	isort astroml/ tests/

# Issue #562 — resolved dependency tree, e.g. to check what a pin bump would
# actually pull in, or to spot conflicting transitive requirements. Requires
# pipdeptree, installed via requirements-dev.txt.
dependency-tree:
	pipdeptree --warn silence

.PHONY: validate-build
validate-build:
	bash scripts/validate_build.sh

run-api:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

validate-config:
	python -c "from main import validate_config; r = validate_config(); print(f'Valid: {r[\"valid\"]}'); [print(f'  ERROR: {e}') for e in r['errors']]"

install:
	pip install -e ".[dev]"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache build/ dist/ *.egg-info
	rm -rf benchmark_results/quickstart .astroml_state_quickstart

install:
	pip install -e "[dev]"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache build/ dist/ *.egg-info
	rm -rf benchmark_results/quickstart .astroml_state_quickstart

# Dev setup target – start full stack, seed data, run health checks
.PHONY: dev-setup
dev-setup:
	@echo "🚀 Starting local development environment…"
	@docker compose -f docker-compose.yml up -d --build
	@./scripts/seed_data.sh
	@./scripts/health_check.sh
	@echo "✅ Development environment ready."

.PHONY: canary-deploy
canary-deploy:
	@echo "🚀 Deploying LLM canary..."
	REGISTRY=$(shell grep -E '^REGISTRY' .github/workflows/llm-cicd.yml | head -n1 | sed 's/.*: //' | tr -d '"')
	IMAGE_TAG=llm-$(shell git rev-parse --short HEAD)
	REPO=$(shell basename $$(pwd))
	NAMESPACE=astroml ./scripts/canary-deploy.sh

.PHONY: canary-promote
canary-promote:
	@echo "✅ Promoting canary to stable..."
	REGISTRY=$(shell grep -E '^REGISTRY' .github/workflows/llm-cicd.yml | head -n1 | sed 's/.*: //' | tr -d '"')
	IMAGE_TAG=llm-$(shell git rev-parse --short HEAD)
	REPO=$(shell basename $$(pwd))
	NAMESPACE=astroml ./scripts/canary-promote.sh

.PHONY: rollback-llm
rollback-llm:
	@echo "🔄 Rolling back LLM deployment..."
	NAMESPACE=astroml STABLE_DEPLOYMENT=astroml-api ./scripts/auto-rollback.sh

.PHONY: llm-test
llm-test:
	@echo "🧪 Running LLM test suite..."
	bash scripts/ci/run-llm-tests.sh

.PHONY: llm-eval
llm-eval:
	@echo "📊 Running LLM evaluation benchmarks..."
	bash scripts/ci/run-eval.sh

.PHONY: llm-cost-check
llm-cost-check:
	@echo "💰 Checking LLM cost against baseline..."
	bash scripts/ci/check-cost.sh

.PHONY: llm-validate-prompts
llm-validate-prompts:
	@echo "📝 Validating all prompt templates..."
	bash scripts/ci/validate-prompts.sh

.PHONY: llm-safety-scan
llm-safety-scan:
	@echo "🔒 Running LLM safety scan..."
	bash scripts/ci/safety-scan.sh

.PHONY: security-audit
security-audit:
	@echo "🔒 Running pip-audit to check for vulnerable dependencies..."
	pip-audit --fix --require-hashes

.PHONY: secrets-scan
secrets-scan:
	@echo "🔍 Running detect-secrets to scan for leaked credentials..."
	detect-secrets scan --baseline .secrets.baseline
