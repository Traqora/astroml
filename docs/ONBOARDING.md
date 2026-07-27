# Contributor Onboarding Guide

Welcome to **AstroML**! 🎉 This guide walks you through the steps to get the repository up and running locally.

## Prerequisites
- Python 3.11+ (or the version specified in `pyproject.toml`)
- Docker & Docker‑Compose
- Git

## Quick Start
```bash
git clone https://github.com/Traqora/astroml.git
cd astroml
make dev-setup   # builds containers, seeds data, runs health checks
```

## Running the Test Suite
```bash
make test       # runs all unit and integration tests
make test-api   # runs API‑specific tests only
```

## Code Style
- Follow the existing formatting (Black, isort). Run `make format` to auto‑format.
- Lint with `make lint`.

## Pull‑Request Checklist
- [ ] All tests pass (`make test`).
- [ ] Linting clean (`make lint`).
- [ ] Updated `CHANGELOG.md` if applicable.

## Requesting a Review
- Push your branch, open a PR, assign reviewers, and add the `needs‑review` label.

Happy hacking! 🚀
