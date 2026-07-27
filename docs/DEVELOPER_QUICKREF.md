# AstroML Developer Quick Reference

One-page reference for common development tasks. Print-friendly.

---

## Common Commands

### Testing

```bash
# Run the full test suite
pytest tests/

# Run API tests only
pytest api/tests/ -v --tb=short

# Run a single test file
pytest tests/test_security.py -v

# Run tests with coverage
pytest tests/ --cov=astroml --cov-report=term-missing

# Run only fast (non-integration) tests
pytest tests/ -m "not integration"
```

### Code Quality

```bash
# Format code (black + isort)
make format
# or directly:
black astroml/ tests/ api/
isort astroml/ tests/ api/

# Lint
make lint
# or directly:
ruff check .
black --check .

# Type checking
mypy astroml/
```

### Database

```bash
# Run Alembic migrations
alembic upgrade head

# Generate a new migration
alembic revision --autogenerate -m "describe your change"

# Downgrade one step
alembic downgrade -1

# Show current migration state
alembic current
```

### Services

```bash
# Start all services via Docker Compose (PostgreSQL + Redis)
docker compose up -d

# Start the FastAPI dev server
make run-api
# or:
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Full local dev environment (Docker + seed data + health checks)
make dev-setup
```

### Security

```bash
# Audit Python dependencies for known CVEs
make security-audit
# or:
pip-audit

# Scan for secrets/credentials in the codebase
make secrets-scan
# or:
detect-secrets scan --baseline .secrets.baseline
```

---

## Troubleshooting

**Tests fail with `ImportError: No module named 'torch'`**
Use the CPU-only requirements for unit testing:
```bash
pip install -r requirements-cpu.txt
```

**`alembic upgrade head` fails with "target database is not up to date"**
Check the current revision and re-run:
```bash
alembic current
alembic upgrade head
```

**FastAPI server won't start — `Address already in use`**
Find and kill the process using port 8000:
```bash
lsof -ti:8000 | xargs kill -9
```

**Redis connection errors in tests**
Tests use in-memory SQLite and mock Redis by default. If you see real Redis errors, ensure `AUTH_ENABLED=false` and `DISABLE_SCHEDULER=true` are set, or run:
```bash
docker compose up -d redis
```

**`mypy` reports missing stubs for `torch` / `networkx`**
Add `--ignore-missing-imports` or install the stubs package:
```bash
pip install types-networkx
mypy astroml/ --ignore-missing-imports
```

---

## Debugging Tips

### pdb (built-in)

Insert a breakpoint anywhere in Python code:
```python
import pdb; pdb.set_trace()
```
Or use the built-in shorthand (Python 3.7+):
```python
breakpoint()
```

Common pdb commands: `n` (next), `s` (step into), `c` (continue), `p <expr>` (print), `q` (quit).

### ipdb (enhanced pdb with tab-completion)

```bash
pip install ipdb
```
```python
import ipdb; ipdb.set_trace()
```

### Debugging FastAPI requests

Enable detailed request logging by setting the log level:
```bash
uvicorn api.app:app --reload --log-level debug
```

Inspect the OpenAPI docs interactively at `http://localhost:8000/docs`.

### Pytest debugging

Run a single failing test with full output and drop into pdb on failure:
```bash
pytest tests/test_security.py::TestSecurity::test_sql_injection -s --pdb
```

Print captured output even when tests pass:
```bash
pytest tests/ -s
```

---

## Project Layout (abridged)

```
astroml/            Core ML pipeline (ingestion, features, models, training)
api/                FastAPI REST API
  routers/          Route handlers (one file per domain)
  schemas/          Pydantic request/response schemas
  auth/             JWT auth + rate limiting
  middleware/       HTTPS, CSP, validation middleware
  tests/            API-level tests
tests/              Full test suite (unit + integration)
docs/               Documentation
configs/            Hydra experiment configs
migrations/         Alembic database migrations
```

---

*Keep this file to one page. For in-depth docs see `docs/` and `README.md`.*
