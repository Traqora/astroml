# Which Workflows Need PostgreSQL

AstroML's CI/CD tree mixes jobs that spin up a PostgreSQL service container
with jobs that never touch a database. This page says which is which, so a
contributor can run exactly the checks they need on a clean machine without
standing up services they do not.

The rule this page is checked against is automated:
`scripts/check_workflow_postgres.py` scans `.github/workflows/*.yml` and
verifies that every workflow listed here as needing PostgreSQL actually starts
a `postgres` service container (or sets a `postgresql://` connection string),
and that every workflow listed as *not* needing it does not. If you add or
remove a PostgreSQL service from a workflow, update the tables below in the
same change or that test fails.

## Supported boundary

- **Needs PostgreSQL** — the job starts a `postgres:*` service container and
  sets `DATABASE_URL` (or `ASTROML_DATABASE_URL`) to a `postgresql://`
  address. It cannot pass without that service running.
- **Does not need PostgreSQL** — the job runs with no database service. It may
  run unit tests, build images, build docs, or lint, none of which connect to
  PostgreSQL. A green run does not prove the database-backed paths work.

The quick start (`make quickstart` / `python -m astroml.quick_start`) uses
synthetic data and SQLite; it does **not** need PostgreSQL. See
[QUICKSTART_GUIDE.md](../QUICKSTART_GUIDE.md).

## CI/CD workflows

### Workflows that need PostgreSQL

| Workflow | Job | Postgres image | Connection string |
| --- | --- | --- | --- |
| [ci.yml](../.github/workflows/ci.yml) | `lint-typecheck-test` | `postgres:14` | `DATABASE_URL=postgresql://astroml:astroml@localhost:5432/astroml_test` |
| [e2e-tests.yml](../.github/workflows/e2e-tests.yml) | `e2e-tests` | `postgres:15` | `DATABASE_URL=postgresql://test:test@localhost:5432/astroml_test` |
| [nightly.yml](../.github/workflows/nightly.yml) | `integration` | `postgres:15` | `ASTROML_DATABASE_URL=postgresql://astroml:astroml@localhost:5432/astroml_test` |

### Workflows that do NOT need PostgreSQL

| Workflow | Job(s) | What it runs |
| --- | --- | --- |
| [pytest.yml](../.github/workflows/pytest.yml) | `test`, `test-api` | `pytest -m "not gpu"` and `pytest api/tests/` with no service container |
| [smoke-tests.yml](../.github/workflows/smoke-tests.yml) | `run-smoke-tests` | Manual run against a deployed API URL |
| [release.yml](../.github/workflows/release.yml) | `release`, `create-release` | Builds and pushes Docker images, creates the GitHub Release |
| [docker.yml](../.github/workflows/docker.yml) | image build | Builds multi-arch images; no test database |
| [docs.yml](../.github/workflows/docs.yml) | `build`, `deploy` | Builds Sphinx docs and runs `scripts/check_doc_links.py` |
| [pre-commit.yml](../.github/workflows/pre-commit.yml) | hooks | Runs pre-commit hooks |
| [security.yml](../.github/workflows/security.yml) | scans | Security scanning |
| [safety-scan.yml](../.github/workflows/safety-scan.yml) | scan | Dependency vulnerability scan |
| [pr-size-limit.yml](../.github/workflows/pr-size-limit.yml) | check | PR size policy |
| [performance.yml](../.github/workflows/performance.yml) | bench | Performance benchmarks (no DB service) |

Workflows not listed above are treated as not needing PostgreSQL. The
automated check enforces this: any workflow that starts a `postgres` service
container must appear in the "needs PostgreSQL" table.

## Local test paths

Whether your local loop needs PostgreSQL depends on what you run, not on
which workflow file it lives in:

| You are running... | PostgreSQL needed? |
| --- | --- |
| `make quickstart` / `python -m astroml.quick_start` | No — synthetic data, SQLite |
| Unit tests under `tests/ingestion/`, `tests/features/` that do not open a session | No |
| Ingestion backfill tests and DB-backed graph tests | Yes |
| `api/tests/` against a real session | Yes |
| `pytest tests/ -m "not gpu"` (the `pytest.yml` shape) | No |
| `make lint`, `make format`, `make lint-docs`, `make complexity` | No |
| Sphinx docs build (`make html` in `docs/`) | No |
| `docker compose up -d postgres redis` then the ingestion/graph tests | Yes |

To bring up only PostgreSQL locally (do not start the whole stack for a unit
test loop — see [ONBOARDING.md](ONBOARDING.md)):

```bash
docker compose up -d postgres redis
export ASTROML_DATABASE_URL="postgresql://astroml:astroml_password@localhost:5432/astroml"
```

Note the variable name is `ASTROML_DATABASE_URL`, and that
`resolve_database_url()` prefers it over the YAML file. The default in
`config/database.yaml` ships an **empty** password, which does not match the
compose service, so set the URL explicitly.

## Docker Compose services

`docker compose up -d` with no arguments starts more than a unit-test loop
needs. Only the services below depend on PostgreSQL:

| Service | Depends on `postgres`? |
| --- | --- |
| `postgres` | is the database |
| `api` | yes |
| `ingestion` | yes |
| `streaming` | yes |
| `feature-store` | yes |
| `training-gpu`, `training-cpu` | yes |
| `dev`, `production` | yes |
| `redis`, `celery-worker`, `flower` | no |
| `prometheus`, `grafana` | no (monitoring profile) |
| `soroban-dev`, `soroban-build`, `soroban-test` | no (soroban profile) |

## Related documentation

- [Onboarding](ONBOARDING.md) — the clean-machine path, including the Docker
  database step.
- [Troubleshooting](TROUBLESHOOTING.md) — database connection failures.
- [Configuration reference](CONFIGURATION.md) — `database.yaml` and
  `ASTROML_DATABASE_URL`.
