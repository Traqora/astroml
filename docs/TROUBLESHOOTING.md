# Troubleshooting Guide

This guide collects common issues encountered while installing, configuring, ingesting data, training models, or running the API.

For the clean-machine install path itself, follow
[ONBOARDING.md](ONBOARDING.md). For which CI/CD and local workflows need
PostgreSQL, see [CI_DATABASE_REQUIREMENTS.md](CI_DATABASE_REQUIREMENTS.md).
Features that are not yet supported are listed in [ROADMAP.md](ROADMAP.md).

## Supported boundaries

The documented v1 workflow is:

1. A fresh virtual environment with `pip install -e ".[dev]"`.
2. Targeted `pytest <path>` runs (not `make test`).
3. PostgreSQL **only** for ingestion, DB-backed graph, and API-session tests;
   everything else runs with no database service.

What is **out of scope** for the v1 path, and therefore not supported by this
guide:

- `make install`, `make dev-setup`, and `make test` as one-shot commands — see
  the failures below for why each is broken and what to run instead.
- The full `docker compose up -d` stack for a unit-test loop. Start only
  `postgres redis` when you need the database.
- Running tests that import optional dependencies (`fastapi`, `polars`,
  `torch_geometric`) without installing the matching requirements file.

## Installation failures

These are the install commands that fail on a clean machine, why they fail,
and the command that works instead. Each is documented because a contributor
following the README in order hits one of them first.

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| `make install` fails with a pip usage error | The `Makefile` defines `install:` twice; GNU Make keeps the last recipe, which is `pip install -e "[dev]"` and is missing the `.` package path | Run `pip install -e ".[dev]"` directly instead of `make install` |
| `make dev-setup` fails with exit 126 | It starts the entire stack, then runs `scripts/seed_data.sh` and `scripts/health_check.sh`, which are committed non-executable and only `echo` placeholders | Bring up only what you need: `docker compose up -d postgres redis`, then the explicit steps in [ONBOARDING.md](ONBOARDING.md) |
| `make test` aborts during collection | `pytest tests/` stops at collection: modules fail to import because `astroml/cache/graph_cache.py` does not compile, and optional dependencies are absent | Run a targeted path, or `pytest tests/ -q --continue-on-collection-errors` for a real signal; compare against a pristine checkout before treating a failure as yours |
| `ModuleNotFoundError: No module named 'astroml'` | The package is not installed in the active interpreter | Activate the venv and run `pip install -e ".[dev]"`; do not rely on `make install` |
| `ModuleNotFoundError: No module named 'fastapi'` (or `polars`, `torch_geometric`) | The CPU/full requirements do not declare the API or optional ML extras | Install `requirements-api.txt` for API tests, `requirements-train.txt` for GPU graph models; install requirement files in **separate** pip invocations to avoid `error: resolution-too-deep` |
| `pip` fails with `error: resolution-too-deep` | Two requirement graphs were resolved in one `pip install -r a.txt -r b.txt` call | Install each requirements file in its own `pip install -r ...` command |
| `python` points to the wrong interpreter | Multiple Python versions are installed | Use the interpreter that owns the active virtual environment |
| Import errors after pulling a new branch | Dependencies changed or the package is not installed in editable mode | Reinstall dependencies and run `pip install -e .` |

## Configuration failures

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| `DATABASE_URL` is set but the app ignores it | The app reads `ASTROML_DATABASE_URL`, not `DATABASE_URL`; `resolve_database_url()` prefers the env var over the YAML file | `export ASTROML_DATABASE_URL="postgresql://astroml:astroml_password@localhost:5432/astroml"` |
| Database connection refused / authentication failed | `config/database.yaml` ships `host: localhost` with an **empty** password, which does not match the compose service (`astroml_password`) | Set `ASTROML_DATABASE_URL` explicitly, or edit [config/database.yaml](../config/database.yaml) to match |
| Works on the host, fails inside compose | `localhost` is right for a Python process on your machine; inside the compose network the host is `postgres` | Use `POSTGRES_HOST=postgres` inside compose (see `.env.example`) and `localhost` on the host |
| Migrations did not apply after a schema change | `migrations/00_init.sql` only runs the **first time the data volume is initialised**; an older volume keeps the old schema | `docker compose down -v postgres` to recreate the volume, then `docker compose up -d postgres` |
| `ASTROML_DATABASE_URL` is ignored | The variable was not exported in the current shell | Export it in the active shell before running the command |

## Installation Issues

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| `ModuleNotFoundError` for a dependency | The virtual environment was not created or dependencies were not installed | Create a fresh virtual environment and run `pip install -r requirements.txt` |
| `python` points to the wrong interpreter | Multiple Python versions are installed | Use the interpreter that owns the active virtual environment |
| Import errors after pulling a new branch | Dependencies changed or the package is not installed in editable mode | Reinstall dependencies and run `pip install -e .` |

## Database Issues

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| Connection refused | PostgreSQL is not running or the host/port is incorrect | Confirm the database service is running and update [config/database.yaml](../config/database.yaml) |
| Authentication failed | Wrong username, password, or role | Check credentials and verify the database user has access to the target database |
| `ASTROML_DATABASE_URL` is ignored | The environment variable is not exported in the current shell | Export the variable in the active shell before running the command |

## Ingestion Issues

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| Backfill stops with timeout errors | Network issues or rate limiting from upstream services | Retry with a smaller ledger range and check network connectivity |
| Duplicate rows appear after reruns | Ingestion is not fully idempotent for the target range | Re-run carefully and verify state tracking before repeating the operation |
| Missing data after ingestion | The requested ledger range or filters were incorrect | Validate the requested range and confirm the database configuration |

## Training Issues

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| CUDA out-of-memory errors | The model or batch size is too large for the GPU | Reduce `batch_size`, lower model width, or switch to CPU |
| Training loss does not improve | Hyperparameters may be poorly chosen or the data split is leaking | Review the config under [configs/](../configs/) and confirm the split settings |
| Reproducibility differs across runs | Random seeds or config values changed | Fix the seed and keep the config under version control |

## API Issues

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| CORS errors when calling the API | Browser policy or API configuration mismatch | Verify the allowed origins and confirm the request headers |
| Authentication failures | Missing or invalid credentials | Check the environment variables or headers used by the client |
| Empty or unexpected responses | The service is failing before the request completes | Review the service logs and confirm that the database connection is healthy |

## Collecting Diagnostic Information

When reporting an issue, gather the following:

- The exact command that failed
- The relevant configuration files, especially [config/database.yaml](../config/database.yaml) and [config/feature_store.yaml](../config/feature_store.yaml)
- The traceback or error output
- The Python version and installed package versions
- Any relevant logs from the failed run

## Related Documentation

- [README.md](../README.md)
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ONBOARDING.md](ONBOARDING.md)
- [CONFIGURATION.md](CONFIGURATION.md)
- [CI_DATABASE_REQUIREMENTS.md](CI_DATABASE_REQUIREMENTS.md) — which workflows need PostgreSQL
- [ROADMAP.md](ROADMAP.md) — features that are not yet supported
- [SECURITY.md](../SECURITY.md)
