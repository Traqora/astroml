# Contributor Onboarding

Welcome! This is the full path from "no checkout" to "PR open" on AstroML.
Every command here was run against the repository as of this writing; where a
convenience wrapper does not work yet, that is stated rather than glossed over.

!!! note "Which docs build is which"
    CI builds these pages with **Sphinx** (`.github/workflows/docs.yml` runs
    `make html` in `docs/`), configured in `docs/conf.py`. A
    `docs/mkdocs.yml` also exists but nothing installs or runs mkdocs, so
    editing it has no effect on the published site.

## What you need

- Python 3.10 or newer (the classifiers and `black`/`ruff` configs target
  `py310`).
- Git, and a GitHub account.
- Docker with the compose plugin — only for the database-backed steps. Most
  tests do not need it.

## 1. Fork and clone

Fork <https://github.com/Traqora/astroml> with the GitHub "Fork" button, then:

```bash
git clone https://github.com/<your-user>/astroml.git
cd astroml
git remote add upstream https://github.com/Traqora/astroml.git
git fetch upstream
```

Keeping `upstream` pointed at `Traqora/astroml` is what makes step 2 cheap.

## 2. Create a branch off current main

```bash
git checkout -b feat/my-change upstream/main
```

Branching from `upstream/main` rather than your fork's `main` avoids building
on a stale copy. Before you start coding on an existing issue, comment on it so
maintainers can assign it to you.

## 3. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

!!! warning "Do not use `make install`"
    `Makefile` defines `install:` twice. GNU Make keeps the **last** recipe,
    which is `pip install -e "[dev]"` — missing the `.` package path, so it
    fails. The `pip install -e ".[dev]"` line above is the first, correct
    definition.

The dev extra brings in `pytest`, `black`, `ruff`, `mypy`, and `interrogate`.

## 4. Run the tests you touched

`make test` is `pytest tests/ -v`, but do not reach for it while iterating —
see the note below. Target a path instead, which is also the fastest loop:

```bash
# One package
pytest tests/ingestion -q

# One file
pytest tests/features/graph/test_snapshot_rfc7807.py -v

# One test, by node id
pytest "tests/features/graph/test_snapshot_rfc7807.py::test_window_snapshot_valid_bounds_do_not_raise" -v
```

Keyword and marker filters need a path argument, because an error during
collection aborts the whole run:

```bash
pytest tests/ingestion -k retry -q      # 10 of 162 selected
pytest tests/ingestion -m "not gpu" -q  # gpu tests auto-skip on CPU runners
```

Two markers are declared in `pyproject.toml`: `gpu` and `e2e`. API tests live
outside `tests/` and are run with `make test-api`
(`pytest api/tests/ -v --tb=short`).

!!! warning "The suite is not green on `main`, and `make test` does not even run"
    `pytest tests/` stops at collection: 46 test modules fail to import because
    `astroml/cache/graph_cache.py` does not compile, and that makes
    `astroml.features` unimportable for anything under it. A further ~49 errors
    come from optional dependencies (`fastapi`, `polars`, `torch_geometric`, …)
    being absent from your interpreter. Adding
    `--continue-on-collection-errors` gets you a real signal instead of an
    abort:

    ```bash
    pytest tests/ -q --continue-on-collection-errors
    # 122 failed, 2402 passed, 3 skipped, 108 errors
    ```

    That count is the baseline on `main`. Your job is to not make it worse, not
    to fix it in a contribution PR. When you are unsure whether a failure is
    yours, run the same test against a pristine checkout:

    ```bash
    git worktree add /tmp/astroml-main upstream/main --detach
    cd /tmp/astroml-main && pytest tests/ingestion/test_retry_logic.py -q
    ```

    For example `tests/ingestion/test_retry_logic.py` (10 failures) and
    `test_snapshot_rfc7807.py` (4 failures) already fail on `main`.


## 5. The Docker database, when you need it

Only ingestion and DB-backed graph tests need PostgreSQL. Bring up just those
services — `docker compose up -d` with no arguments also starts the API,
Celery, the feature store, and two training workers, which is much more than a
unit-test loop needs:

```bash
docker compose up -d postgres redis
docker compose ps postgres
```

`postgres` is `postgres:15-alpine`, listening on the host's `5432`, with
database/user `astroml` and password `astroml_password`. `migrations/00_init.sql`
is mounted into `/docker-entrypoint-initdb.d`, so extensions and the `astroml`
schema are created automatically — but **only the first time the data volume is
initialised**. If you had an older volume, recreate it:

```bash
docker compose down -v postgres   # -v deletes the postgres_data volume
docker compose up -d postgres
```

Now point the app at it. The default in `config/database.yaml` is
`host: localhost` with an **empty** password, which does not match the compose
service, so set the URL explicitly. Note the variable name is
`ASTROML_DATABASE_URL`, not the `DATABASE_URL` that `.env.example` shows, and
that `resolve_database_url()` prefers it over the YAML file:

```bash
export ASTROML_DATABASE_URL="postgresql://astroml:astroml_password@localhost:5432/astroml"
```

`localhost` and the literal password are right for a Python process running on
your machine. Inside the compose network the host is `postgres` instead, which
is what `POSTGRES_HOST=postgres` in `.env.example` is for.

Verify the connection:

```bash
python -c "from astroml.db.session import resolve_database_url as u; print(u())"
```

To stop: `docker compose stop postgres redis`, add `down -v` to also discard the
data.

!!! warning "`make dev-setup` does not do what its name suggests"
    `make dev-setup` runs `docker compose -f docker-compose.yml up -d --build`
    for the **entire** stack, then `./scripts/seed_data.sh` and
    `./scripts/health_check.sh`. Both scripts are committed non-executable, so
    those last two calls fail with exit 126, and both are placeholders that only
    `echo` — they seed nothing and check nothing. Use the explicit steps above.

## 6. Checks before you push

```bash
make format       # black + ruff --fix --select I on astroml/, tests/, api/
make lint         # ruff check, then mypy astroml/ --ignore-missing-imports
make lint-docs    # interrogate, enforces the docstring coverage floor
make complexity   # xenon maintainability budget
```

`make lint-docs` enforces a coverage floor (`fail-under = 68.0` in
`pyproject.toml`), with `ignore-init-method = true` — the house style documents
constructor arguments in the class docstring's `Args:` block, not in
`__init__`. Match that.

Docs get built the same way CI does:

```bash
cd docs && pip install -r requirements.txt && make html
```

## 7. Open the pull request

```bash
git add <paths>            # avoid `git add -A`; it picks up build junk
git commit -m "fix(ingestion): <what and why>"
git push -u origin feat/my-change
gh pr create --repo Traqora/astroml --fill
```

Prefer Conventional Commit subjects (`feat:`, `fix:`, `docs:`, `refactor:`,
`test:`, `chore:`), matching recent history. Then:

- Say which issue it closes with `Closes #<number>` so it links automatically.
- Describe what changed and why, not just what files you touched.
- Note design decisions and trade-offs, especially where you chose between two
  reasonable approaches.
- For behaviour changes, include the command that demonstrates them.
- Keep one PR per concern; unrelated drive-by fixes make review slower.

Maintainers review and assign. If a check fails in CI, read the failing job's
log rather than re-pushing a guess — the docs, pytest, and pre-commit workflows
are separate and say which one broke.

## DO / DON'T

Distilled from `CONTRIBUTING.md`, which is the source of truth if the two
disagree.

| Do | Don't |
| --- | --- |
| Keep new functions under McCabe complexity **10** | Push a function over the **15** hard CI limit |
| Refactor an over-budget function when you touch it | Leave complexity alone because it predates you |
| Log through `astroml.utils.logging` | Use `print()` in library code |
| Emit structured logs on critical paths (ingestion, training, API entrypoints) | Ship a new pipeline stage with no observability |
| Use `logger.exception(...)` inside `except` blocks | Swallow failures or log only `str(e)` |
| Annotate public API signatures fully | Land a new public function untyped |
| Use built-in generics: `dict[str, Any]`, `list[str]` | Add `typing.Dict` / `typing.List` imports |
| Target Python 3.10+ | Use syntax that breaks the declared minimum |
| Run `black`, `ruff`, `make lint-docs` before submitting | Ask reviewers to fix formatting |
| Keep functions small and testable | Write a 200-line `run()` |
| Document public classes, methods, functions | Ship an undocumented public class |
| Comment on an issue to claim it before starting | Duplicate someone else's in-flight PR |

## Where to read next

- [Graph construction](graph-construction.rst) — how rolling windows, snapshots,
  and edge types fit together.
- `README.md` — project overview and quickstart.
- `CONTRIBUTING.md` — the standards summarised above.
