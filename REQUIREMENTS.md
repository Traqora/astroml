# Python requirements files

Nine requirements files live in this repo. Pick the one(s) that match your
environment — most compose with `-r`, so "which files" is usually "one
environment-shaped file" plus "zero or more purpose-shaped add-ons."

## Decision tree

```
Need to train models (GPU/CUDA available)?
└─ yes → pip install -r requirements.txt
└─ no
   ├─ Need to run training or feature jobs (CPU only)?
   │  └─ yes → pip install -r requirements-cpu.txt
   ├─ Only want to train models, nothing else (no feature store/notebooks/viz)?
   │  └─ yes → pip install -r requirements-train.txt
   ├─ Only running the FastAPI service?
   │  └─ yes → pip install -r requirements-api.txt   (alias for api/requirements.txt)
   ├─ Just want to run the test suite / linters, no ML stack?
   │  └─ yes → pip install -r requirements-dev.txt
   ├─ Need MLflow experiment tracking?
   │  └─ yes → add  -r requirements-mlflow.txt  on top of whichever base above
   └─ Just want to load Hydra config / parse dataframes /
      run unit tests that don't touch torch?
      └─ yes → pip install -r requirements-minimal.txt
```

Building docs? See `docs/requirements.txt` (Sphinx + friends) — unrelated to
the astroml runtime stack, install separately.

## What each file ships

### `requirements.txt` — full GPU training stack
The everything-on-board file. Pulls the full GPU `torch` wheel,
`pytorch-lightning`, `mlflow`, the feature-store stack (redis, pyarrow,
fastparquet, networkx, click, rich), visualization (matplotlib, seaborn),
notebooks, and dev tooling (pytest + black + flake8 + mypy + ruff).

Use this on GPU CI runners and developer machines that build dashboards or
notebooks. This is what `Dockerfile`'s `base` and GPU-training stages
install — kept unchanged content-wise by the purpose-file split below so
existing builds don't break.

### `requirements-cpu.txt` — CPU-only training stack
Same shape as `requirements.txt` but pins the **CPU-only** torch wheels
from the official PyTorch CPU index:

```
torch==2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

Drops `mlflow`, `scikit-learn` standalone (still pulled transitively via
some libs), the feature-store stack, visualization, dev tooling, and
notebooks — they're not needed for headless CPU jobs. Pick this when:

- You're building the Docker image for production / CI.
- You're running batch ingestion or model serving on a CPU box.
- You want the fastest possible `pip install` for a smoke test.

### `requirements-minimal.txt` — Hydra + dataframes only
The smallest viable set: `numpy`, `pandas`, `polars`, `pyyaml`,
`hydra-core`, `omegaconf`. Nothing else. Use it when:

- You just want to import `astroml.config` and resolve a Hydra schema.
- You're running config-only unit tests in CI.
- You're embedding a small piece of astroml into another service and want
  to keep the install footprint tiny.

This is also the common base every purpose-built file below composes on top
of via `-r`.

### Purpose-built subsets (issue #561)

These four exist so you can install *only* what a given task needs, instead
of always reaching for the full `requirements.txt`. Each is a thin `-r`
composition, not a parallel copy of package lists — bump a version in the
base file and every purpose file picks it up.

- **`requirements-dev.txt`** — `-r requirements-minimal.txt` + testing
  (pytest and plugins, hypothesis) + linting/formatting/type-checking
  (black, flake8, mypy, ruff, isort, pre-commit) + `pipdeptree` (for `make
  dependency-tree`, issue #562). For contributors who want to run `pytest
  tests/` and pre-commit hooks without installing torch/redis/pyarrow/etc.
- **`requirements-api.txt`** — `-r api/requirements.txt`. A root-level
  alias so the FastAPI service's dependencies are discoverable next to the
  other `requirements-*.txt` files without duplicating them; `api/Dockerfile`
  keeps installing `api/requirements.txt` directly and is unaffected.
- **`requirements-train.txt`** — `-r requirements-minimal.txt` + `torch`,
  `torch-geometric`, `pytorch-lightning`, `scikit-learn`, `scipy`, `joblib`,
  `tqdm`. Everything needed to train a model and nothing else — no
  feature-store stack, no notebooks, no visualization, no mlflow. Smaller
  and faster to install than `requirements.txt` when all you're doing is
  training.
- **`requirements-mlflow.txt`** — `mlflow>=2.10.0` alone. Additive on top
  of `requirements-train.txt` (or any other base) — split out because not
  every training run needs a tracking server, and mlflow's own dependency
  footprint (Flask, alembic, gitpython, opentelemetry, …) is worth opting
  into explicitly:
  ```bash
  pip install -r requirements-train.txt -r requirements-mlflow.txt
  ```

**Known caveat:** `api/requirements.txt` (and therefore `requirements-api.txt`)
currently has a version conflict discovered while wiring this up: it pins
`python-multipart==0.0.6`, but `strawberry-graphql[fastapi]>=0.237.0`
requires `python-multipart>=0.0.7`, so `pip install -r requirements-api.txt`
fails dependency resolution as of this writing. This predates and is
independent of the #561/#562 work (a separate, unrelated `python-cors==1.0.0`
pin in the same file — for a package that doesn't exist on PyPI — was fixed
here since it's what the API actually imports via
`fastapi.middleware.cors.CORSMiddleware`, no separate package needed). The
`python-multipart`/`strawberry-graphql` conflict needs an explicit call on
which pin to move and is left for whoever owns the API service's dependency
set.

## Pin policy

Where a package appears in more than one file, the lower bound is held in
sync across all of them. The actual lower bounds in use:

| package          | pin                | files                                           |
|------------------|--------------------|-------------------------------------------------|
| `numpy`          | `>=1.24`           | requirements.txt, -cpu.txt, -minimal.txt        |
| `pandas`         | `>=2.0`            | requirements.txt, -cpu.txt, -minimal.txt        |
| `polars`         | `>=1.0`            | requirements.txt, -cpu.txt, -minimal.txt        |
| `pyyaml`         | `>=6.0`            | requirements.txt, -cpu.txt, -minimal.txt        |
| `hydra-core`     | `>=1.3.0`          | requirements.txt, -cpu.txt, -minimal.txt        |
| `omegaconf`      | `>=2.3.0`          | requirements.txt, -cpu.txt, -minimal.txt        |
| `torch`          | `>=2.0.0` / `+cpu` | requirements.txt (GPU), -cpu.txt (CPU), -train.txt |
| `torch-geometric`| `>=2.3.0`          | requirements.txt, -cpu.txt, -train.txt          |
| `scikit-learn`   | `>=1.3.0`          | requirements.txt, -train.txt                    |
| `mlflow`         | `>=2.10.0`         | requirements.txt, -mlflow.txt                   |
| `sqlalchemy`     | `>=2.0`            | requirements.txt, -cpu.txt                      |
| `psycopg2-binary`| `>=2.9`            | requirements.txt, -cpu.txt                      |
| `aiohttp`        | `>=3.9`            | requirements.txt, -cpu.txt                      |
| `stellar-sdk`    | `>=9.0.0`          | requirements.txt, -cpu.txt                      |
| `pytest`         | `>=7.4.0`          | requirements.txt, -dev.txt                      |

If you bump one, run `grep -E "^<package>\b" requirements*.txt` to confirm
you've bumped them in lockstep.

## Critical dependencies (issue #562)

Why each of these is here, not just what it is:

| Package | Purpose | Why this pin | Notes |
|---|---|---|---|
| `torch` | GNN model training/inference backbone | `>=2.0.0` for the current `torch.compile`/dynamo APIs the training code assumes | GPU wheel in `requirements.txt`, `+cpu` wheel in `requirements-cpu.txt` — see "Upgrading a major dependency" below before bumping the major version |
| `torch-geometric` | Graph-specific layers (message passing, `NeighborLoader`, etc.) on top of `torch` | `>=2.3.0` tracks the `torch` floor above — PyG's own compatibility matrix ties minor versions to specific torch releases | Installing it against a mismatched `torch` is the most common cause of import-time C-extension errors |
| `pytorch-lightning` | Training loop orchestration (checkpointing, early stopping, multi-GPU) | `>=2.0.0` for the current `Trainer` API | Only needed where `astroml.training` uses `Trainer`, not for plain `torch` model code |
| `mlflow` | Experiment tracking / model registry | `>=2.10.0` | Split into its own `requirements-mlflow.txt` (#561) — sizeable transitive footprint (Flask, alembic, gitpython, opentelemetry, databricks-sdk) that not every training run needs |
| `sqlalchemy` | ORM + connection pooling for Postgres-backed ingestion/feature storage | `>=2.0` for the 2.x `select()`-style query API used throughout `astroml.db`/`astroml.ingestion` | 1.x-style `Query` API is not used anywhere in this codebase — don't downgrade |
| `alembic` | Schema migrations for the above | `>=1.12` | Paired 1:1 with the sqlalchemy floor |
| `psycopg2-binary` | Postgres DB driver | `>=2.9` | Binary wheel intentionally, to avoid requiring a C toolchain + libpq-dev on every dev machine; swap for `psycopg2` (source build) only if you specifically need it |
| `redis` | Cache backend (`astroml.cache`) + Celery broker | `>=5.0.0` | `celery[redis]` below depends on this |
| `celery[redis]` / `flower` | Background task queue + its monitoring UI | `>=5.3` / `>=2.0` | Only exercised by async feature-computation jobs, not the synchronous ingestion path |
| `networkx` | In-memory graph algorithms (centrality, connectivity) used alongside the custom `TransactionGraph`/`snapshot.py` structures | `>=3.2.0` | Not used for large-graph storage — see `docs/scaling-optimization.md` for why snapshot-building avoids materialising a `networkx.Graph` for the full history |
| `pyarrow` / `fastparquet` | Columnar storage for the feature store | `>=14.0.0` / `>=2024.2.0` | Two parquet engines are kept because `pandas.to_parquet(engine=...)` callers pick between them depending on schema needs (fastparquet handles some legacy schemas pyarrow doesn't) |
| `stellar-sdk` | Horizon API client for ledger/effect/operation ingestion | `>=9.0.0` | Direct dependency of everything under `astroml/ingestion/` |
| `hydra-core` / `omegaconf` | Structured config composition (`config/`, `configs/`) | `>=1.3.0` / `>=2.3.0` | In `requirements-minimal.txt` because config loading shouldn't require the ML stack |
| `starlette` | Transitive — pulled in by mlflow/fastapi-style deps | `>=1.0.1` | See "Transitive dependencies of concern" below |
| `ruff` | Fast linter, the primary one CI runs (`ruff check .`) | `>=0.4.0` | `flake8` is still listed and still runs via `make lint` — the two haven't been consolidated yet, ruff isn't a drop-in replacement for the flake8 plugin set currently configured, so both stay until that migration is deliberately done |

### Transitive dependencies of concern

- **`starlette>=1.0.1`** — pinned directly in `requirements.txt` even
  though nothing in this repo imports `starlette` itself. `mlflow` and
  other FastAPI-style dependencies pull an older `starlette` in
  transitively; without this floor, `pip-audit` flags **PYSEC-2026-161**
  (a Host-header path-injection issue) on the resolver's default pick.
  Re-run `pip-audit -r requirements.txt` after any dependency bump to
  confirm this (and any newly-introduced transitive CVE) is still covered.
- **`python-multipart`** — see the "Known caveat" note above:
  `api/requirements.txt` currently pins a version older than what
  `strawberry-graphql[fastapi]` requires. Flagged, not fixed, here.

### Upgrading a major dependency

Worked example: `torch` 2.x → 3.x (hypothetical — no 3.x exists at time of
writing, but the same steps apply to any major bump of a core numeric
dependency such as `numpy` or `pandas`).

1. **Check the compatibility matrix first**, not just `torch`'s own release
   notes: `torch-geometric` and `pytorch-lightning` both pin against
   specific `torch` major versions. Bumping `torch` alone without also
   moving those two is the most common source of import-time breakage
   (see the `torch-geometric` row above).
2. **Stage the bump in `requirements-train.txt` first**, not the full
   `requirements.txt` — it's the smallest file that actually exercises
   `torch`, so it's the fastest way to get a signal.
3. **Re-run the memory/performance benchmarks that assume the current
   numeric stack**: `tests/test_graph_memory_profile.py` (issue #546) and
   `tests/test_ingestion_service_streaming.py` (issue #547) don't import
   `torch` directly, but anything under `astroml.benchmarking` or
   `astroml.training` that does should be re-profiled — a major numeric
   library bump can shift both memory and wall-clock numbers enough to
   invalidate previously-documented budgets.
4. **Update the pin in all four places it appears** (`requirements.txt`,
   `requirements-cpu.txt`, `requirements-train.txt`, and the CUDA-specific
   `pip install torch ...` line in `Dockerfile`'s GPU training stage) —
   see "Pin policy" above.
5. **Roll out via the GPU CI runner before touching CPU-only environments**
   — GPU-specific breakage (CUDA kernel API changes) tends to surface
   first there, before it would show up in `requirements-cpu.txt`/
   `requirements-minimal.txt` users.

### Dependency tree (issue #562)

```bash
make dependency-tree
# or directly:
pipdeptree --warn silence
```

`pipdeptree` is installed via `requirements-dev.txt`. Use it to check what a
given top-level pin actually pulls in before bumping it, or to spot two
packages fighting over the same transitive dependency (the
`python-multipart` conflict above is exactly the kind of thing
`pipdeptree --warn silence` — or `pip install --dry-run` — surfaces).
