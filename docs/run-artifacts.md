# V1 run artifacts

Every v1 run can be stored as one **self-contained directory** that records
what was run, on which inputs, and what came out. The API lives in
`astroml/artifacts/run_artifacts.py`. It uses only the Python standard
library and the local filesystem, so it works without torch, cloud storage,
MLflow or any other optional service.

```python
from astroml.artifacts.run_artifacts import write_run_artifacts, read_run_artifacts

result = write_run_artifacts(
    "outputs/runs",                      # root directory for all runs
    "gcn-cora-001",                      # unique run id
    config={"model": "gcn", "hidden_dim": 16, "lr": 0.01, "epochs": 200},
    inputs={"dataset": "Cora", "split": "public"},
    metrics={"val": {"accuracy": 0.81}, "test": {"accuracy": 0.79}},
)
print(result.path)                       # outputs/runs/gcn-cora-001

run = read_run_artifacts("outputs/runs", "gcn-cora-001")
run["metrics"]["metrics"]["test"]["accuracy"]
```

## Layout

```
outputs/runs/gcn-cora-001/
├── run.json       run id, created_at, config, inputs, fingerprint
└── metrics.json   machine-readable metrics
```

### `metrics.json` (issue #1216)

```json
{
  "fingerprint": "9f2c…",
  "metrics": {"test": {"accuracy": 0.79}, "val": {"accuracy": 0.81}},
  "run_id": "gcn-cora-001",
  "schema_version": 1
}
```

* Strict JSON: `NaN` and `Infinity` are **rejected** with
  `InvalidRunInputError` (nothing is written) instead of producing a file
  other parsers cannot read. Metric values must be JSON types.
* `schema_version` lets a consumer detect format changes;
  `read_run_artifacts` raises `RunArtifactError` for a version it does not
  support, a missing or corrupt file, or a file that belongs to another run id.
* `fingerprint` is the SHA-256 of the canonical JSON of `config` + `inputs`
  (independent of key order). Equal fingerprints mean the same settings on
  the same inputs.

## Reproducing a saved run (issue #1223)

This section explains how to reproduce a saved run. A v1 run artifact records
enough information to repeat a run's **configuration and inputs** from a local
directory. The `config` and `inputs` objects in `run.json` are the source of
truth. `metrics.json` records the result; it is not an input to the repeated
run.

### Steps

1. Read the run with `read_run_artifacts(root, run_id)`. The reader rejects a
   missing, corrupt, unsupported-schema, or mismatched-run-id directory.
2. Recompute the fingerprint from the saved `config` and `inputs` before using
   them. A mismatch means the artifact changed and must not be reproduced.
3. Run the same code with the saved settings, the same input contents, and the
   same dependency versions. Supply the same random seed when the training
   entry point accepts one.
4. Store the repeat under a **new run id** with `write_run_artifacts` and check
   that its fingerprint equals the saved fingerprint. `replayed_metrics` is the
   value returned by the repeated training and evaluation call.

```python
from astroml.artifacts.run_artifacts import (
    compute_fingerprint,
    read_run_artifacts,
    write_run_artifacts,
)

root = "outputs/runs"
saved = read_run_artifacts(root, "gcn-cora-001")
config = saved["run"]["config"]
inputs = saved["run"]["inputs"]
fingerprint = saved["run"]["fingerprint"]

if compute_fingerprint(config, inputs) != fingerprint:
    raise RuntimeError("saved config and inputs do not match the fingerprint")

repeat = write_run_artifacts(
    root,
    "gcn-cora-001-replay",
    config=config,
    inputs=inputs,
    metrics=replayed_metrics,
)
assert repeat.fingerprint == fingerprint
```

The saved `config` is plain JSON, not a command line, Hydra override file, or
source-code revision. Record any extra detail needed to repeat the run in
`config` or `inputs` (for example, a dataset hash or seed).

### Failure cases

* `RunArtifactError`: the saved directory is missing, corrupt, incomplete, has
  an unsupported `schema_version`, or belongs to another run id. Stop; there
  is no trustworthy configuration to reproduce.
* Fingerprint mismatch: the saved `config` or `inputs` no longer match
  `run.json`. Stop and obtain a valid artifact.
* `RunConflictError`: the chosen id already exists. `same_inputs is True`
  identifies a re-run with identical settings; `False` is an id collision.
  Use a new id unless replacing the saved run is intentional.
* `InvalidRunInputError`: the new id is unsafe or the supplied config, inputs,
  or metrics are not strict JSON.

### Boundaries

* Reproduction uses only the local filesystem and the Python standard library.
  It works without torch, MLflow, a database, cloud or object storage, Horizon,
  Redis, or any other optional service.
* The artifact contains configuration, input identifiers, and metrics. It does
  not contain raw dataset bytes, model weights, checkpoints, a dependency
  lockfile, or a source revision; preserve those separately when they matter.
* Equal fingerprints prove equal `config` and `inputs` only. They do not prove
  identical code, dependencies, hardware, randomness, or metrics.
* Bit-for-bit reproducibility is not a v1 guarantee: the baseline `train()`
  does not seed RNGs and uses a GPU when available (see
  [baseline-model-assumptions.md](baseline-model-assumptions.md)).

## Atomic writes (issue #1219)

All files are first written into a hidden staging directory
(`.<run_id>.<random>.staging`), each via a temp file that is flushed and
`fsync`ed and then moved with `os.replace`. Only when every file is complete
is the staging directory renamed to `<run_id>`.

* A reader sees the whole run or no run, never a partial one.
* If a write fails or the process is interrupted, no `<run_id>` directory is
  created. After a crash, leftover `.*.staging` directories can be cleared
  with `remove_stale_staging(root)` (call it only when no writer is active).
* `atomic_write_json(path, obj)` is available for single files; on failure
  the previous file content is kept.

Boundaries: the guarantee holds on a local filesystem where `rename` is
atomic within a directory (POSIX filesystems, NTFS). It does not extend to
network or object stores.

## Conflicting run ids (issue #1220)

A run id may be used once. Writing an id that already exists raises
`RunConflictError` **before anything is replaced**, and the existing run is
left untouched:

```python
try:
    write_run_artifacts(root, "gcn-cora-001", config=cfg, inputs=inp, metrics=m)
except RunConflictError as err:
    err.same_inputs            # True: same config+inputs (a re-run)
                               # False: different config/inputs (an id collision)
                               # None: existing run's fingerprint unreadable
    err.existing_fingerprint, err.new_fingerprint
```

* Choose a new run id, or pass `overwrite=True` to replace the run on
  purpose. The replacement is swap-based: if it fails part-way the previous
  run is restored.
* Two writers racing for the same new id: one wins, the other gets
  `RunConflictError`; the winner's files are never clobbered.
* Run ids are 1-128 characters from `A-Z a-z 0-9 . _ -`, starting with a
  letter or digit. Anything else (`../x`, `a/b`, empty, hidden names) raises
  `InvalidRunInputError`, so an id can never point outside the root.

## Errors

All errors derive from `AstroMLError` (see [v1-errors.md](v1-errors.md)):

| Class                  | Code                          | Meaning                                  |
| ---------------------- | ----------------------------- | ---------------------------------------- |
| `RunArtifactError`     | `RUN_ARTIFACT_ERROR`          | A run could not be read (missing, corrupt, wrong schema) |
| `InvalidRunInputError` | `RUN_ARTIFACT_INVALID_INPUT`  | Bad run id, non-JSON or non-finite value (also a `ValueError`) |
| `RunConflictError`     | `RUN_ID_CONFLICT`             | The run id already exists                |

The baseline model these artifacts describe is documented in
[baseline-model-assumptions.md](baseline-model-assumptions.md).
