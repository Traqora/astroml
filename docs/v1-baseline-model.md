# V1 Baseline Model

The v1 baseline is the 2-layer `GCN` in `astroml.models.gcn`, trained for node
classification. This page states what input it accepts, how a trained model is saved and
reloaded, what is recorded about the environment a run used, and how to check the whole path
on a machine with no GPU.

## Model input

The model takes node features `x` (`[N, F]`, floating point) and a COO `edge_index`
(`[2, E]`, integer). Check them once before a run:

```python
from astroml.models.graph_input import prepare_graph_input

prepared = prepare_graph_input(x, edge_index)
out = model(prepared.x, prepared.edge_index)
print(prepared.num_isolated, "of", prepared.num_nodes, "nodes are isolated")
```

`prepare_graph_input` raises `ValueError` naming the problem for any of these:

| Input | Result |
|---|---|
| `x` is not a 2-D floating-point tensor | `ValueError` |
| `x` holds `NaN` or infinity | `ValueError` naming the first bad node |
| `edge_index` is not shaped `[2, E]` | `ValueError` |
| a non-empty `edge_index` is not an integer tensor | `ValueError` |
| an index is below 0 or at/above `N` | `ValueError` with the min and max index |
| an empty `edge_index` of any dtype | accepted, returned as `torch.long` |

### Isolated nodes and graphs with no edges

Both are valid input, not errors. `GCN` adds a self-loop to every node, so a node with no
neighbours is represented by its own features alone: its output is the same as it would be in
a one-node graph. A graph with no edges at all runs for the same reason.

A node whose only edge is a self-loop has no neighbour, so it counts as isolated.

`torch.tensor([[], []])` is a float tensor, which the scatter kernel rejects with
`Expected dtype int32/int64 for index`. `GCN.forward` converts an empty edge list to
`torch.long`, so the usual way of writing "no edges" works.

## Checkpoints

Use `astroml.models.checkpoint` instead of saving a bare `state_dict`:

```python
from astroml.models.checkpoint import GCNConfig, load_gcn_checkpoint, save_gcn_checkpoint

save_gcn_checkpoint(model, "gcn.pt", metadata={"run_id": "2026-11-20-a"})

loaded = load_gcn_checkpoint("gcn.pt", expected=GCNConfig(input_dim=1433, hidden_dim=16, output_dim=7))
model = loaded.model            # weights loaded, in evaluation mode
loaded.metadata["environment"]  # the versions it was saved with
```

The file records the format name, a format version, the architecture and the weights. It holds
only tensors, numbers and strings, so it loads with `torch.load(..., weights_only=True)` and
never runs pickled code. `metadata` must be JSON-serialisable.

### Compatibility rules

`load_gcn_checkpoint` accepts a checkpoint only when all of these hold. Otherwise it raises
`CheckpointCompatibilityError`, which says which rule failed.

1. The file is an AstroML GCN checkpoint (its `format` matches). A bare `state_dict` is not.
2. Its format version is between `MIN_SUPPORTED_VERSION` and `CHECKPOINT_VERSION` (both 1
   today). A newer version is refused rather than guessed at.
3. Its `config` is complete: `input_dim`, `hidden_dim` and `output_dim` are positive
   integers and `dropout` is in `[0, 1)`.
4. If you pass `expected`, `input_dim`, `hidden_dim` and `output_dim` match it. `dropout` holds
   no weights, so a different value is allowed and the file's value is used.
5. Its weights have exactly the keys and shapes of a model built from its `config`. The error
   lists the missing keys, the unexpected keys and the keys with the wrong shape.

An unreadable file (corrupt, truncated, or containing anything `weights_only` refuses) also
raises `CheckpointCompatibilityError`. A path that does not exist raises `FileNotFoundError`.

`CheckpointCompatibilityError` is a `ModelError` (see [V1 Errors](v1-errors.md)) with the code
`CHECKPOINT_INCOMPATIBLE`. Its `to_public_dict()` carries only the code and a generic message,
so a path or reason never reaches an API client.

### Writes are atomic

`save_gcn_checkpoint` writes `<path>.tmp` and renames it over the target. If the write fails
part-way, the previous checkpoint is untouched and no `.tmp` file is left behind.

Changing the format means bumping `CHECKPOINT_VERSION`, keeping `MIN_SUPPORTED_VERSION` for as
long as older files must still load, and adding a test for the old version.

## Environment recorded with a run

Every `TrainingReport` stores the environment its run started in, under `environment` in
`to_dict()` and so in the registry's `training_report` entry. Checkpoints store the same
snapshot under `metadata["environment"]`.

```json
{
  "python": "3.14.0",
  "implementation": "CPython",
  "platform": "macOS-15.0-arm64-arm-64bit",
  "packages": {"astroml": "0.1.0", "torch": "2.15.0", "torch-geometric": "2.8.1", "numpy": "2.5.4"}
}
```

- Versions come from installed package metadata (`importlib.metadata`). No package is
  imported, so a missing or broken optional library cannot make the snapshot fail.
- A package that is not installed is recorded as `null`, which is different from leaving it out.
- The tracked packages are `astroml.utils.environment.TRACKED_PACKAGES`. Pass `packages=` to
  `collect_environment` to change them.
- `TrainingReport(environment=...)` records a mapping you supply, and `environment={}` records
  none. Reports persisted before this change have no `environment` key.

## Smoke test

`tests/models/test_gcn_smoke.py` runs the whole path on a small synthetic graph that includes
isolated nodes: validate the input, train 60 epochs with `NodeClassificationObjective`, record a
report, save and reload a checkpoint, and check the predictions match.

It needs no GPU, dataset download, network, metrics server or database, and takes a few
seconds:

```bash
pytest tests/models/test_gcn_smoke.py
```

It is skipped when `torch` or `torch_geometric` is not installed.
