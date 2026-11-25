# V1 baseline model: assumptions and supported boundaries

The v1 baseline is one small graph model with a fixed input and output
contract: a **2-layer GCN for transductive node classification**, trained
full-batch on the Cora citation graph. It exists as a reference point for
comparing other models, not as a production fraud detector.

The source of truth is the code, not this page:

| Piece                    | Where                                                       |
| ------------------------ | ----------------------------------------------------------- |
| Model                    | `astroml/models/gcn.py` (`GCN`)                             |
| Training loop            | `astroml/training/train_gcn.py` (`train`)                   |
| Loss, metrics, checks    | `astroml/training/graph_objectives.py`                      |

`tests/test_baseline_model_docs.py` checks the numbers below against the
source, so this page cannot silently drift.

## Model

`GCNConv(input_dim, hidden_dim) → ReLU → Dropout → GCNConv(hidden_dim, output_dim) → log_softmax(dim=1)`

| Setting         | Baseline value | Notes                                              |
| --------------- | -------------- | -------------------------------------------------- |
| Layers          | 2              | Fixed by the class; not configurable.              |
| `hidden_dim`    | 16             | Single hidden width.                               |
| `dropout`       | 0.5            | Active only in training mode.                      |
| Output          | log-probabilities | Shape `[N, C]`; each row sums to 1 in probability space. |

## Training protocol

| Setting          | Baseline value                              |
| ---------------- | ------------------------------------------- |
| Dataset          | Cora (`Planetoid`, public split)            |
| Feature transform| `NormalizeFeatures` (row-normalised)        |
| Optimiser        | Adam, learning rate `0.01`, weight decay `5e-4` |
| Epochs           | 200 (argument `epochs`)                     |
| Validation cadence | every 20 epochs and the last epoch (argument `eval_every`) |
| Loss             | negative log-likelihood (`NodeClassificationObjective(log_input=True)`) on the train mask |
| Reported metrics | `accuracy` and `macro_f1`, on validation during training and on the test mask at the end |
| Batching         | none: the whole graph is processed every step |
| Randomness       | seeded by `train()` with `seed=42` before dataset loading and model initialisation |

## Input contract

* `x`: float node features, shape `[N, F]`, with `F == input_dim`.
* `edge_index`: COO edge list, shape `[2, E]`, integer dtype (`int32`/`int64`),
  every id in `[0, N)`. A self-loop is added internally by `GCNConv`.
* `train_mask`, `val_mask`, `test_mask`: boolean, shape `[N]`, **pairwise
  disjoint**.
* `y`: integer class labels, shape `[N]`, with values in `[0, C)`.

`train()` runs `validate_edge_index` and `validate_masks` before the first
forward pass, so a wrong shape or dtype, an out-of-range or negative node id,
or overlapping masks raises `ValueError` up front instead of producing a
plausible-looking but meaningless metric.

## Deterministic setup (issue #1206)

`train(seed=42)` calls `astroml.training.seed.set_seed` before loading Cora or
constructing the GCN. The seed sets Python `random`, NumPy, PyTorch CPU and all
CUDA devices, the cuDNN deterministic flags and `PYTHONHASHSEED` for child
processes. It must be an integer in `[0, 2**32 - 1]`; anything else raises
`ValueError` before any generator is touched. Pass `seed=None` to keep the
caller's existing random state. Passing no argument keeps the documented v1
default of `42`.

The Hydra path (`python train.py`) uses the same utility through
`experiment.seed` or the top-level `--seed` flag, which overrides it. NumPy
and PyTorch are optional to the utility itself: when one is missing, the other
generators are still seeded. No database, network service or cloud artifact
store is involved.

## Early stopping (issue #1208)

`train()` accepts an `EarlyStoppingConfig` and uses the shared
`astroml.training.early_stopping.EarlyStopper`. The supported monitors are
`train_loss`, `train_accuracy`, `val_loss` and `val_accuracy`; the default is
`val_loss` with `mode="min"`, `patience=50` and `min_delta=1e-4`. Loss monitors
default to `min`, accuracy monitors to `max`, and an explicit `mode` overrides
the default.

The baseline evaluates every `eval_every` epochs and on the final epoch, so the
stopper checks the monitor on that cadence rather than on every optimizer step.
`val_loss` is computed from the same objective and validation mask as the
reported validation accuracy. A run stops after `patience` consecutive
evaluations without an improvement greater than `min_delta`; `patience=0`
disables stopping. The returned `TrainingReport` contains every epoch that ran,
including the stopping epoch, so a shortened run remains inspectable.

## Output contract

* Model: `[N, C]` log-probabilities (feed them to `nll_loss`, not
  `cross_entropy`; that is what `log_input=True` encodes).
* `train()` returns a `TrainingReport` holding per-epoch train/validation
  metrics and the best epoch. Persist the final numbers as machine-readable
  run artifacts with `astroml.artifacts.run_artifacts`
  (see [run-artifacts.md](run-artifacts.md)).

## Supported boundaries

Supported:

* **Transductive** node classification: the test nodes are part of the graph
  seen during training; only their labels are hidden.
* A single static graph that fits in memory, on CPU or one CUDA device.
* Single-label, multi-class targets.

Not supported (use a different model or extend the baseline first):

* Inductive prediction on nodes or graphs unseen at training time; see
  `docs/GRAPHSAGE_INDUCTIVE_LEARNING.md` for the inductive path.
* Edge features or edge weights, heterogeneous graphs, multi-label targets.
* Mini-batch / neighbour-sampled training, so very large graphs are out of
  scope.
* Temporal dynamics: each run sees one snapshot.
* Calibrated probabilities, class-imbalance handling (the loss is unweighted
  unless you pass your own objective), or any fraud-detection claim.

## Not guaranteed

* **Bit-for-bit reproducibility across hardware or library versions.** The
  seed makes reruns reproducible on the same device and library versions, but
  GPU kernels and BLAS implementations can still differ across machines.
  Record the config and input identifiers with every run (`run.json`) so
  differences are explainable.
* Absolute accuracy. Cora numbers do not transfer to ledger graphs.

## Where older docs disagree

* `configs/model/gcn.yaml` lists `hidden_dims: [64, 32]` and variants such as
  `batch_norm` and `residual`. `GCN` takes one `hidden_dim` and has no such
  options, and `train()` does not read this file, so it does **not** describe
  the baseline.
* `docs/tutorial-gcn.md` calls `astroml.models.gcn.train_gcn`, which does not
  exist; use `astroml.training.train_gcn.train`.
