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
| Randomness       | **not seeded** by `train()`; see "Not guaranteed"   |

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

* **Bit-for-bit reproducibility.** `train()` does not set a random seed and
  uses GPU kernels when available, so two runs differ slightly. Compare
  models by repeated runs, and record the config and input identifiers with
  every run (`run.json`) so differences are explainable.
* Absolute accuracy. Cora numbers do not transfer to ledger graphs.

## Where older docs disagree

* `configs/model/gcn.yaml` lists `hidden_dims: [64, 32]` and variants such as
  `batch_norm` and `residual`. `GCN` takes one `hidden_dim` and has no such
  options, and `train()` does not read this file, so it does **not** describe
  the baseline.
* `docs/tutorial-gcn.md` calls `astroml.models.gcn.train_gcn`, which does not
  exist; use `astroml.training.train_gcn.train`.
