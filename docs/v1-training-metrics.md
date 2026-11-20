# V1 Training Metrics

What the v1 training path records and reports: loss for every epoch, link prediction metrics
and exactly when they are undefined, the threshold behind precision and recall, and what
happens when the loss stops being a number. The code is in
`astroml/training/graph_objectives.py`, `astroml/training/train_gcn.py` and
`astroml/tracking/training_report.py`.

## Loss every epoch

`train_gcn.train()` records both losses for **every** epoch in the `TrainingReport`:

| Field | Meaning |
|---|---|
| `train["loss"]` | Loss on the training nodes, from the forward pass the optimiser used (dropout on). |
| `val["loss"]` | Loss on the validation nodes from a separate evaluation pass (dropout off). |

The full metric set (`accuracy`, `macro_f1`, and so on) is computed on the `eval_every`
cadence and on the last epoch. On those epochs `val` holds all of it, including `loss`; on
the others `val` holds only `loss`. The validation pass adds one forward pass per epoch.

```python
report = train(epochs=200, eval_every=20)
for record in report.records:
    print(record.epoch, record.train["loss"], record.val["loss"])
```

## Link prediction metrics

`LinkPredictionObjective` reads scores as logits. An edge is **predicted** when its
probability is strictly greater than `threshold`, which defaults to `0.5` (a logit above 0).
`threshold` must be strictly between 0 and 1.

| Metric | Definition | When it is undefined |
|---|---|---|
| `auc` | ROC-AUC (rank identity, ties averaged) | Fewer than one positive **and** one negative edge: reported as `0.5`, with `auc_defined = 0.0` |
| `auc_defined` | `1.0` when `auc` is meaningful, `0.0` when it is not | never |
| `average_precision` | Area under the precision-recall curve as a cumulative pass over ranked scores | No positive edges: `0.0` |
| `accuracy` | Fraction of edges where predicted equals actual | no edges: `0.0` |
| `precision` | true positives / predicted positives | Nothing predicted positive: `0.0` |
| `recall` | true positives / actual positives | No positive edges: `0.0` |

- `auc`, `auc_defined` and `average_precision` rank by score and ignore `threshold`.
  `accuracy`, `precision` and `recall` all use it.
- The reported `0.5` for an undefined AUC is kept for compatibility. Use `auc_defined` to tell
  it apart from a model that scored 0.5 on both classes.
- `precision` and `recall` follow the same convention as `average_precision` and macro-F1 in
  this module: undefined means `0.0`, matching scikit-learn's `zero_division=0`.

### Choosing the best epoch

`TrainingReport` picks the best epoch by the objective's monitored metric (`auc` for link
prediction). An epoch whose `<monitor>_defined` value is `0.0` is **skipped**, so a validation
split with one class cannot win by reporting `0.5`. It does not fall back to the training
value either. If every epoch is undefined, `best` is `None`. A metric with no `_defined` key is
treated as defined, so older records and other objectives behave as before.

```python
objective = LinkPredictionObjective(threshold=0.8)
metrics = objective.evaluate(scores, labels)
metrics["precision"], metrics["recall"]
```

## Non-finite loss

A loss that is NaN or infinite means the run has diverged, for example a learning rate that is
too high, an overflow, or bad features. Carrying on would corrupt the weights with the next
optimiser step and write NaN into the report, which is not valid JSON.

`ensure_finite_loss(loss, epoch=..., phase=...)` raises `NonFiniteLossError` for NaN, `+inf` or
`-inf`, and otherwise returns the loss as a float.

- `train_gcn.train()` checks the training loss **before** `backward()` and `optimizer.step()`,
  and the validation loss every epoch. The run stops at the first bad value.
- `train_link_prediction` checks each step's loss. Its `train_step` has already stepped the
  optimiser by the time it returns, so there the run stops but that one step cannot be
  prevented.
- `NonFiniteLossError` is a `ModelError` (see [V1 Errors](v1-errors.md)) with the code
  `NON_FINITE_LOSS`. `error.context` has `epoch`, `phase` (`"train"` or `"val"`) and `value`
  for logs. `to_public_dict()` shows only the code and a generic message.
- A `TrainingReport` you pass in keeps every epoch recorded before the failure, and contains no
  NaN:

```python
report = TrainingReport(NodeClassificationObjective(log_input=True))
try:
    train(report=report)
except NonFiniteLossError as error:
    print("diverged at epoch", error.context["epoch"], "after", len(report), "good epochs")
    report.persist(registry, version_id)  # the finite history is still valid JSON
```

## Tests

`tests/training/test_training_metrics_v1.py` covers the metric definitions, the threshold and
the single-class cases, with precision and recall cross-checked against scikit-learn.
`tests/training/test_train_gcn_loop.py` runs the real loop on a small in-memory graph (no
dataset download, network or metrics server) to check the per-epoch records and that a NaN loss
stops training before the optimiser steps.
