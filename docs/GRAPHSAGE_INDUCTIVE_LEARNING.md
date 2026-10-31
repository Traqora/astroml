# GraphSAGE inductive learning (#843)

Scores a Stellar account that didn't exist when the model was trained, by
producing an embedding for it inductively, without retraining the encoder
on the full graph. Built per
`docs/superpowers/plans/2026-03-26-graphsage-inductive-learning.md` and
`docs/superpowers/specs/2026-03-26-graphsage-inductive-learning-design.md`.

## Why inductive, not transductive

A transductive GNN (plain GCN trained on a fixed adjacency matrix) can
only produce embeddings for the exact nodes it saw during training. A
brand-new Stellar account created after training has no embedding at all
until the whole model is retrained on the updated graph.

GraphSAGE avoids this: the encoder learns a function of a node's sampled
*neighborhood features*, not a per-node lookup table. Any new node with
edges into the existing graph can be embedded by sampling its neighbors
and running the same forward pass, with no retraining.

## Components

| Module | Responsibility |
|---|---|
| `astroml/features/gnn/sampler.py` (`MultiHopSampler`) | K-hop neighbor sampling for a batch of target nodes. |
| `astroml/models/sage_encoder.py` (`InductiveSAGEEncoder`) | Multi-layer `SAGEConv` stack that consumes the sampler's per-hop adjacency and produces embeddings. |
| `astroml/training/train_sage.py` | Unsupervised reconstruction-loss training (`build_reconstruction_target`, `train_epoch`), no labels required. |
| `astroml/pipeline/inductive.py` (`InductiveGraphSAGE`) | Orchestrator: raw transaction edges in, per-node embeddings out. Wraps feature computation, graph construction, sampling, and the encoder forward pass into one call. |
| `astroml/pipeline/scoring.py` (`InductiveAnomalyScorer`) | Feeds `InductiveGraphSAGE` embeddings into a trained `DeepSVDD` model to score new/unseen accounts for anomalies. Results are cached (`@cached_prediction`, 300s TTL). |

## Training

`train_sage.py` is a library of training-loop functions, not a CLI script;
call `train_epoch` directly (see `tests/test_inductive_generalization.py`
for a complete worked example):

```python
import torch
from astroml.features.gnn.sampler import MultiHopSampler
from astroml.models.sage_encoder import InductiveSAGEEncoder
from astroml.training.train_sage import train_epoch

encoder = InductiveSAGEEncoder(
    input_dim=8, hidden_dim=16, output_dim=8, num_layers=2,
    dropout=0.0, aggregator="mean",
)
sampler = MultiHopSampler(edge_index, num_nodes, fanout=[25, 10])
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

for _ in range(num_epochs):
    train_epoch(encoder, sampler, features, edge_index, train_nodes, optimizer, batch_size=512)
```

`configs/model/sage.yaml` and `configs/sampling/default.yaml` hold the
encoder/sampling hyperparameters; `configs/experiment/inductive.yaml`
composes both plus the reconstruction loss and `DeepSVDD` scoring mode
(`embeddings_only` or `concatenated`) for a full experiment run through
the repo's Hydra config system.

## Scoring a new (unseen) account

```python
from astroml.pipeline.inductive import InductiveGraphSAGE
from astroml.pipeline.scoring import InductiveAnomalyScorer
from astroml.models.deep_svdd import DeepSVDD

pipeline = InductiveGraphSAGE(encoder=trained_encoder, fanout=[25, 10])
scorer = InductiveAnomalyScorer(pipeline=pipeline, svdd=trained_svdd, mode="embeddings_only")

scores = scorer.score_new_accounts(
    edges=recent_transaction_edges,   # list of {src, dst, amount, timestamp, asset}
    account_ids=["GABC...", "GXYZ..."],  # accounts to score, including ones never trained on
    ref_time=current_ledger_timestamp,
)
# {"GABC...": 0.42, "GXYZ...": 1.87}  -- higher is more anomalous
```

`account_ids` that don't appear anywhere in `edges` get a score of
`float("inf")` (no graph context to embed from) rather than being silently
dropped. Check for that sentinel before treating a result as a real score.

## Testing

- `tests/test_multi_hop_sampler.py`, `tests/test_sage_encoder.py`, unit
  tests for the sampler and encoder in isolation.
- `tests/test_inductive_pipeline.py`, `tests/test_inductive_scoring.py`,
  integration tests for `InductiveGraphSAGE`/`InductiveAnomalyScorer`.
- `tests/test_inductive_training.py`, training loop integration.
- `tests/test_inductive_generalization.py`, the end-to-end property this
  whole feature exists for: train on one subgraph, embed nodes from a
  disjoint, unseen subgraph, and verify the embeddings are non-zero and
  vary across nodes (i.e. the encoder is actually using neighborhood
  structure, not just memorizing training-set node identities).
