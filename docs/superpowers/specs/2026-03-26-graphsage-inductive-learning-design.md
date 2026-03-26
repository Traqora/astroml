# GraphSAGE for Inductive Learning on New Accounts

**Issue:** #70
**Date:** 2026-03-26
**Approach:** Standalone inductive embedding module (Approach A), designed to evolve into full pipeline replacement (Approach B)

## Goal

Allow the model to generalize to Stellar accounts not seen during training. New accounts appear constantly on the network and need embeddings immediately for anomaly scoring via the existing Deep SVDD pipeline.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary task | General-purpose embedding generation | Feeds into Deep SVDD and any future downstream task without coupling |
| Inference modes | Both batch and on-demand | Core embedding function is stateless; batch vs on-demand is a caller concern |
| Neighborhood config | 2-hop, fanout [25, 10] as defaults; fully configurable via Hydra | Sensible defaults with sweep capability via existing hyperparameter_search preset |
| Feature initialization | Derive from triggering transactions, zero-pad remainder | Even one transaction yields degree, volume, asset type, timestamp; existing `compute_node_features()` handles this naturally |

## Architecture Overview

```
Graph Snapshot + Target Node List
        |
        v
  MultiHopSampler (K-hop neighbor sampling)
        |
        v
  compute_node_features() (existing pipeline)
        |
        v
  InductiveSAGEEncoder (multi-layer SAGEConv)
        |
        v
  Node Embeddings [N, output_dim]
        |
        v
  InductiveAnomalyScorer (Deep SVDD integration)
        |
        v
  Anomaly Scores per account
```

## Components

### 1. InductiveSAGEEncoder

**File:** `astroml/models/sage_encoder.py`

Multi-layer GraphSAGE encoder that stacks existing `SAGEConv` layers with ReLU and dropout between them.

```python
class InductiveSAGEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, aggregator)
    def forward(self, x, adjs) -> Tensor  # [N, output_dim]
```

- `adjs` is a list of `(edge_index, size)` tuples, one per layer, produced by `MultiHopSampler`
- Processes layers bottom-up: largest subgraph first, progressively narrowing to target nodes
- Each layer uses `SAGEConv` from `astroml/features/gnn/sage.py`
- `input_dim` = 8 (from `compute_node_features`: in_degree, out_degree, total_received, total_sent, account_age, first_seen, unique_asset_count, asset_entropy)
- Designed as a reusable encoder that can later replace GCN in `train.py` (Approach B evolution)

### 2. MultiHopSampler

**File:** `astroml/features/gnn/sampler.py`

Extends the existing `sample_neighbors()` to support multi-hop sampling with configurable fanout per layer.

```python
class MultiHopSampler:
    def __init__(self, edge_index, num_nodes, fanout: List[int])
    def sample(self, target_nodes) -> Tuple[List[Tuple[Tensor, Tuple[int, int]]], Tensor]
        # Returns (adjs, all_node_ids)
```

- Walks outward from target nodes: hop-1 samples `fanout[0]` neighbors, hop-2 samples `fanout[1]` from those
- Reuses existing `sample_neighbors()` internally for each hop
- Returns the `adjs` list consumed by `InductiveSAGEEncoder.forward()` plus the union of all touched node IDs (for feature matrix slicing)
- Number of hops is derived from `len(fanout)`

### 3. InductiveGraphSAGE (Pipeline Orchestrator)

**File:** `astroml/pipeline/inductive.py`

Ties graph snapshots, feature computation, sampling, and encoding into a single callable.

```python
class InductiveGraphSAGE:
    def __init__(self, encoder, sampler, device)

    def embed_nodes(self, edges, target_nodes, ref_time) -> Dict[str, Tensor]
        # 1. Compute node features via compute_node_features(edges)
        # 2. Build node-to-index mapping
        # 3. Sample K-hop neighborhoods via MultiHopSampler
        # 4. Slice feature matrix to sampled nodes
        # 5. Run InductiveSAGEEncoder
        # 6. Return {node_id: embedding} for target nodes only

    def embed_snapshot(self, edges, start_ts, end_ts, target_nodes) -> Dict[str, Tensor]
        # Calls window_snapshot() then embed_nodes()
```

- **Batch path:** call `embed_snapshot()` with a time window and list of new account IDs
- **On-demand path:** call `embed_nodes()` with edges from the triggering transaction(s) plus K-hop neighborhood edges from the current graph
- Stateless over graph structure: same trained encoder works on any snapshot

### 4. InductiveAnomalyScorer (Deep SVDD Integration)

**File:** `astroml/pipeline/scoring.py`

Connects inductive embeddings to the existing `DeepSVDD` model for anomaly scoring.

```python
class InductiveAnomalyScorer:
    def __init__(self, inductive_pipeline: InductiveGraphSAGE, svdd: DeepSVDD)

    def score_new_accounts(self, edges, account_ids, ref_time) -> Dict[str, float]
        # 1. Get embeddings from inductive_pipeline.embed_nodes()
        # 2. Feed embeddings through svdd.predict()
        # 3. Return {account_id: anomaly_score}
```

- No changes to `DeepSVDD` itself; it already accepts arbitrary feature vectors
- Two modes via config:
  - `embeddings_only`: feed GraphSAGE embeddings directly
  - `concatenated`: concatenate embeddings with raw node features (lets SVDD see both graph structure and local statistics)
- Training flow: train GraphSAGE encoder -> freeze -> generate embeddings for all training nodes -> train Deep SVDD on those embeddings

### 5. Training Script

**File:** `astroml/training/train_sage.py`

Hydra-driven training script following the pattern in `train.py` but using mini-batch neighbor sampling.

```python
def train_epoch(model, sampler, features, train_nodes, optimizer):
    # Shuffle train_nodes into mini-batches
    # For each batch: sample K-hop subgraph, slice features, forward, compute loss, step

def train_inductive(cfg: DictConfig):
    # 1. Load graph edges (from snapshot or dataset)
    # 2. Compute node features via compute_node_features()
    # 3. Build MultiHopSampler from edge_index
    # 4. Temporal split: train on accounts before time T, eval on accounts after T
    # 5. Train InductiveSAGEEncoder with early stopping
    # 6. Save encoder checkpoint
```

- **Temporal split:** train on accounts seen before time T, validate/test on accounts appearing after T. Directly measures inductive performance.
- **Loss function:** unsupervised reconstruction loss. For each node, the encoder produces an embedding; the loss is MSE between this embedding (projected through a linear decoder) and the mean of the node's immediate neighbors' raw features. This incentivizes the encoder to capture neighborhood structure without requiring labels.
- **Optional supervised head:** classification or SVDD head composable on top via Hydra config.

### 6. Hydra Configuration

New config files in the existing `configs/` structure:

```yaml
# configs/model/sage.yaml
model:
  _target_: astroml.models.sage_encoder.InductiveSAGEEncoder
  hidden_dim: 64
  output_dim: 32
  num_layers: 2
  dropout: 0.5
  aggregator: "mean"

# configs/sampling/default.yaml
sampling:
  fanout: [25, 10]
  batch_size: 512

# configs/experiment/inductive.yaml
defaults:
  - /model: sage
  - /sampling: default
  - /training: default

experiment:
  name: "inductive_sage"
  temporal_split_ratio: 0.7
  loss: "reconstruction"
  svdd_mode: "embeddings_only"
```

- `sampling/` is a new config group for neighbor sampling params
- All decisions (2-hop, 25/10, mean aggregator) are defaults, overridable via CLI
- Sweepable with `python train.py --multirun sampling.fanout=[25,10],[15,5] model.hidden_dim=32,64,128`

## Testing Strategy

All tests use small synthetic graphs (10-50 nodes) with no database or network dependencies.

### Unit Tests

- **`test_sage_encoder.py`** — output shapes, gradient flow, multi-layer stacking, dropout behavior
- **`test_multi_hop_sampler.py`** — correct hop counts, fanout limits, handles isolated nodes, deterministic with seed
- **`test_inductive_pipeline.py`** — `embed_nodes()` returns correct node IDs, embedding dimensions, handles single-transaction accounts

### Integration Tests

- **`test_inductive_training.py`** — full train loop runs for 2 epochs on synthetic graph without crashing, loss decreases
- **`test_inductive_scoring.py`** — `InductiveAnomalyScorer` produces scores for nodes not in training set, scores are finite floats

### Inductive Validation

- **`test_inductive_generalization.py`** — train on subgraph A, embed nodes from subgraph B sharing no nodes with A, verify embeddings are non-zero and vary across nodes. This is the core property issue #70 demands.

## File Summary

| Component | File | Purpose |
|-----------|------|---------|
| InductiveSAGEEncoder | `astroml/models/sage_encoder.py` | Multi-layer GraphSAGE encoder |
| MultiHopSampler | `astroml/features/gnn/sampler.py` | K-hop neighbor sampling |
| InductiveGraphSAGE | `astroml/pipeline/inductive.py` | Embedding orchestrator |
| InductiveAnomalyScorer | `astroml/pipeline/scoring.py` | SVDD integration for anomaly scoring |
| Training script | `astroml/training/train_sage.py` | Hydra-driven mini-batch training |
| Model config | `configs/model/sage.yaml` | Encoder hyperparameters |
| Sampling config | `configs/sampling/default.yaml` | Neighbor sampling params |
| Experiment config | `configs/experiment/inductive.yaml` | Inductive experiment preset |
| Tests | `tests/test_sage_encoder.py`, `tests/test_multi_hop_sampler.py`, `tests/test_inductive_pipeline.py`, `tests/test_inductive_training.py`, `tests/test_inductive_scoring.py`, `tests/test_inductive_generalization.py` | Unit + integration + validation |

## Evolution Path

This design (Approach A) is intentionally structured so that the `InductiveSAGEEncoder` can later replace the GCN in `train.py` (Approach B). When ready:

1. Swap `create_model()` in `train.py` to instantiate `InductiveSAGEEncoder`
2. Replace full-graph forward pass with mini-batch neighbor sampling
3. The embedding module and scoring pipeline remain unchanged

Embedding caching (Approach C) can be layered on if inference latency becomes a bottleneck, by storing embeddings in PostgreSQL and adding a staleness-based refresh job.
