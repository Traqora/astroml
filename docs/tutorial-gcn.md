# Tutorial: Build a Snapshot and Train a Baseline GCN

AstroML allows you to ingest blockchain data, build temporal graphs, and train models. This tutorial walks through building a Graph Convolutional Network (GCN) model on a snapshot of Ledger data.

## Prerequisites
Make sure you have installed AstroML and its dependencies:
```bash
pip install astroml[train]
```

## Step 1: Backfill a Ledger Range

We start by fetching historical data from the network. 

```python
from astroml.preprocessing.ledger_backfill import run_backfill

# Backfill a small range for the tutorial
run_backfill(start_ledger=1000, end_ledger=2000, output_dir="data/raw")
```

## Step 2: Build a Snapshot

Once the data is ingested, we convert it into a temporal snapshot suitable for GCNs.

```python
from astroml.deployment.state_snapshot import build_snapshot

snapshot_path = build_snapshot(
    ledger_range=(1000, 2000), 
    input_dir="data/raw", 
    output_path="data/snapshot.pt"
)
print(f"Snapshot saved to {snapshot_path}")
```

## Step 3: Train the Baseline GCN

Now we train the baseline GCN model using the snapshot.

```python
from astroml.models.gcn import train_gcn

model, metrics = train_gcn(
    snapshot_path="data/snapshot.pt",
    epochs=50,
    learning_rate=0.01,
    hidden_dims=[64, 32]
)
print(f"Training completed. Final accuracy: {metrics['accuracy']:.2f}")
```

## FAQ

**Q: What if I run out of memory (OOM)?**
A: Try reducing the batch size in `train_gcn` or use a smaller ledger range when building the snapshot.

**Q: Where can I find the trained model?**
A: By default, it is saved to `outputs/models/baseline_gcn.pt`.
