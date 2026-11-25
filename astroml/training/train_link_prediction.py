from __future__ import annotations

import argparse
import random

import torch
from torch.optim import Adam

from astroml.features.graph.snapshot import Edge
from astroml.models.link_prediction import LinkPredictor
from astroml.tasks.link_prediction_task import LedgerSplit, LinkPredictionTask

FEATURE_DIM = 32


def _build_synthetic_ledger_edges(
    num_accounts: int, num_edges: int, num_ledgers: int
) -> list[Edge]:
    rng = random.Random(42)
    accounts = [f"account_{i}" for i in range(num_accounts)]
    edges: list[Edge] = []
    for i in range(num_edges):
        src = rng.choice(accounts)
        dst = rng.choice([a for a in accounts if a != src])
        timestamp = rng.randint(1, num_ledgers)
        edges.append(Edge(src=src, dst=dst, timestamp=timestamp))
    return sorted(edges, key=lambda e: e.timestamp)


def build_node_features(num_nodes: int, feature_dim: int = FEATURE_DIM) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(num_nodes, feature_dim)


def _partition_train_eval_splits(splits: list[LedgerSplit]) -> tuple[list[LedgerSplit], LedgerSplit]:
    if len(splits) < 2:
        raise RuntimeError("At least two temporal splits are required for held-out evaluation.")
    eval_split = splits[-1]
    eval_start = min(edge.timestamp for edge in eval_split.future_edges)
    training_splits = [
        split
        for split in splits[:-1]
        if max(edge.timestamp for edge in split.future_edges) < eval_start
    ]
    if not training_splits:
        raise RuntimeError("No training split ends before the held-out evaluation window.")
    return training_splits, eval_split


def train_link_prediction(
    num_accounts: int = 20,
    num_edges: int = 200,
    num_ledgers: int = 20,
    n_future: int = 3,
    context_ledgers: int = 5,
    neg_sampling_ratio: float = 1.0,
    epochs: int = 10,
    device: torch.device = torch.device("cpu"),
) -> None:
    edges = _build_synthetic_ledger_edges(
        num_accounts=num_accounts,
        num_edges=num_edges,
        num_ledgers=num_ledgers,
    )

    task = LinkPredictionTask(
        edges,
        n_future=n_future,
        context_ledgers=context_ledgers,
        neg_sampling_ratio=neg_sampling_ratio,
        device=device,
        seed=42,
    )
    splits = task.build_splits()
    if not splits:
        raise RuntimeError("No training splits could be built from the synthetic ledger data.")
    training_splits, eval_split = _partition_train_eval_splits(splits)

    torch.manual_seed(42)
    model = LinkPredictor(
        input_dim=FEATURE_DIM,
        hidden_dims=[64],
        embedding_dim=32,
        dropout=0.5,
        decoder="dot",
    ).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)

    print(
        f"Built {len(training_splits)} training splits and 1 held-out evaluation split."
    )
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for split in training_splits:
            node_features = build_node_features(split.num_nodes, FEATURE_DIM).to(device)
            loss = task.train_step(model, optimizer, split, node_features)
            total_loss += loss

        avg_loss = total_loss / len(training_splits)
        print(f"Epoch {epoch:3d}/{epochs} | Avg Loss: {avg_loss:.4f}")

    node_features = build_node_features(eval_split.num_nodes, FEATURE_DIM).to(device)
    metrics = task.evaluate(model, eval_split, node_features)
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a self-supervised link prediction model.")
    parser.add_argument("--num-accounts", type=int, default=20)
    parser.add_argument("--num-edges", type=int, default=200)
    parser.add_argument("--num-ledgers", type=int, default=20)
    parser.add_argument("--n-future", type=int, default=3)
    parser.add_argument("--context-ledgers", type=int, default=5)
    parser.add_argument("--neg-sampling-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_link_prediction(
        num_accounts=args.num_accounts,
        num_edges=args.num_edges,
        num_ledgers=args.num_ledgers,
        n_future=args.n_future,
        context_ledgers=args.context_ledgers,
        neg_sampling_ratio=args.neg_sampling_ratio,
        epochs=args.epochs,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
