from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam

from astroml.models.link_prediction import LinkPredictor
from astroml.tasks.link_prediction_task import LinkPredictionTask
from astroml.training.synthetic_dataset import generate_synthetic_ledger_dataset


def train_link_prediction(
    num_accounts: int = 20,
    num_edges: int = 200,
    num_ledgers: int = 20,
    n_future: int = 3,
    context_ledgers: int = 5,
    neg_sampling_ratio: float = 1.0,
    epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    metadata_path: str | Path | None = None,
) -> None:
    dataset = generate_synthetic_ledger_dataset(
        num_accounts=num_accounts,
        num_edges=num_edges,
        num_ledgers=num_ledgers,
    )
    if metadata_path is not None:
        dataset.save_metadata(metadata_path)

    task = LinkPredictionTask(
        dataset.edges,
        n_future=n_future,
        context_ledgers=context_ledgers,
        neg_sampling_ratio=neg_sampling_ratio,
        device=device,
        seed=42,
    )
    splits = task.build_splits()
    if not splits:
        raise RuntimeError("No training splits could be built from the synthetic ledger data.")

    model = LinkPredictor(
        input_dim=32,
        hidden_dims=[64],
        embedding_dim=32,
        dropout=0.5,
        decoder="dot",
    ).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)

    print(f"Built {len(splits)} self-supervised training splits.")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for split in splits:
            node_features = dataset.features_for(split.node_index).to(device)
            loss = task.train_step(model, optimizer, split, node_features)
            total_loss += loss

        avg_loss = total_loss / len(splits)
        print(f"Epoch {epoch:3d}/{epochs} | Avg Loss: {avg_loss:.4f}")

    eval_split = splits[-1]
    node_features = dataset.features_for(eval_split.node_index).to(device)
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
    parser.add_argument("--metadata-path", type=Path)
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
        metadata_path=args.metadata_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
