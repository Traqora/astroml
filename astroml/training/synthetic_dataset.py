"""Deterministic offline datasets for temporal graph tests and baselines."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from astroml.features.graph.snapshot import (
    Edge,
    SnapshotLineage,
    SnapshotWindow,
    snapshot_lineage,
)


@dataclass(frozen=True)
class SyntheticLedgerDataset:
    """Generated edges, aligned node features, and serializable lineage metadata."""

    edges: list[Edge]
    node_features: torch.Tensor
    account_index: dict[str, int]
    lineage: SnapshotLineage

    @property
    def metadata(self) -> dict[str, object]:
        """Return counts, time bounds, seed, and feature width as JSON data."""
        return {
            **self.lineage.to_dict(),
            "feature_dim": int(self.node_features.shape[1]),
        }

    def features_for(self, node_index: dict[str, int]) -> torch.Tensor:
        """Return features ordered by a split's contiguous local node indices."""
        accounts = sorted(node_index, key=node_index.__getitem__)
        if [node_index[account] for account in accounts] != list(range(len(accounts))):
            raise ValueError("node_index values must be contiguous from zero")
        try:
            rows = [self.account_index[account] for account in accounts]
        except KeyError as exc:
            raise ValueError(f"account {exc.args[0]!r} is not in this dataset") from exc
        if not rows:
            return self.node_features.new_empty((0, self.node_features.shape[1]))
        return self.node_features[rows]

    def save_metadata(self, path: str | Path) -> Path:
        """Persist lineage metadata as a local JSON sidecar, without DVC or a database."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path


def generate_synthetic_ledger_dataset(
    *,
    num_accounts: int = 20,
    num_edges: int = 200,
    num_ledgers: int = 20,
    feature_dim: int = 32,
    seed: int = 42,
) -> SyntheticLedgerDataset:
    """Build a seeded, in-memory temporal graph fixture without external services.

    Edges are directed, have no self-loops, and receive integer timestamps in
    ``[1, num_ledgers]``. Empty datasets are supported; non-empty edge sets need
    at least two accounts. Features are seeded standard-normal values and are
    not intended to represent real Stellar account attributes.
    """
    if num_accounts < 0:
        raise ValueError("num_accounts must be non-negative")
    if num_edges < 0:
        raise ValueError("num_edges must be non-negative")
    if num_ledgers < 1:
        raise ValueError("num_ledgers must be at least one")
    if feature_dim < 1:
        raise ValueError("feature_dim must be at least one")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if num_edges and num_accounts < 2:
        raise ValueError("num_accounts must be at least two when generating edges")

    rng = random.Random(seed)
    accounts = [f"account_{index}" for index in range(num_accounts)]
    edges: list[Edge] = []
    for _ in range(num_edges):
        src_index = rng.randrange(num_accounts)
        dst_index = rng.randrange(num_accounts - 1)
        if dst_index >= src_index:
            dst_index += 1
        edges.append(
            Edge(
                src=accounts[src_index],
                dst=accounts[dst_index],
                timestamp=rng.randint(1, num_ledgers),
            )
        )
    edges.sort(key=lambda edge: edge.timestamp)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    node_features = torch.randn(num_accounts, feature_dim, generator=generator)
    account_index = {account: index for index, account in enumerate(accounts)}
    window = SnapshotWindow(
        index=0,
        start=datetime.fromtimestamp(1, timezone.utc),
        end=datetime.fromtimestamp(num_ledgers, timezone.utc),
        edges=edges,
        nodes=set(accounts),
    )
    lineage = snapshot_lineage(
        window,
        snapshot_id=f"synthetic-ledger:{seed}:{num_accounts}:{num_edges}:{num_ledgers}",
        source="synthetic_ledger_dataset",
        seed=seed,
    )
    return SyntheticLedgerDataset(
        edges=edges,
        node_features=node_features,
        account_index=account_index,
        lineage=lineage,
    )
