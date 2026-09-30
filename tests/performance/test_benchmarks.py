"""Performance regression tests for critical operations.

This module benchmarks key operations to ensure performance does not degrade
beyond acceptable thresholds. Results are stored as CI artifacts for comparison.

Thresholds:
- Graph building (10k edges): <5s
- Feature computation (1000 nodes): <2s
- Database query (1000 records): <500ms
- Model inference (batch of 100): <100ms
"""

from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np
import pytest

from astroml.features.frequency import compute_frequency_metrics
from astroml.features.graph_validation import check_isolated_nodes
from astroml.graph_utils import graph_to_pyg_data


@pytest.mark.benchmark(group="graph-building")
def test_graph_building_10k_edges(benchmark):
    """Benchmark graph building with 10k edges.

    Threshold: should complete in <5s
    """
    # Generate 10k edges
    num_nodes = 2000
    num_edges = 10000

    # Create random node features
    node_features = np.random.rand(num_nodes, 32).astype(np.float32)

    # Create random edges
    edge_index = np.random.randint(0, num_nodes, size=(2, num_edges), dtype=np.int64)

    # Benchmark graph conversion
    result = benchmark(
        graph_to_pyg_data,
        node_features=node_features,
        edge_index=edge_index,
    )

    assert result.num_nodes == num_nodes
    assert result.edge_index.shape[1] == num_edges


@pytest.mark.benchmark(group="feature-computation")
def test_feature_computation_1000_nodes(benchmark):
    """Benchmark feature computation for 1000 nodes.

    Threshold: should complete in <2s
    """
    import pandas as pd

    num_accounts = 1000
    tx_per_account = 5

    # `compute_frequency_metrics` groups the transaction frame by `account` and
    # buckets `timestamp` into daily counts, so those are the columns it needs.
    days = pd.date_range("2024-01-01", periods=30, freq="D")
    transactions = [
        {
            "account": f"node_{i}",
            "timestamp": days[(i + j) % len(days)],
            "amount": float(np.random.rand() * 1000),
        }
        for i in range(num_accounts)
        for j in range(tx_per_account)
    ]
    transactions_df = pd.DataFrame(transactions)

    # Benchmark feature computation
    result = benchmark(compute_frequency_metrics, transactions_df)

    assert len(result) == num_accounts


@pytest.mark.benchmark(group="database-query")
def test_database_query_1000_records(benchmark, db_session):
    """Benchmark database query for 1000 records.

    Threshold: should complete in <500ms
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import select

    from astroml.db.schema import NormalizedTransaction

    # Create test data
    now = datetime.now(timezone.utc)
    transactions = []
    for i in range(1000):
        tx = NormalizedTransaction(
            sender=f"account_{i % 100}",
            receiver=f"account_{(i + 1) % 100}",
            amount=float(np.random.rand() * 1000),
            timestamp=now - timedelta(days=i % 30),
            ledger_sequence=i,
        )
        db_session.add(tx)
        transactions.append(tx)

    db_session.commit()

    # Benchmark query
    def query_1000_records():
        result = db_session.execute(
            select(NormalizedTransaction)
            .where(NormalizedTransaction.timestamp >= now - timedelta(days=30))
            .limit(1000)
        )
        return result.scalars().all()

    result = benchmark(query_1000_records)
    assert len(result) <= 1000

    # Cleanup
    for tx in transactions:
        db_session.delete(tx)
    db_session.commit()


@pytest.mark.benchmark(group="model-inference")
def test_model_inference_batch_100(benchmark):
    """Benchmark model inference for batch of 100.

    Threshold: should complete in <100ms
    """
    import torch
    from torch_geometric.data import Data

    # Create a simple GCN model for benchmarking
    num_nodes = 100
    num_features = 32
    hidden_dim = 64

    # Create test data
    node_features = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 200))

    data = Data(x=node_features, edge_index=edge_index)

    # Simple linear layer as proxy for model inference
    model = torch.nn.Linear(num_features, hidden_dim)
    model.eval()

    def inference():
        with torch.no_grad():
            return model(data.x)

    result = benchmark(inference)
    assert result.shape == (num_nodes, hidden_dim)


@pytest.mark.benchmark(group="graph-validation")
def test_graph_validation_large_graph(benchmark):
    """Benchmark graph validation on large graph (10k nodes, 100k edges).

    Threshold: should complete in <5s
    """
    import pandas as pd

    # Create large graph
    num_nodes = 10000
    num_edges = 100000

    edges = []
    for i in range(num_edges):
        edges.append(
            {
                "source": f"node_{np.random.randint(0, num_nodes)}",
                "target": f"node_{np.random.randint(0, num_nodes)}",
            }
        )

    edges_df = pd.DataFrame(edges)

    # Benchmark validation
    result = benchmark(
        check_isolated_nodes,
        edges=edges_df,
        source_col="source",
        target_col="target",
        allow_isolated=True,
    )

    connected, isolated = result
    assert isinstance(connected, set)
    assert isinstance(isolated, set)


@pytest.fixture
def db_session():
    """Create a test database session.

    Skips when the configured database is unreachable: the performance job runs
    with no postgres service, so there is nothing to time and a query benchmark
    that never issues a query says nothing about a regression.
    """
    from sqlalchemy import text

    from astroml.db.session import get_session

    try:
        session = get_session()
        session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"no reachable database for this runner: {exc}")
    try:
        yield session
    finally:
        session.close()
