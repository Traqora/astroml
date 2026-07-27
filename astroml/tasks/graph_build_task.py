"""Celery task: build transaction graph from a ledger range — issue #296.

In production this task would load transaction edges from the database,
construct a NetworkX / PyG graph, and persist it to the feature store.
The current implementation is a placeholder that simulates the workload
with a short sleep and returns node/edge count metadata.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from astroml.tasks.celery_app import app

logger = logging.getLogger(__name__)


@app.task(
    name="astroml.tasks.graph_build_task.build_graph",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 5},
)
def build_graph(
    self,
    account_ids: list[str],
    ledger_from: int,
    ledger_to: int,
) -> dict[str, Any]:
    """Build a transaction graph for the given accounts and ledger range.

    Parameters
    ----------
    account_ids:
        Stellar account public keys to include as graph nodes.
    ledger_from:
        First ledger sequence number (inclusive) to pull transactions from.
    ledger_to:
        Last ledger sequence number (inclusive) to pull transactions from.

    Returns
    -------
    dict with keys:
        node_count  — number of unique accounts in the graph
        edge_count  — number of transaction edges
        ledger_from — echoed input
        ledger_to   — echoed input
    """
    logger.info(
        "build_graph started: accounts=%d ledger=[%d, %d]",
        len(account_ids),
        ledger_from,
        ledger_to,
    )

    # Simulate I/O-bound graph construction.
    time.sleep(0.05)

    # Placeholder: treat each account as a node, each adjacent pair as an edge.
    node_count = len(account_ids)
    edge_count = max(0, node_count - 1)

    result = {
        "node_count": node_count,
        "edge_count": edge_count,
        "ledger_from": ledger_from,
        "ledger_to": ledger_to,
    }
    logger.info("build_graph finished: %s", result)
    return result
