"""AstroML specific tools that expose framework capabilities to agents.

These are thin, well-typed wrappers around existing AstroML functionality so
an agent can inspect transaction graphs and score accounts without importing
the heavy ML stack itself.  Graph tools are pure Python; feature tools use
``pandas`` (already a core dependency); model backed tools are created on
demand through :func:`anomaly_score_tool` so ``torch`` /
``torch-geometric`` are only imported when a caller actually has a trained
model.

The graph tools accept edges as JSON-friendly mappings with the keys used
throughout the repository (``src``, ``dst``, ``timestamp`` and optionally
``amount`` / ``asset``)::

    {"src": "GA...", "dst": "GB...", "timestamp": 1700000000, "amount": 42.0, "asset": "XLM"}
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from astroml.features.graph.snapshot import Edge, window_snapshot

from .tools import Tool, ToolError, ToolRegistry, tool_from_callable

logger = logging.getLogger(__name__)

#: Maximum number of entries returned for "top N" style answers.
DEFAULT_TOP_K = 5


def _coerce_edges(raw_edges: Iterable[Any]) -> List[Edge]:
    """Convert mapping/``Edge`` inputs into :class:`Edge` objects.

    Raises:
        ToolError: when an entry is not an edge, or ``src``/``dst``/``timestamp``
            are missing or malformed.
    """
    edges: List[Edge] = []
    for index, item in enumerate(raw_edges or []):
        if isinstance(item, Edge):
            edges.append(item)
            continue
        if not isinstance(item, Mapping):
            raise ToolError(
                f"edge #{index} must be an object or Edge, got {type(item).__name__}"
            )
        src = item.get("src")
        dst = item.get("dst")
        if src is None or dst is None:
            raise ToolError(f"edge #{index} is missing 'src' or 'dst'")
        try:
            timestamp = int(item.get("timestamp", 0) or 0)
        except (TypeError, ValueError):
            raise ToolError(
                f"edge #{index} has a non-numeric 'timestamp'"
            ) from None
        edges.append(Edge(src=str(src), dst=str(dst), timestamp=timestamp))
    return edges


def _to_float(value: Any) -> Optional[float]:
    """Best effort float conversion; returns ``None`` when not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def graph_overview(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise a transaction graph: size, assets, density, busiest accounts."""
    coerced = _coerce_edges(edges)
    nodes = set()
    in_degree: Counter = Counter()
    out_degree: Counter = Counter()
    self_loops = 0

    for edge in coerced:
        nodes.add(edge.src)
        nodes.add(edge.dst)
        out_degree[edge.src] += 1
        in_degree[edge.dst] += 1
        if edge.src == edge.dst:
            self_loops += 1

    assets = Counter(
        str(item.get("asset"))
        for item in edges or []
        if isinstance(item, Mapping) and item.get("asset")
    )

    num_nodes = len(nodes)
    num_edges = len(coerced)
    density = 0.0
    if num_nodes > 1:
        density = num_edges / (num_nodes * (num_nodes - 1))

    degree = {node: in_degree[node] + out_degree[node] for node in nodes}
    ranked = sorted(
        (
            {
                "account": node,
                "degree": total,
                "in_degree": in_degree[node],
                "out_degree": out_degree[node],
            }
            for node, total in degree.items()
        ),
        key=lambda item: (-item["degree"], item["account"]),
    )

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_self_loops": self_loops,
        "density": round(density, 6),
        "assets": dict(assets.most_common(10)),
        "top_accounts": ranked[:DEFAULT_TOP_K],
    }


def window_stats(
    edges: List[Dict[str, Any]],
    start_ts: int,
    end_ts: int,
) -> Dict[str, Any]:
    """Count nodes and edges inside an inclusive ``[start_ts, end_ts]`` window."""
    start = int(start_ts)
    end = int(end_ts)
    if start > end:
        raise ToolError("start_ts must be <= end_ts")

    coerced = _coerce_edges(edges)
    nodes, window_edges = window_snapshot(coerced, start, end, presorted=False)

    return {
        "start_ts": start,
        "end_ts": end,
        "num_edges": len(window_edges),
        "num_nodes": len(nodes),
        "window_seconds": end - start,
        "accounts": sorted(nodes)[:25],
    }


def account_features(
    edges: List[Dict[str, Any]],
    ref_time: float,
    accounts: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-account node features (degrees, volume, age, asset diversity)."""
    from astroml.features.node_features import compute_node_features

    frame = compute_node_features(edges, ref_time=float(ref_time))

    if accounts:
        missing = [str(account) for account in accounts if account not in frame.index]
        if missing:
            raise ToolError(f"unknown accounts: {', '.join(missing)}")
        frame = frame.loc[[str(account) for account in accounts]]

    return {
        str(node): {str(column): _to_float(value) for column, value in row.items()}
        for node, row in frame.iterrows()
    }


def top_accounts(
    edges: List[Dict[str, Any]],
    metric: str = "volume",
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """Rank accounts by transaction volume, total degree, in-degree or out-degree."""
    if metric not in ("volume", "degree", "in_degree", "out_degree"):
        raise ToolError(
            "unsupported metric "
            f"{metric!r}; expected volume, degree, in_degree or out_degree"
        )
    if int(top_k) < 1:
        raise ToolError("top_k must be >= 1")

    from astroml.features.node_features import compute_node_features

    frame = compute_node_features(edges)
    if frame.empty:
        return []

    if metric == "volume":
        required = ["total_sent", "total_received"]
    else:
        required = [metric]

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ToolError(f"feature frame is missing columns: {', '.join(missing)}")

    scores = frame[required].sum(axis=1)
    ranked = scores.sort_values(ascending=False).head(int(top_k))

    return [
        {"account": str(account), "score": float(value)}
        for account, value in ranked.items()
    ]


def anomaly_score_tool(
    scorer: Any,
    *,
    name: str = "score_accounts",
    description: Optional[str] = None,
) -> Tool:
    """Wrap a trained ``InductiveAnomalyScorer`` as an agent tool.

    A factory is used so that ``torch`` / ``torch-geometric`` are only
    imported by callers who actually own a trained scorer::

        from astroml.agent.domain_tools import anomaly_score_tool
        tools.register(anomaly_score_tool(my_scorer))
    """

    def score_accounts(
        edges: List[Dict[str, Any]],
        accounts: List[str],
        ref_time: float,
    ) -> Dict[str, float]:
        """Score accounts for anomalies with the configured Deep SVDD model."""
        if not hasattr(scorer, "score_new_accounts"):
            raise ToolError(
                "scorer must expose score_new_accounts(edges, account_ids, ref_time)"
            )
        if not accounts:
            raise ToolError("accounts must not be empty")
        scores = scorer.score_new_accounts(
            edges, [str(account) for account in accounts], float(ref_time)
        )
        return {str(key): float(value) for key, value in dict(scores).items()}

    return tool_from_callable(
        score_accounts, name=name, description=description
    )


#: Tools registered by :func:`build_default_registry`.
DOMAIN_TOOLS: Tuple[Tool, ...] = (
    tool_from_callable(graph_overview),
    tool_from_callable(window_stats),
    tool_from_callable(account_features),
    tool_from_callable(top_accounts),
)


def build_default_registry(
    *,
    include: Optional[Sequence[str]] = None,
) -> ToolRegistry:
    """Return a registry populated with the default AstroML domain tools.

    Args:
        include: Optional subset of tool names; every registered name must
            exist in :data:`DOMAIN_TOOLS`.
    """
    registry = ToolRegistry()
    for item in DOMAIN_TOOLS:
        if include is None or item.name in include:
            registry.register(item)
    registry_names = set(registry.names())
    unknown = [name for name in include or [] if name not in registry_names]
    if unknown:
        raise ToolError(f"unknown domain tools: {', '.join(unknown)}")
    return registry
