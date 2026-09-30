"""Data flow diagram generation for ML pipelines.

Resolves part of #646.

A pipeline is described as a small directed acyclic graph of
:class:`DataFlowNode` objects joined by :class:`DataFlowEdge` objects.  The
graph renders to Mermaid (for Markdown and MkDocs), Graphviz DOT (for
publication-quality images) and JSON (for downstream tooling), with no
third-party dependency.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "DataFlowDiagram",
    "DataFlowEdge",
    "DataFlowNode",
    "NodeKind",
]


class NodeKind(str, Enum):
    """The role a node plays in the pipeline."""

    SOURCE = "source"
    TRANSFORM = "transform"
    FEATURE = "feature"
    MODEL = "model"
    VALIDATION = "validation"
    SINK = "sink"


#: Mermaid node shape delimiters keyed by node kind.
_MERMAID_SHAPES: dict[NodeKind, tuple[str, str]] = {
    NodeKind.SOURCE: ("[(", ")]"),
    NodeKind.TRANSFORM: ("[", "]"),
    NodeKind.FEATURE: ("([", "])"),
    NodeKind.MODEL: ("{{", "}}"),
    NodeKind.VALIDATION: (">", "]"),
    NodeKind.SINK: ("[(", ")]"),
}

#: Graphviz node shapes keyed by node kind.
_DOT_SHAPES: dict[NodeKind, str] = {
    NodeKind.SOURCE: "cylinder",
    NodeKind.TRANSFORM: "box",
    NodeKind.FEATURE: "ellipse",
    NodeKind.MODEL: "hexagon",
    NodeKind.VALIDATION: "diamond",
    NodeKind.SINK: "cylinder",
}

#: Fill colours keyed by node kind, chosen to stay legible on light and dark.
_DOT_COLORS: dict[NodeKind, str] = {
    NodeKind.SOURCE: "#cfe8ff",
    NodeKind.TRANSFORM: "#e6e6e6",
    NodeKind.FEATURE: "#d9f2d9",
    NodeKind.MODEL: "#ffe0b3",
    NodeKind.VALIDATION: "#f7d6e0",
    NodeKind.SINK: "#cfe8ff",
}


@dataclass(frozen=True)
class DataFlowNode:
    """A stage in the pipeline."""

    node_id: str
    label: str
    kind: NodeKind = NodeKind.TRANSFORM
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the node identifier."""
        if not self.node_id:
            raise ValueError("node_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the node."""
        return {
            "node_id": self.node_id,
            "label": self.label,
            "kind": self.kind.value,
            "description": self.description,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DataFlowEdge:
    """A directed dependency between two stages."""

    source: str
    target: str
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the edge."""
        return {"source": self.source, "target": self.target, "label": self.label}


class DataFlowDiagram:
    """A pipeline data flow graph that renders to Mermaid, DOT and JSON."""

    def __init__(self, name: str = "pipeline") -> None:
        self.name = name
        self._nodes: dict[str, DataFlowNode] = {}
        self._edges: list[DataFlowEdge] = []

    # ── Construction ─────────────────────────────────────────────────────────

    def add_node(self, node: DataFlowNode) -> DataFlowNode:
        """Add (or replace) a node."""
        self._nodes[node.node_id] = node
        return node

    def add_nodes(self, nodes: Iterable[DataFlowNode]) -> None:
        """Add several nodes."""
        for node in nodes:
            self.add_node(node)

    def add_edge(self, source: str, target: str, *, label: str = "") -> DataFlowEdge:
        """Connect two existing nodes.

        Raises ``KeyError`` if either endpoint is unknown, so typos in a
        pipeline definition surface at build time rather than in the diagram.
        """
        for endpoint in (source, target):
            if endpoint not in self._nodes:
                raise KeyError(f"unknown node {endpoint!r}")
        edge = DataFlowEdge(source=source, target=target, label=label)
        self._edges.append(edge)
        return edge

    @property
    def nodes(self) -> list[DataFlowNode]:
        """Return every node in insertion order."""
        return list(self._nodes.values())

    @property
    def edges(self) -> list[DataFlowEdge]:
        """Return every edge in insertion order."""
        return list(self._edges)

    # ── Analysis ─────────────────────────────────────────────────────────────

    def topological_order(self) -> list[str]:
        """Return node ids in dependency order.

        Raises ``ValueError`` when the graph contains a cycle — an ML pipeline
        with a cycle is a definition bug, not something to render.
        """
        indegree: dict[str, int] = {node_id: 0 for node_id in self._nodes}
        successors: defaultdict[str, list[str]] = defaultdict(list)
        for edge in self._edges:
            indegree[edge.target] += 1
            successors[edge.source].append(edge.target)

        queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
        order: list[str] = []
        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            for successor in successors[node_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    queue.append(successor)
        if len(order) != len(self._nodes):
            remaining = sorted(set(self._nodes) - set(order))
            raise ValueError(f"data flow contains a cycle involving {remaining}")
        return order

    def validate(self) -> list[str]:
        """Return structural problems: orphan nodes, cycles, missing sinks."""
        problems: list[str] = []
        if not self._nodes:
            return ["diagram has no nodes"]
        connected = {edge.source for edge in self._edges} | {edge.target for edge in self._edges}
        for node_id in self._nodes:
            if len(self._nodes) > 1 and node_id not in connected:
                problems.append(f"node {node_id!r} is not connected to the pipeline")
        try:
            self.topological_order()
        except ValueError as exc:
            problems.append(str(exc))
        if not any(node.kind is NodeKind.SOURCE for node in self._nodes.values()):
            problems.append("pipeline has no source node")
        if not any(node.kind is NodeKind.SINK for node in self._nodes.values()):
            problems.append("pipeline has no sink node")
        return problems

    # ── Rendering ────────────────────────────────────────────────────────────

    def to_mermaid(self, *, direction: str = "LR") -> str:
        """Render the graph as a Mermaid ``flowchart``."""
        if direction not in ("LR", "RL", "TB", "BT"):
            raise ValueError("direction must be one of LR, RL, TB, BT")
        lines = [f"flowchart {direction}"]
        for node in self._nodes.values():
            open_shape, close_shape = _MERMAID_SHAPES[node.kind]
            label = _escape_mermaid(node.label)
            lines.append(f'    {_safe_id(node.node_id)}{open_shape}"{label}"{close_shape}')
        for edge in self._edges:
            arrow = f'-- "{_escape_mermaid(edge.label)}" -->' if edge.label else "-->"
            lines.append(f"    {_safe_id(edge.source)} {arrow} {_safe_id(edge.target)}")
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Render the graph as Graphviz DOT."""
        lines = [
            f'digraph "{_escape_dot(self.name)}" {{',
            "    rankdir=LR;",
            '    node [style="filled,rounded", fontname="Helvetica"];',
        ]
        for node in self._nodes.values():
            lines.append(
                f'    "{_escape_dot(node.node_id)}" '
                f'[label="{_escape_dot(node.label)}", '
                f"shape={_DOT_SHAPES[node.kind]}, "
                f'fillcolor="{_DOT_COLORS[node.kind]}"];'
            )
        for edge in self._edges:
            label = f' [label="{_escape_dot(edge.label)}"]' if edge.label else ""
            lines.append(
                f'    "{_escape_dot(edge.source)}" -> "{_escape_dot(edge.target)}"{label};'
            )
        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the graph."""
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return the graph as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DataFlowDiagram:
        """Rebuild a diagram from :meth:`to_dict` output."""
        diagram = cls(name=payload.get("name", "pipeline"))
        for raw in payload.get("nodes", []):
            diagram.add_node(
                DataFlowNode(
                    node_id=raw["node_id"],
                    label=raw.get("label", raw["node_id"]),
                    kind=NodeKind(raw.get("kind", NodeKind.TRANSFORM.value)),
                    description=raw.get("description", ""),
                    metadata=dict(raw.get("metadata", {})),
                )
            )
        for raw in payload.get("edges", []):
            diagram.add_edge(raw["source"], raw["target"], label=raw.get("label", ""))
        return diagram

    @classmethod
    def from_stages(
        cls,
        name: str,
        stages: list[tuple[str, NodeKind]],
    ) -> DataFlowDiagram:
        """Build a linear diagram from an ordered ``(label, kind)`` list."""
        diagram = cls(name=name)
        previous: str | None = None
        for index, (label, kind) in enumerate(stages):
            node_id = f"s{index}"
            diagram.add_node(DataFlowNode(node_id=node_id, label=label, kind=kind))
            if previous is not None:
                diagram.add_edge(previous, node_id)
            previous = node_id
        return diagram


def _safe_id(node_id: str) -> str:
    """Return a Mermaid-safe node identifier."""
    return "".join(char if char.isalnum() or char == "_" else "_" for char in node_id)


def _escape_mermaid(text: str) -> str:
    """Escape characters that break Mermaid labels."""
    return text.replace('"', "'").replace("\n", " ")


def _escape_dot(text: str) -> str:
    """Escape characters that break DOT string literals."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
