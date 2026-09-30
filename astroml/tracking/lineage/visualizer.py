"""Text-based and mermaid-format DAG visualization for lineage data.

Provides visualization capabilities without external dependencies,
supporting ASCII art trees, mermaid format, and simple HTML output.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LineageVisualizer:
    """Generates visualizations of lineage and provenance data.

    Supports ASCII DAG, mermaid format, and HTML output.
    """

    @staticmethod
    def visualize_dag(
        lineage_data: dict[str, Any],
        output_path: str | Path | None = None,
    ) -> str:
        """Generate a text-based DAG visualization.

        Args:
            lineage_data: Lineage data dict as returned by
                ``DataLineageTracker.get_lineage()``.
            output_path: Optional path to write output. If None, returns
                the text as a string.

        Returns:
            The visualisation as a string.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("LINEAGE DAG")
        lines.append("=" * 60)

        entity = lineage_data.get("entity", {})
        if entity:
            lines.append(
                f"\nEntity: {entity.get('id', 'unknown')} ({entity.get('type', 'unknown')})"
            )
            lines.append("-" * 40)

        upstream = lineage_data.get("upstream", [])
        downstream = lineage_data.get("downstream", [])

        if upstream:
            lines.append("\nUpstream dependencies:")
            _render_tree(entity.get("id", "") if entity else "", upstream, lines, prefix="")

        if downstream:
            lines.append("\nDownstream dependencies:")
            _render_tree(entity.get("id", "") if entity else "", downstream, lines, prefix="")

        if not upstream and not downstream:
            lines.append("\n(No lineage relationships found)")

        result = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(result)
            logger.info("DAG visualization written to %s", output_path)

        return result

    @staticmethod
    def visualize_timeline(
        provenance_data: dict[str, Any],
        output_path: str | Path | None = None,
    ) -> str:
        """Generate a timeline visualization from provenance data.

        Args:
            provenance_data: Provenance chain dict as returned by
                ``ProvenanceTracker.export_provenance(run_id, fmt="dict")``.
            output_path: Optional path to write output.

        Returns:
            The timeline as a string.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("PROVENANCE TIMELINE")
        lines.append("=" * 60)

        if not provenance_data:
            lines.append("\n(No provenance data)")
            result = "\n".join(lines)
            if output_path:
                Path(output_path).write_text(result)
            return result

        run_id = provenance_data.get("run_id", "unknown")
        created_at = provenance_data.get("created_at", "")
        lines.append(f"\nRun: {run_id}")
        lines.append(f"Created: {created_at}")
        lines.append("-" * 60)

        stages = provenance_data.get("stages", [])
        if not stages:
            lines.append("\n(No stages recorded)")
        else:
            for i, stage in enumerate(stages):
                _render_stage_timeline(i, stage, lines, indent=0)

        result = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(result)
            logger.info("Timeline visualization written to %s", output_path)

        return result

    @staticmethod
    def visualize_impact(
        impact_data: dict[str, Any],
        output_path: str | Path | None = None,
    ) -> str:
        """Generate an impact analysis visualization.

        Args:
            impact_data: Dict with keys: entity_id, entity_type,
                downstream_entities, metrics.
            output_path: Optional path to write output.

        Returns:
            The impact analysis as a string.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("IMPACT ANALYSIS")
        lines.append("=" * 60)

        entity_id = impact_data.get("entity_id", "unknown")
        entity_type = impact_data.get("entity_type", "unknown")
        lines.append(f"\nEntity: {entity_id} ({entity_type})")
        lines.append("-" * 40)

        downstream = impact_data.get("downstream_entities", [])
        if downstream:
            lines.append(f"\nDownstream impact ({len(downstream)} entities):")
            for de in downstream:
                if isinstance(de, dict):
                    de_id = de.get("id", str(de))
                    de_type = de.get("type", "?")
                    lines.append(f"  - {de_id} [{de_type}]")
                else:
                    lines.append(f"  - {de}")
        else:
            lines.append("\n(No downstream impact)")

        metrics = impact_data.get("metrics", {})
        if metrics:
            lines.append("\nImpact metrics:")
            for k, v in metrics.items():
                lines.append(f"  {k}: {v}")

        result = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(result)
            logger.info("Impact visualization written to %s", output_path)

        return result

    @staticmethod
    def to_mermaid(lineage_data: dict[str, Any]) -> str:
        """Generate a mermaid-format string for the lineage DAG.

        Args:
            lineage_data: Lineage data dict as returned by
                ``DataLineageTracker.get_lineage()``.

        Returns:
            Mermaid flowchart definition as a string.
        """
        full_dag = lineage_data.get("full_dag", {})
        nodes = full_dag.get("nodes", {})
        edges = full_dag.get("edges", [])

        lines: list[str] = []
        lines.append("flowchart TD")

        if not nodes:
            lines.append("  %% No nodes")
            return "\n".join(lines)

        for nid, node in nodes.items():
            safe_id = nid.replace("-", "_").replace(" ", "_")
            node_type = node.get("type", "unknown")
            shape = {
                "dataset": f"{safe_id}[{nid}]",
                "transformation": f"{safe_id}({nid})",
                "model": f"{safe_id}(({nid}))",
            }.get(node_type, f"{safe_id}[{nid}]")
            lines.append(f"  {shape}")

        for edge in edges:
            src = edge.get("source", "").replace("-", "_").replace(" ", "_")
            tgt = edge.get("target", "").replace("-", "_").replace(" ", "_")
            lines.append(f"  {src} --> {tgt}")

        return "\n".join(lines)

    @staticmethod
    def export_html(
        lineage_data: dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """Generate a simple HTML visualization of the lineage DAG.

        Creates a self-contained HTML file with embedded mermaid rendering.

        Args:
            lineage_data: Lineage data dict.
            output_path: Path to write the HTML file.
        """
        mermaid_code = LineageVisualizer.to_mermaid(lineage_data)
        entity = lineage_data.get("entity", {})
        entity_id = entity.get("id", "Unknown") if entity else "Unknown"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lineage DAG - {entity_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
  .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  h1 {{ color: #333; }}
  pre {{ background: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; padding: 10px; overflow-x: auto; }}
</style>
</head>
<body>
<div class="container">
  <h1>Lineage DAG: {entity_id}</h1>
  <pre class="mermaid">
{mermaid_code}
  </pre>
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
</script>
</body>
</html>"""

        Path(output_path).write_text(html)
        logger.info("HTML visualization written to %s", output_path)


def _render_tree(
    entity_id: str,
    records: list[dict[str, Any]],
    lines: list[str],
    prefix: str = "",
) -> None:
    """Render a list of records as a tree structure.

    Args:
        entity_id: ID of the root entity to exclude from tree.
        records: List of record dicts to render.
        lines: Accumulator for output lines.
        prefix: Current indentation prefix.
    """
    for i, rec in enumerate(records):
        is_last = i == len(records) - 1
        connector = "└── " if is_last else "├── "
        rid = rec.get("id", "?")
        rtype = rec.get("type", "?")
        if rid == entity_id:
            continue
        lines.append(f"{prefix}{connector}{rid} [{rtype}]")
        child_prefix = prefix + ("    " if is_last else "│   ")
        children = rec.get("child_ids", [])
        if children:
            lines.append(f"{child_prefix}(children: {', '.join(children)})")


def _render_stage_timeline(
    index: int,
    stage: dict[str, Any],
    lines: list[str],
    indent: int = 0,
) -> None:
    """Render a single stage for timeline visualization.

    Args:
        index: Stage index.
        stage: Stage record dict.
        lines: Accumulator for output lines.
        indent: Indentation level for nested stages.
    """
    prefix = "  " * indent
    name = stage.get("name", "?")
    duration = stage.get("duration_seconds", "N/A")
    lines.append(f"\n{prefix}Stage {index}: {name}")
    lines.append(
        f"{prefix}  Duration: {duration}s" if duration is not None else f"{prefix}  Duration: N/A"
    )

    input_rows = stage.get("row_count_input")
    output_rows = stage.get("row_count_output")
    if input_rows is not None:
        lines.append(f"{prefix}  Rows: {input_rows} -> {output_rows}")
    if stage.get("checksum_input") and stage.get("checksum_output"):
        lines.append(
            f"{prefix}  Checksum: {stage['checksum_input'][:16]}... -> {stage['checksum_output'][:16]}..."
        )

    input_schema = stage.get("input_schema", {})
    output_schema = stage.get("output_schema", {})
    if input_schema:
        lines.append(
            f"{prefix}  Input schema: {', '.join(f'{k}={v}' for k, v in input_schema.items())}"
        )
    if output_schema:
        lines.append(
            f"{prefix}  Output schema: {', '.join(f'{k}={v}' for k, v in output_schema.items())}"
        )

    nested = stage.get("nested_stages", [])
    for j, ns in enumerate(nested):
        _render_stage_timeline(j, ns, lines, indent=indent + 1)
