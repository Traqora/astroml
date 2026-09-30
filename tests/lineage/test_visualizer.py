"""Tests for LineageVisualizer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from astroml.tracking.lineage.visualizer import LineageVisualizer

SAMPLE_LINEAGE = {
    "entity": {
        "id": "model_v1",
        "type": "model",
        "timestamp": "2025-01-01T00:00:00",
        "metadata": {},
        "parent_ids": ["processed_data"],
        "child_ids": ["pred_001"],
    },
    "upstream": [
        {
            "id": "raw_data",
            "type": "dataset",
            "timestamp": "2025-01-01T00:00:00",
            "metadata": {},
            "parent_ids": [],
            "child_ids": ["clean_tx"],
        },
        {
            "id": "clean_tx",
            "type": "transformation",
            "timestamp": "2025-01-01T00:00:00",
            "metadata": {},
            "parent_ids": ["raw_data"],
            "child_ids": ["processed_data"],
        },
        {
            "id": "processed_data",
            "type": "dataset",
            "timestamp": "2025-01-01T00:00:00",
            "metadata": {},
            "parent_ids": ["clean_tx"],
            "child_ids": ["model_v1"],
        },
    ],
    "downstream": [
        {
            "id": "pred_001",
            "type": "dataset",
            "timestamp": "2025-01-01T00:00:00",
            "metadata": {},
            "parent_ids": ["model_v1", "processed_data"],
            "child_ids": [],
        },
    ],
    "full_dag": {
        "nodes": {
            "raw_data": {"id": "raw_data", "type": "dataset", "label": "raw_data"},
            "clean_tx": {"id": "clean_tx", "type": "transformation", "label": "clean_tx"},
            "processed_data": {
                "id": "processed_data",
                "type": "dataset",
                "label": "processed_data",
            },
            "model_v1": {"id": "model_v1", "type": "model", "label": "model_v1"},
            "pred_001": {"id": "pred_001", "type": "dataset", "label": "pred_001"},
        },
        "edges": [
            {"source": "raw_data", "target": "clean_tx"},
            {"source": "clean_tx", "target": "processed_data"},
            {"source": "processed_data", "target": "model_v1"},
            {"source": "model_v1", "target": "pred_001"},
        ],
    },
}

EMPTY_LINEAGE = {
    "entity": {},
    "upstream": [],
    "downstream": [],
    "full_dag": {"nodes": {}, "edges": []},
}

SAMPLE_PROVENANCE = {
    "run_id": "run_001",
    "stages": [
        {
            "name": "ingest",
            "start_time": "2025-01-01T00:00:00",
            "end_time": "2025-01-01T00:01:00",
            "duration_seconds": 60.0,
            "row_count_input": None,
            "row_count_output": 100,
            "checksum_input": None,
            "checksum_output": "abc123",
            "input_schema": {},
            "output_schema": {"col1": "int"},
            "metadata": {},
            "nested_stages": [],
        },
    ],
    "created_at": "2025-01-01T00:01:00",
}

SAMPLE_IMPACT = {
    "entity_id": "model_v1",
    "entity_type": "model",
    "downstream_entities": [
        {"id": "pred_001", "type": "dataset"},
        {"id": "pred_002", "type": "dataset"},
    ],
    "metrics": {"total_downstream": 2, "avg_confidence": 0.85},
}


class TestLineageVisualizer:
    """Tests for LineageVisualizer."""

    def test_visualize_dag_returns_string(self) -> None:
        result = LineageVisualizer.visualize_dag(SAMPLE_LINEAGE)
        assert isinstance(result, str)
        assert "LINEAGE DAG" in result
        assert "model_v1" in result

    def test_visualize_dag_empty_lineage(self) -> None:
        result = LineageVisualizer.visualize_dag(EMPTY_LINEAGE)
        assert isinstance(result, str)
        assert "No lineage" in result or "LINEAGE DAG" in result

    def test_visualize_dag_to_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            outpath = f.name

        try:
            LineageVisualizer.visualize_dag(SAMPLE_LINEAGE, output_path=outpath)
            content = Path(outpath).read_text()
            assert "LINEAGE DAG" in content
        finally:
            Path(outpath).unlink(missing_ok=True)

    def test_visualize_timeline(self) -> None:
        result = LineageVisualizer.visualize_timeline(SAMPLE_PROVENANCE)
        assert isinstance(result, str)
        assert "PROVENANCE TIMELINE" in result
        assert "ingest" in result

    def test_visualize_timeline_empty(self) -> None:
        result = LineageVisualizer.visualize_timeline({})
        assert "No provenance data" in result

    def test_visualize_timeline_no_stages(self) -> None:
        data = {"run_id": "run1", "stages": [], "created_at": "2025-01-01T00:00:00"}
        result = LineageVisualizer.visualize_timeline(data)
        assert "No stages" in result

    def test_visualize_timeline_to_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            outpath = f.name

        try:
            LineageVisualizer.visualize_timeline(SAMPLE_PROVENANCE, output_path=outpath)
            content = Path(outpath).read_text()
            assert "PROVENANCE TIMELINE" in content
        finally:
            Path(outpath).unlink(missing_ok=True)

    def test_visualize_impact(self) -> None:
        result = LineageVisualizer.visualize_impact(SAMPLE_IMPACT)
        assert isinstance(result, str)
        assert "IMPACT ANALYSIS" in result
        assert "model_v1" in result
        assert "pred_001" in result

    def test_visualize_impact_no_downstream(self) -> None:
        data = {"entity_id": "x", "entity_type": "model", "downstream_entities": [], "metrics": {}}
        result = LineageVisualizer.visualize_impact(data)
        assert "No downstream impact" in result

    def test_visualize_impact_to_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            outpath = f.name

        try:
            LineageVisualizer.visualize_impact(SAMPLE_IMPACT, output_path=outpath)
            content = Path(outpath).read_text()
            assert "IMPACT ANALYSIS" in content
        finally:
            Path(outpath).unlink(missing_ok=True)

    def test_to_mermaid(self) -> None:
        result = LineageVisualizer.to_mermaid(SAMPLE_LINEAGE)
        assert isinstance(result, str)
        assert "flowchart TD" in result
        assert "raw_data" in result
        assert "-->" in result

    def test_to_mermaid_empty(self) -> None:
        result = LineageVisualizer.to_mermaid(EMPTY_LINEAGE)
        assert "flowchart TD" in result

    def test_to_mermaid_shapes(self) -> None:
        result = LineageVisualizer.to_mermaid(SAMPLE_LINEAGE)
        assert "raw_data[" in result  # dataset
        assert "clean_tx(" in result  # transformation
        assert "model_v1((" in result  # model

    def test_export_html(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            outpath = f.name

        try:
            LineageVisualizer.export_html(SAMPLE_LINEAGE, output_path=outpath)
            content = Path(outpath).read_text()
            assert "Lineage DAG" in content
            assert "mermaid" in content
            assert "flowchart TD" in content
        finally:
            Path(outpath).unlink(missing_ok=True)
