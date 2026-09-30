from typing import Any, Dict, List, Optional, Union, Callable
"""Inference pipelines for AstroML."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "InductiveGraphSAGE",
    "InductiveAnomalyScorer",
    "SchemaContract",
    "QualityContract",
    "SemanticContract",
    "ContractVerifier",
    "DataFlowDiagram",
    "ModelCard",
    "ModelCardBuilder",
    "PipelineDocGenerator",
    "PipelineMetadata",
]

_LAZY: dict[str, tuple[str, str]] = {
    "InductiveGraphSAGE": ("astroml.pipeline.inductive", "InductiveGraphSAGE"),
    "InductiveAnomalyScorer": ("astroml.pipeline.scoring", "InductiveAnomalyScorer"),
    "SchemaContract": ("astroml.pipeline.contracts.schema_contract", "SchemaContract"),
    "QualityContract": ("astroml.pipeline.contracts.quality_contract", "QualityContract"),
    "SemanticContract": ("astroml.pipeline.contracts.semantic_contract", "SemanticContract"),
    "ContractVerifier": ("astroml.pipeline.contracts.verifier", "ContractVerifier"),
    "DataFlowDiagram": ("astroml.pipeline.documentation.data_flow", "DataFlowDiagram"),
    "ModelCard": ("astroml.pipeline.documentation.model_card", "ModelCard"),
    "ModelCardBuilder": ("astroml.pipeline.documentation.model_card", "ModelCardBuilder"),
    "PipelineDocGenerator": ("astroml.pipeline.documentation.generator", "PipelineDocGenerator"),
    "PipelineMetadata": ("astroml.pipeline.documentation.generator", "PipelineMetadata"),
}


def __getattr__(name -> Any: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        module = import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
