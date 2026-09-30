"""Automated ML pipeline documentation generation.

Resolves #646.

Submodules
----------
``generator``
    Metadata extraction, Markdown rendering, versioning and publishing.
``model_card``
    Google Model Cards specification support.
``data_flow``
    Mermaid / Graphviz data flow diagrams.
``templates``
    Packaged ``string.Template`` documents.
"""

from __future__ import annotations

from astroml.pipeline.documentation.data_flow import (
    DataFlowDiagram,
    DataFlowEdge,
    DataFlowNode,
    NodeKind,
)
from astroml.pipeline.documentation.generator import (
    DocumentVersion,
    GeneratedDoc,
    IOSpec,
    PipelineDocGenerator,
    PipelineMetadata,
    StageMetadata,
    extract_metadata,
)
from astroml.pipeline.documentation.model_card import (
    ConsiderationSection,
    EvaluationData,
    IntendedUse,
    MetricEntry,
    ModelCard,
    ModelCardBuilder,
    ModelDetails,
    QuantitativeAnalysis,
    TrainingData,
)
from astroml.pipeline.documentation.templates import available_templates, load_template

__all__ = [
    "ConsiderationSection",
    "DataFlowDiagram",
    "DataFlowEdge",
    "DataFlowNode",
    "DocumentVersion",
    "EvaluationData",
    "GeneratedDoc",
    "IOSpec",
    "IntendedUse",
    "MetricEntry",
    "ModelCard",
    "ModelCardBuilder",
    "ModelDetails",
    "NodeKind",
    "PipelineDocGenerator",
    "PipelineMetadata",
    "QuantitativeAnalysis",
    "StageMetadata",
    "TrainingData",
    "available_templates",
    "extract_metadata",
    "load_template",
]
