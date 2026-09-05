from importlib import import_module

__all__ = [
    "temporal_split",
    "TemporalSplitter",
    "temporal_graph_split",
    "validate_graph_split",
    "train_link_prediction",
    "train_link_prediction_main",
    "TrainingConfig",
    "EarlyStoppingConfig",
    "TemporalSplitConfig",
    "OptimizerConfig",
    "ONNXConverter",
    "ONNXOptimizer",
    "QuantizationConfig",
    "ContinuousLearningPipeline",
    "ModelVersionManager",
    "GraphObjective",
    "NodeClassificationObjective",
    "LinkPredictionObjective",
    "get_objective",
    "register_objective",
    "available_objectives",
    "validate_edge_index",
    "validate_masks",
]

_LAZY = {
    "temporal_split": ("astroml.training.temporal_split", None),
    "TemporalSplitter": ("astroml.training.temporal_split", "TemporalSplitter"),
    "temporal_graph_split": ("astroml.training.temporal_split", "temporal_graph_split"),
    "validate_graph_split": ("astroml.training.temporal_split", "validate_graph_split"),
    "train_link_prediction": ("astroml.training.train_link_prediction", "train_link_prediction"),
    "train_link_prediction_main": ("astroml.training.train_link_prediction", "main"),
    "ONNXConverter": ("astroml.training.optimization.onnx_converter", "ONNXConverter"),
    "ONNXOptimizer": ("astroml.training.optimization.onnx_optimizer", "ONNXOptimizer"),
    "QuantizationConfig": ("astroml.training.optimization.quantization", "QuantizationConfig"),
    "TrainingConfig": ("astroml.training.config", "TrainingConfig"),
    "EarlyStoppingConfig": ("astroml.training.config", "EarlyStoppingConfig"),
    "TemporalSplitConfig": ("astroml.training.config", "TemporalSplitConfig"),
    "OptimizerConfig": ("astroml.training.config", "OptimizerConfig"),
    "ContinuousLearningPipeline": ("astroml.training.continuous_learning", "ContinuousLearningPipeline"),
    "ModelVersionManager": ("astroml.training.continuous_learning", "ModelVersionManager"),
    "GraphObjective": ("astroml.training.graph_objectives", "GraphObjective"),
    "NodeClassificationObjective": (
        "astroml.training.graph_objectives",
        "NodeClassificationObjective",
    ),
    "LinkPredictionObjective": (
        "astroml.training.graph_objectives",
        "LinkPredictionObjective",
    ),
    "get_objective": ("astroml.training.graph_objectives", "get_objective"),
    "register_objective": ("astroml.training.graph_objectives", "register_objective"),
    "available_objectives": ("astroml.training.graph_objectives", "available_objectives"),
    "validate_edge_index": ("astroml.training.graph_objectives", "validate_edge_index"),
    "validate_masks": ("astroml.training.graph_objectives", "validate_masks"),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        module = import_module(module_path)
        value = getattr(module, attr) if attr else module
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
