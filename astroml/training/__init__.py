from importlib import import_module

from . import temporal_split
from .temporal_split import TemporalSplitter, temporal_graph_split, validate_graph_split
from .config import (
    TrainingConfig,
    EarlyStoppingConfig,
    TemporalSplitConfig,
    OptimizerConfig,
)

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
]

_LAZY = {
    "train_link_prediction": ("astroml.training.train_link_prediction", "train_link_prediction"),
    "train_link_prediction_main": ("astroml.training.train_link_prediction", "main"),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        module = import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
