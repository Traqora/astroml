"""Benchmarking suite for GNN models on Stellar data."""

from .config import BenchmarkConfig as FullBenchmarkConfig
from .config import (
    ConfigManager,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    create_config_from_template,
    validate_config,
)
from .core import BenchmarkConfig, BenchmarkResult, ModelBenchmark
from .metrics import (
    AnomalyDetectionMetrics,
    ClassificationMetrics,
    LinkPredictionMetrics,
    MetricCalculator,
    RegressionMetrics,
)
from .utils import (
    MemoryMonitor,
    Timer,
    format_memory,
    format_time,
    get_device_info,
    get_environment_info,
    measure_gpu_memory,
    measure_memory_usage,
    set_random_seed,
)

__all__ = [
    # Core
    "BenchmarkConfig",
    "BenchmarkResult",
    "ModelBenchmark",
    # Metrics
    "ClassificationMetrics",
    "LinkPredictionMetrics",
    "AnomalyDetectionMetrics",
    "RegressionMetrics",
    "MetricCalculator",
    # Configuration
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "FullBenchmarkConfig",
    "ConfigManager",
    "create_config_from_template",
    "validate_config",
    # Utilities
    "Timer",
    "MemoryMonitor",
    "measure_memory_usage",
    "measure_gpu_memory",
    "format_time",
    "format_memory",
    "set_random_seed",
    "get_device_info",
    "get_environment_info",
]
