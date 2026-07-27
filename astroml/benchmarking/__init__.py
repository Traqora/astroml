"""Benchmarking suite for GNN models on Stellar data."""

from .core import BenchmarkConfig, BenchmarkResult, ModelBenchmark
from .metrics import (
    ClassificationMetrics,
    LinkPredictionMetrics,
    AnomalyDetectionMetrics,
    RegressionMetrics,
    MetricCalculator
)
from .config import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    BenchmarkConfig as FullBenchmarkConfig,
    ConfigManager,
    create_config_from_template,
    validate_config
)
from .utils import (
    Timer,
    MemoryMonitor,
    measure_memory_usage,
    measure_gpu_memory,
    format_time,
    format_memory,
    set_random_seed,
    get_device_info,
    get_environment_info
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
    "get_environment_info"
]
