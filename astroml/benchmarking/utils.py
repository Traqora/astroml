"""Utility functions for benchmarking."""

from __future__ import annotations

import time
import psutil
import os
from typing import Dict, Any, Optional
import torch


def measure_memory_usage() -> Dict[str, float]:
    """Measure current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
    }


def measure_gpu_memory() -> Optional[float]:
    """Measure GPU memory usage if available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    return None


def format_time(seconds: float) -> str:
    """Format time in human readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_memory(mb: float) -> str:
    """Format memory in human readable format."""
    if mb < 1024:
        return f"{mb:.1f}MB"
    else:
        gb = mb / 1024
        return f"{gb:.1f}GB"


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        
        if self.elapsed < 1.0:
            print(f"{self.description}: {self.elapsed*1000:.2f}ms")
        else:
            print(f"{self.description}: {self.elapsed:.2f}s")
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed is None:
            raise RuntimeError("Timer has not been stopped")
        return self.elapsed


class MemoryMonitor:
    """Context manager for monitoring memory usage."""
    
    def __init__(self, description: str = "Memory usage"):
        self.description = description
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        self.gpu_start = None
        self.gpu_end = None
    
    def __enter__(self):
        self.start_memory = measure_memory_usage()
        self.gpu_start = measure_gpu_memory()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_memory = measure_memory_usage()
        self.gpu_end = measure_gpu_memory()
        
        # Calculate peak memory (approximation)
        self.peak_memory = max(self.start_memory['rss_mb'], self.end_memory['rss_mb'])
        
        # Print memory usage
        rss_delta = self.end_memory['rss_mb'] - self.start_memory['rss_mb']
        print(f"{self.description}:")
        print(f"  RSS: {format_memory(self.start_memory['rss_mb'])} -> {format_memory(self.end_memory['rss_mb'])} ({format_memory(abs(rss_delta))} {'increase' if rss_delta > 0 else 'decrease'})")
        
        if self.gpu_start is not None and self.gpu_end is not None:
            gpu_delta = self.gpu_end - self.gpu_start
            print(f"  GPU: {format_memory(self.gpu_start)} -> {format_memory(self.gpu_end)} ({format_memory(abs(gpu_delta))} {'increase' if gpu_delta > 0 else 'decrease'})")
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if self.peak_memory is None:
            raise RuntimeError("MemoryMonitor has not been stopped")
        return self.peak_memory


def validate_config(config: Dict[str, Any]) -> None:
    """Validate benchmark configuration."""
    required_fields = ['model_name', 'model_params', 'epochs', 'batch_size']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate data ratios
    ratios = ['train_ratio', 'val_ratio', 'test_ratio']
    total_ratio = sum(config.get(ratio, 0) for ratio in ratios)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")
    
    # Validate model parameters
    model_name = config['model_name'].lower()
    valid_models = ['gcn', 'link_predictor', 'sage_encoder', 'deep_svdd']
    
    if model_name not in valid_models:
        raise ValueError(f"Unknown model: {model_name}. Valid models: {valid_models}")


def create_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to JSON file."""
    import json
    
    create_output_directory(os.path.dirname(filepath))
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    import json
    
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Compute model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random  # noqa: E402
    import numpy as np  # noqa: E402
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    info = {
        'cpu_count': psutil.cpu_count(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info[f'cuda_device_{i}'] = {
                'name': props.name,
                'total_memory': props.total_memory / (1024**3),  # GB
                'compute_capability': f"{props.major}.{props.minor}"
            }
    
    return info


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    epochs: int,
    samples_per_second: float = 100.0  # Estimated processing rate
) -> float:
    """Estimate training time based on dataset size and hardware."""
    batches_per_epoch = num_samples / batch_size
    total_batches = batches_per_epoch * epochs
    estimated_seconds = total_batches / samples_per_second
    return estimated_seconds


def create_progress_callback(description: str):
    """Create a progress callback for training."""
    def callback(epoch: int, loss: float, metrics: Dict[str, float]):
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return callback


def get_environment_info() -> Dict[str, Any]:
    """Collect environment information for reproducibility."""
    import sys
    import platform
    from importlib.metadata import version
    
    env_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'platform_system': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'platform_machine': platform.machine(),
        'processor': platform.processor(),
    }
    
    # Get library versions
    libraries = ['torch', 'numpy', 'scikit-learn', 'pandas', 'torch-geometric']
    for lib in libraries:
        try:
            env_info[f'{lib}_version'] = version(lib)
        except Exception:
            try:
                # Fallback for packages with different import names
                import importlib
                module = importlib.import_module(lib.replace('-', '_'))
                env_info[f'{lib}_version'] = getattr(module, '__version__', 'unknown')
            except Exception:
                env_info[f'{lib}_version'] = 'not_installed'
    
    return env_info
