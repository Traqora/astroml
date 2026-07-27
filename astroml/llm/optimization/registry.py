"""
Registry for optimized models with metadata and versioning.

Tracks optimized models, their metrics, and deployment targets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class OptimizedModelEntry:
    """Entry in the optimized model registry."""
    model_id: str
    model_path: str
    base_model: str
    optimization_type: str  # quantization, distillation, compression
    created_at: str
    quality_score: float
    size_reduction: float
    speedup: float
    deployment_targets: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "base_model": self.base_model,
            "optimization_type": self.optimization_type,
            "created_at": self.created_at,
            "quality_score": self.quality_score,
            "size_reduction": self.size_reduction,
            "speedup": self.speedup,
            "deployment_targets": self.deployment_targets,
            "metadata": self.metadata,
        }


class OptimizedModelRegistry:
    """
    Registry for tracking optimized models and their metrics.

    Stores model metadata, quality metrics, and deployment compatibility.
    """

    def __init__(self):
        """Initialize registry."""
        self._models: Dict[str, OptimizedModelEntry] = {}

    def register_model(
        self,
        model_id: str,
        model_path: str,
        base_model: str,
        optimization_type: str,
        quality_score: float,
        size_reduction: float,
        speedup: float,
        deployment_targets: Optional[List[str]] = None,
    ) -> OptimizedModelEntry:
        """
        Register an optimized model.

        Args:
            model_id: Unique model identifier
            model_path: Path to optimized model file
            base_model: Base model name
            optimization_type: Type of optimization applied
            quality_score: Quality retention score (0-1)
            size_reduction: Size reduction ratio (0-1)
            speedup: Inference speedup factor
            deployment_targets: List of compatible deployment targets

        Returns:
            OptimizedModelEntry
        """
        entry = OptimizedModelEntry(
            model_id=model_id,
            model_path=model_path,
            base_model=base_model,
            optimization_type=optimization_type,
            created_at=datetime.now().isoformat(),
            quality_score=quality_score,
            size_reduction=size_reduction,
            speedup=speedup,
            deployment_targets=deployment_targets or [],
        )

        self._models[model_id] = entry
        return entry

    def get_model(self, model_id: str) -> Optional[OptimizedModelEntry]:
        """Get model entry by ID."""
        return self._models.get(model_id)

    def list_models(self) -> List[OptimizedModelEntry]:
        """List all registered models."""
        return list(self._models.values())

    def find_best_model_for_target(
        self,
        base_model: str,
        deployment_target: str,
        min_quality: float = 0.90,
    ) -> Optional[OptimizedModelEntry]:
        """
        Find best optimized model for a deployment target.

        Args:
            base_model: Base model name
            deployment_target: Deployment target (e.g., 'jetson')
            min_quality: Minimum quality score required

        Returns:
            Best matching OptimizedModelEntry or None
        """
        candidates = [
            m for m in self._models.values()
            if m.base_model == base_model
            and m.quality_score >= min_quality
            and deployment_target in m.deployment_targets
        ]

        if not candidates:
            return None

        # Return model with best speedup
        return max(candidates, key=lambda m: m.speedup)

    def list_by_optimization_type(self, optimization_type: str) -> List[OptimizedModelEntry]:
        """List models by optimization type."""
        return [
            m for m in self._models.values()
            if m.optimization_type == optimization_type
        ]

    def list_by_base_model(self, base_model: str) -> List[OptimizedModelEntry]:
        """List all optimizations of a base model."""
        return [
            m for m in self._models.values()
            if m.base_model == base_model
        ]

    def update_deployment_targets(
        self,
        model_id: str,
        deployment_targets: List[str],
    ) -> bool:
        """
        Update deployment targets for a model.

        Args:
            model_id: Model ID
            deployment_targets: New list of deployment targets

        Returns:
            True if updated successfully
        """
        if model_id not in self._models:
            return False

        self._models[model_id].deployment_targets = deployment_targets
        return True

    def delete_model(self, model_id: str) -> bool:
        """Delete model from registry."""
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False

    def export_registry(self, output_path: str) -> None:
        """
        Export registry to JSON file.

        Args:
            output_path: Path to save registry JSON
        """
        data = [m.to_dict() for m in self._models.values()]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def import_registry(self, input_path: str) -> None:
        """
        Import registry from JSON file.

        Args:
            input_path: Path to registry JSON
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        for item in data:
            entry = OptimizedModelEntry(
                model_id=item['model_id'],
                model_path=item['model_path'],
                base_model=item['base_model'],
                optimization_type=item['optimization_type'],
                created_at=item['created_at'],
                quality_score=item['quality_score'],
                size_reduction=item['size_reduction'],
                speedup=item['speedup'],
                deployment_targets=item.get('deployment_targets', []),
                metadata=item.get('metadata', {}),
            )
            self._models[entry.model_id] = entry

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        models = list(self._models.values())

        if not models:
            return {
                "total_models": 0,
                "avg_quality_score": 0.0,
                "avg_size_reduction": 0.0,
                "avg_speedup": 0.0,
            }

        return {
            "total_models": len(models),
            "by_type": self._count_by_type(models),
            "by_base_model": self._count_by_base_model(models),
            "avg_quality_score": sum(m.quality_score for m in models) / len(models),
            "avg_size_reduction": sum(m.size_reduction for m in models) / len(models),
            "avg_speedup": sum(m.speedup for m in models) / len(models),
        }

    def _count_by_type(self, models: List[OptimizedModelEntry]) -> Dict[str, int]:
        """Count models by optimization type."""
        counts = {}
        for model in models:
            counts[model.optimization_type] = counts.get(model.optimization_type, 0) + 1
        return counts

    def _count_by_base_model(self, models: List[OptimizedModelEntry]) -> Dict[str, int]:
        """Count models by base model."""
        counts = {}
        for model in models:
            counts[model.base_model] = counts.get(model.base_model, 0) + 1
        return counts
