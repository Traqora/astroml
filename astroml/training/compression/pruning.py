from astroml.utils.exceptions import AstroMLError
"""Model weight pruning framework supporting unstructured and structured pruning strategies.

Supports L1/L2 norm unstructured magnitude pruning, structured channel/layer pruning,
global pruning, and sparsity evaluation.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)


class PruningMethod(Enum):
    """Supported pruning algorithms."""

    L1_UNSTRUCTURED = "l1_unstructured"
    RANDOM_UNSTRUCTURED = "random_unstructured"
    LN_STRUCTURED = "ln_structured"
    GLOBAL_UNSTRUCTURED = "global_unstructured"


@dataclass
class PruningConfig:
    """Configuration for model pruning."""

    method: PruningMethod = PruningMethod.L1_UNSTRUCTURED
    amount: float = 0.3  # Fraction of weights to prune (0.0 to 1.0)
    target_layer_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv1d, nn.Conv2d)
    norm_n: int = 1  # 1 for L1, 2 for L2
    dim: int = 0  # Dimension for structured pruning


class ModelPruner:
    """Pruning engine for reducing model parameter count and size."""

    def __init__(self, config: PruningConfig | None = None) -> None:
        """Initialize pruner with configuration."""
        self.config = config or PruningConfig()

    def prune_unstructured(
        self,
        model: nn.Module,
        amount: float = 0.3,
        target_layer_types: tuple[type[nn.Module], ...] | None = None,
        make_permanent: bool = True,
    ) -> nn.Module:
        """Apply L1 unstructured magnitude pruning to target layers."""
        types = target_layer_types or self.config.target_layer_types
        pruned_model = copy.deepcopy(model)

        for name, module in pruned_model.named_modules():
            if (
                isinstance(module, types)
                and hasattr(module, "weight")
                and module.weight is not None
            ):
                prune.l1_unstructured(module, name="weight", amount=amount)
                if make_permanent:
                    prune.remove(module, "weight")

        sparsity = self.compute_sparsity(pruned_model)
        logger.info(
            "Unstructured pruning completed: target amount=%.2f, overall sparsity=%.2f%%",
            amount,
            sparsity * 100,
        )
        return pruned_model

    def prune_structured(
        self,
        model: nn.Module,
        amount: float = 0.3,
        dim: int = 0,
        n: int = 1,
        target_layer_types: tuple[type[nn.Module], ...] | None = None,
        make_permanent: bool = True,
    ) -> nn.Module:
        """Apply Ln-norm structured pruning along specified dimension."""
        types = target_layer_types or self.config.target_layer_types
        pruned_model = copy.deepcopy(model)

        for name, module in pruned_model.named_modules():
            if (
                isinstance(module, types)
                and hasattr(module, "weight")
                and module.weight is not None
            ):
                prune.ln_structured(module, name="weight", amount=amount, n=n, dim=dim)
                if make_permanent:
                    prune.remove(module, "weight")

        sparsity = self.compute_sparsity(pruned_model)
        logger.info(
            "Structured pruning completed: target amount=%.2f, overall sparsity=%.2f%%",
            amount,
            sparsity * 100,
        )
        return pruned_model

    def prune_global(
        self,
        model: nn.Module,
        amount: float = 0.3,
        target_layer_types: tuple[type[nn.Module], ...] | None = None,
        make_permanent: bool = True,
    ) -> nn.Module:
        """Apply global magnitude pruning across all parameters in target layers."""
        types = target_layer_types or self.config.target_layer_types
        pruned_model = copy.deepcopy(model)

        parameters_to_prune = []
        for name, module in pruned_model.named_modules():
            if (
                isinstance(module, types)
                and hasattr(module, "weight")
                and module.weight is not None
            ):
                parameters_to_prune.append((module, "weight"))

        if parameters_to_prune:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
            if make_permanent:
                for module, name in parameters_to_prune:
                    prune.remove(module, name)

        sparsity = self.compute_sparsity(pruned_model)
        logger.info(
            "Global pruning completed: amount=%.2f, sparsity=%.2f%%", amount, sparsity * 100
        )
        return pruned_model

    def compute_sparsity(self, model: nn.Module) -> float:
        """Compute the fraction of zero weights in the model (0.0 = dense, 1.0 = all zero)."""
        total_elements = 0
        zero_elements = 0

        for param in model.parameters():
            if param is not None:
                total_elements += param.numel()
                zero_elements += int((param == 0).sum().item())

        if total_elements == 0:
            for module in model.modules():
                if hasattr(module, "_packed_params"):
                    try:
                        w, _ = module._packed_params._weight_bias()
                        total_elements += w.numel()
                        zero_elements += int((w.dequantize() == 0).sum().item())
                    except AstroMLError:
                        pass

        return float(zero_elements / total_elements) if total_elements > 0 else 0.0

    def prune(
        self,
        model: nn.Module,
        config: PruningConfig | None = None,
    ) -> nn.Module:
        """Execute pruning according to configuration."""
        cfg = config or self.config
        if cfg.method == PruningMethod.L1_UNSTRUCTURED:
            return self.prune_unstructured(
                model, amount=cfg.amount, target_layer_types=cfg.target_layer_types
            )
        elif cfg.method == PruningMethod.LN_STRUCTURED:
            return self.prune_structured(
                model,
                amount=cfg.amount,
                dim=cfg.dim,
                n=cfg.norm_n,
                target_layer_types=cfg.target_layer_types,
            )
        elif cfg.method == PruningMethod.GLOBAL_UNSTRUCTURED:
            return self.prune_global(
                model, amount=cfg.amount, target_layer_types=cfg.target_layer_types
            )
        return model
