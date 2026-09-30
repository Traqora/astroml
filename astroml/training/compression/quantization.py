from astroml.utils.exceptions import AstroMLError
"""Model quantization utilities supporting Post-Training Quantization (PTQ) and QAT.

Supports INT8 dynamic quantization, INT8 static calibration quantization,
FP16 half-precision casting, and Quantization-Aware Training (QAT).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Supported quantization strategies."""

    DYNAMIC_INT8 = "dynamic_int8"
    STATIC_INT8 = "static_int8"
    FP16 = "fp16"
    QAT = "qat"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    quant_type: QuantizationType = QuantizationType.DYNAMIC_INT8
    target_modules: set[type[nn.Module]] = field(
        default_factory=lambda: {nn.Linear, nn.LSTM, nn.GRU}
    )
    dtype: torch.dtype = torch.qint8
    backend: str = "fbgemm"
    calibration_batches: int = 10


class ModelQuantizer:
    """Quantizes PyTorch models for edge and latency-critical inference."""

    def __init__(self, config: QuantizationConfig | None = None) -> None:
        """Initialize quantizer with configuration."""
        self.config = config or QuantizationConfig()

    def quantize_dynamic(
        self,
        model: nn.Module,
        target_modules: set[type[nn.Module]] | None = None,
        dtype: torch.dtype = torch.qint8,
    ) -> nn.Module:
        """Apply dynamic INT8 post-training quantization to weights."""
        target_mods = target_modules or self.config.target_modules
        model_eval = copy.deepcopy(model).eval()
        try:
            quantized = torch.quantization.quantize_dynamic(
                model_eval,
                qconfig_spec=target_mods,
                dtype=dtype,
            )
            logger.info("Successfully applied dynamic quantization to model")
            return quantized
        except Exception as exc:
            logger.warning("Dynamic quantization failed: %s; returning original model", exc)
            return model_eval

    def quantize_fp16(self, model: nn.Module) -> nn.Module:
        """Cast model weights to half-precision FP16."""
        model_fp16 = copy.deepcopy(model).half()
        logger.info("Converted model to FP16 half-precision")
        return model_fp16

    def quantize_static(
        self,
        model: nn.Module,
        calibration_data: Any | None = None,
        backend: str = "fbgemm",
    ) -> nn.Module:
        """Apply static calibration INT8 quantization."""
        model_eval = copy.deepcopy(model).eval()
        try:
            # Set quantization backend
            torch.backends.quantized.engine = backend
            qconfig = torch.quantization.get_default_qconfig(backend)
            prepared = torch.quantization.prepare(model_eval, inplace=False)
            prepared.qconfig = qconfig

            # Calibrate if sample data is provided
            if calibration_data is not None:
                with torch.no_grad():
                    if isinstance(calibration_data, torch.Tensor):
                        prepared(calibration_data)
                    elif hasattr(calibration_data, "__iter__"):
                        for batch in calibration_data:
                            if isinstance(batch, (list, tuple)):
                                prepared(batch[0])
                            elif isinstance(batch, torch.Tensor):
                                prepared(batch)

            quantized = torch.quantization.convert(prepared, inplace=False)
            logger.info("Successfully applied static calibration quantization")
            return quantized
        except Exception as exc:
            logger.warning("Static quantization failed: %s; falling back to dynamic", exc)
            return self.quantize_dynamic(model)

    def prepare_qat(
        self,
        model: nn.Module,
        backend: str = "fbgemm",
    ) -> nn.Module:
        """Prepare model for Quantization-Aware Training with fake-quantization modules."""
        model_train = copy.deepcopy(model).train()
        try:
            torch.backends.quantized.engine = backend
            model_train.qconfig = torch.quantization.get_default_qat_qconfig(backend)
            prepared = torch.quantization.prepare_qat(model_train, inplace=False)
            logger.info("Prepared model for QAT fine-tuning")
            return prepared
        except Exception as exc:
            logger.warning("Prepare QAT failed: %s; returning original model", exc)
            return model_train

    def convert_qat(self, qat_model: nn.Module) -> nn.Module:
        """Convert a trained QAT model into an actual quantized integer model."""
        qat_eval = copy.deepcopy(qat_model).eval()
        try:
            quantized = torch.quantization.convert(qat_eval, inplace=False)
            logger.info("Successfully converted QAT model to quantized model")
            return quantized
        except Exception as exc:
            logger.warning("Convert QAT failed: %s; returning original model", exc)
            return qat_eval

    def quantize(
        self,
        model: nn.Module,
        calibration_data: Any | None = None,
        config: QuantizationConfig | None = None,
    ) -> nn.Module:
        """Execute quantization according to configured strategy."""
        cfg = config or self.config
        if cfg.quant_type == QuantizationType.DYNAMIC_INT8:
            return self.quantize_dynamic(model, target_modules=cfg.target_modules, dtype=cfg.dtype)
        elif cfg.quant_type == QuantizationType.STATIC_INT8:
            return self.quantize_static(
                model, calibration_data=calibration_data, backend=cfg.backend
            )
        elif cfg.quant_type == QuantizationType.FP16:
            return self.quantize_fp16(model)
        elif cfg.quant_type == QuantizationType.QAT:
            return self.prepare_qat(model, backend=cfg.backend)
        return model
