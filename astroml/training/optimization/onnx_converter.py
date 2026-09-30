from astroml.utils.exceptions import AstroMLError
"""ONNX model converter for PyTorch and scikit-learn models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import onnx
except ImportError:
    onnx = None  # type: ignore[assignment]

try:
    import skl2onnx
except ImportError:
    skl2onnx = None  # type: ignore[assignment]


class ONNXConverter:
    """Convert PyTorch and scikit-learn models to ONNX format."""

    SUPPORTED_OPSET_VERSIONS = range(7, 21)

    @staticmethod
    def convert(
        model: Any,
        input_sample: Any,
        output_path: str | Path,
        opset_version: int = 17,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
    ) -> Path:
        """Convert a PyTorch model to ONNX format.

        Args:
            model: The PyTorch model to convert.
            input_sample: Example input tensor(s) for tracing.
            output_path: Path where the ONNX model will be saved.
            opset_version: ONNX opset version to use (default: 17).
            dynamic_axes: Dictionary mapping tensor names to dynamic axes.

        Returns:
            Path to the saved ONNX model.

        Raises:
            ValueError: If the opset version is not supported.
            RuntimeError: If the conversion fails.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if opset_version not in ONNXConverter.SUPPORTED_OPSET_VERSIONS:
            msg = f"Unsupported opset version {opset_version}. Supported: {list(ONNXConverter.SUPPORTED_OPSET_VERSIONS)}"
            raise ValueError(msg)

        if dynamic_axes is None:
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        model.eval()

        try:
            torch.onnx.export(
                model,
                input_sample,
                str(output_path),
                opset_version=opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )
        except Exception as exc:
            msg = f"Failed to convert PyTorch model to ONNX: {exc}"
            raise RuntimeError(msg) from exc

        logger.info("Converted PyTorch model to ONNX at %s", output_path)
        return output_path

    @staticmethod
    def convert_from_sklearn(
        estimator: Any,
        output_path: str | Path,
    ) -> Path:
        """Convert a scikit-learn estimator to ONNX format.

        Args:
            estimator: The scikit-learn estimator to convert.
            output_path: Path where the ONNX model will be saved.

        Returns:
            Path to the saved ONNX model.

        Raises:
            ValueError: If the estimator type is not supported.
            RuntimeError: If the conversion fails.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            onnx_model = skl2onnx.convert_sklearn(estimator)
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
        except Exception as exc:
            msg = f"Failed to convert sklearn model to ONNX: {exc}"
            raise RuntimeError(msg) from exc

        logger.info("Converted sklearn model to ONNX at %s", output_path)
        return output_path

    @staticmethod
    def validate_onnx(onnx_path: str | Path) -> bool:
        """Validate an ONNX model using onnx.checker.

        Args:
            onnx_path: Path to the ONNX model file.

        Returns:
            True if the model is valid.

        Raises:
            FileNotFoundError: If the ONNX file does not exist.
            RuntimeError: If validation fails.
        """
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            msg = f"ONNX file not found: {onnx_path}"
            raise FileNotFoundError(msg)

        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
        except Exception as exc:
            msg = f"ONNX validation failed: {exc}"
            raise RuntimeError(msg) from exc

        logger.info("ONNX model validated successfully: %s", onnx_path)
        return True

    @staticmethod
    def get_model_metadata(onnx_path: str | Path) -> dict[str, Any]:
        """Get metadata from an ONNX model.

        Args:
            onnx_path: Path to the ONNX model file.

        Returns:
            Dict containing inputs, outputs, and opset version.

        Raises:
            FileNotFoundError: If the ONNX file does not exist.
        """
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            msg = f"ONNX file not found: {onnx_path}"
            raise FileNotFoundError(msg)

        onnx_model = onnx.load(str(onnx_path))

        inputs = [
            {
                "name": inp.name,
                "shape": [
                    dim.dim_param if dim.dim_param else dim.dim_value
                    for dim in inp.type.tensor_type.shape.dim
                ],
                "dtype": inp.type.tensor_type.elem_type,
            }
            for inp in onnx_model.graph.input
        ]

        outputs = [
            {
                "name": out.name,
                "shape": [
                    dim.dim_param if dim.dim_param else dim.dim_value
                    for dim in out.type.tensor_type.shape.dim
                ],
                "dtype": out.type.tensor_type.elem_type,
            }
            for out in onnx_model.graph.output
        ]

        opset = {imp.domain: imp.version for imp in onnx_model.opset_import}

        return {
            "inputs": inputs,
            "outputs": outputs,
            "opset": opset,
            "producer_name": onnx_model.producer_name,
            "producer_version": onnx_model.producer_version,
        }

    @staticmethod
    def convert_batch(
        models: dict[str, Any],
        input_samples: dict[str, Any],
        output_dir: str | Path,
        model_type: str = "pytorch",
        opset_version: int = 17,
    ) -> dict[str, Path]:
        """Convert multiple models to ONNX format.

        Args:
            models: Dict mapping model names to model instances.
            input_samples: Dict mapping model names to input samples.
            output_dir: Directory to save the ONNX models.
            model_type: Either "pytorch" or "sklearn".
            opset_version: ONNX opset version (PyTorch only).

        Returns:
            Dict mapping model names to their output paths.

        Raises:
            ValueError: If model_type is not supported.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Path] = {}

        for name, model in models.items():
            input_sample = input_samples.get(name)
            output_path = output_dir / f"{name}.onnx"

            if model_type == "pytorch":
                if input_sample is None:
                    logger.warning("Skipping %s: no input sample provided", name)
                    continue
                ONNXConverter.convert(
                    model,
                    input_sample,
                    output_path,
                    opset_version=opset_version,
                )
            elif model_type == "sklearn":
                ONNXConverter.convert_from_sklearn(model, output_path)
            else:
                msg = f"Unsupported model type: {model_type}. Use 'pytorch' or 'sklearn'."
                raise ValueError(msg)

            results[name] = output_path

        return results
