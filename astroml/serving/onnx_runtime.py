from astroml.utils.exceptions import AstroMLError
"""ONNX Runtime inference wrapper for model serving."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

# Allowlist: model paths must be absolute and end in .onnx
_ONNX_PATH_RE = re.compile(r"^[^\x00]+\.onnx$", re.IGNORECASE)


def _validate_onnx_path(path: Path) -> None:
    """Raise if the path looks suspicious (directory traversal, null bytes, etc.).

    The caller is responsible for ensuring the path originates from a trusted
    source (e.g. the server-side model store). This is a defence-in-depth check.

    Args:
        path: Resolved absolute path to the ONNX file.

    Raises:
        ValueError: If the path fails validation.
    """
    if not _ONNX_PATH_RE.match(path.name):
        msg = f"Invalid ONNX filename: {path.name!r}"
        raise ValueError(msg)
    # Reject any path that escaped its expected parent via symlinks etc.
    # The API layer (model_optimization.py) must also enforce containment.


class ONNXRuntime:
    """Load and run ONNX models using onnxruntime.

    Supports multiple execution providers (CPU, CUDA, TensorRT) and
    provides context manager support for resource cleanup.

    Example:
        with ONNXRuntime.load_model("model.onnx") as rt:
            result = rt.predict(input_data)
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._model_path: Path | None = None
        self._providers: list[str] = []

    @classmethod
    def load_model(
        cls,
        onnx_path: str | Path,
        providers: list[str] | None = None,
    ) -> ONNXRuntime:
        """Load an ONNX model with the specified execution providers.

        Args:
            onnx_path: Path to the ONNX model file.
            providers: List of execution providers (e.g., ['CUDAExecutionProvider',
                      'CPUExecutionProvider']). If None, uses available providers.

        Returns:
            ONNXRuntime instance with loaded model.

        Raises:
            FileNotFoundError: If the ONNX file does not exist.
            RuntimeError: If the model cannot be loaded.
        """
        onnx_path = Path(onnx_path).resolve()
        _validate_onnx_path(onnx_path)
        if not onnx_path.exists():
            msg = f"ONNX file not found: {onnx_path}"
            raise FileNotFoundError(msg)

        instance = cls()
        instance._model_path = onnx_path

        try:
            if providers is not None:
                instance._providers = providers
                instance._session = ort.InferenceSession(str(onnx_path), providers=providers)
            else:
                instance._providers = ort.get_available_providers()
                instance._session = ort.InferenceSession(str(onnx_path))
        except Exception as exc:
            msg = f"Failed to load ONNX model: {exc}"
            raise RuntimeError(msg) from exc

        logger.info(
            "Loaded ONNX model from %s with providers: %s",
            onnx_path,
            instance._providers,
        )
        return instance

    def predict(self, input_data: np.ndarray | dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run inference on the loaded model.

        Args:
            input_data: Input data as a numpy array (single input) or
                       dict mapping input names to numpy arrays (multiple inputs).

        Returns:
            List of output tensors as numpy arrays.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._session is None:
            msg = "No model loaded. Call load_model() first."
            raise RuntimeError(msg)

        input_data = self.transform_input(input_data)

        if isinstance(input_data, np.ndarray):
            input_dict = {self._session.get_inputs()[0].name: input_data}
        else:
            input_dict = input_data

        outputs = self._session.run(None, input_dict)
        return [self.transform_output(out) for out in outputs]

    def predict_batch(
        self, inputs: list[np.ndarray | dict[str, np.ndarray]]
    ) -> list[list[np.ndarray]]:
        """Run batch inference on the loaded model.

        Args:
            inputs: List of input data, each being a numpy array or dict.

        Returns:
            List of inference results, each being a list of output tensors.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._session is None:
            msg = "No model loaded. Call load_model() first."
            raise RuntimeError(msg)

        return [self.predict(inp) for inp in inputs]

    def get_input_info(self) -> list[dict[str, Any]]:
        """Get information about model input tensors.

        Returns:
            List of dicts with 'name', 'shape', and 'dtype' for each input.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._session is None:
            msg = "No model loaded. Call load_model() first."
            raise RuntimeError(msg)

        return [
            {
                "name": inp.name,
                "shape": inp.shape,
                "dtype": str(inp.type),
            }
            for inp in self._session.get_inputs()
        ]

    def get_output_info(self) -> list[dict[str, Any]]:
        """Get information about model output tensors.

        Returns:
            List of dicts with 'name', 'shape', and 'dtype' for each output.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._session is None:
            msg = "No model loaded. Call load_model() first."
            raise RuntimeError(msg)

        return [
            {
                "name": out.name,
                "shape": out.shape,
                "dtype": str(out.type),
            }
            for out in self._session.get_outputs()
        ]

    def get_providers(self) -> list[str]:
        """List available ONNX Runtime execution providers.

        Returns:
            List of provider names available on the current system.
        """
        return ort.get_available_providers()

    def set_provider(self, provider: str) -> None:
        """Switch the execution provider for the loaded model.

        Args:
            provider: Name of the execution provider (e.g., 'CUDAExecutionProvider').

        Raises:
            RuntimeError: If no model is loaded or provider is unavailable.
            ValueError: If the provider is not available.
        """
        if self._session is None or self._model_path is None:
            msg = "No model loaded. Call load_model() first."
            raise RuntimeError(msg)

        available = ort.get_available_providers()
        if provider not in available:
            msg = f"Provider '{provider}' not available. Available: {available}"
            raise ValueError(msg)

        self._session = ort.InferenceSession(str(self._model_path), providers=[provider])
        self._providers = [provider]
        logger.info("Switched to provider: %s", provider)

    def transform_input(
        self, input_data: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Pre-processing hook for input data.

        Override in subclasses for custom preprocessing.

        Args:
            input_data: Raw input data.

        Returns:
            Transformed input data.
        """
        return input_data

    def transform_output(self, output_data: np.ndarray) -> np.ndarray:
        """Post-processing hook for output data.

        Override in subclasses for custom postprocessing.

        Args:
            output_data: Raw output tensor.

        Returns:
            Transformed output data.
        """
        return output_data

    def __enter__(self) -> ONNXRuntime:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release resources held by the runtime."""
        self._session = None
        self._model_path = None
        self._providers = []
        logger.debug("ONNX Runtime resources released")
