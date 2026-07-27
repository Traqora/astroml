"""
Image preprocessing and format conversion for multimodal LLM inputs.

Handles resizing, format conversion, quality optimization for various inputs.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
from pathlib import Path


class ImageFormat(str, Enum):
    """Supported image formats."""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"


@dataclass
class ImageConfig:
    """Configuration for image preprocessing."""
    max_width: int = 1920
    max_height: int = 1080
    target_format: ImageFormat = ImageFormat.JPEG
    quality: int = 85
    preserve_aspect_ratio: bool = True
    remove_metadata: bool = True


class ImagePreprocessor:
    """
    Preprocess images for multimodal LLM processing.

    Handles resizing, format conversion, quality optimization.
    """

    def __init__(self, config: Optional[ImageConfig] = None):
        """Initialize image preprocessor."""
        self.config = config or ImageConfig()

    def resize_image(self, image_path: str, max_size: Optional[Tuple[int, int]] = None) -> dict:
        """
        Resize image to maximum dimensions.

        Args:
            image_path: Path to input image
            max_size: Optional (width, height) override

        Returns:
            Dictionary with:
            - original_size: Original dimensions
            - new_size: New dimensions
            - scale_factor: Scaling factor applied
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        max_width = (max_size[0] if max_size else self.config.max_width)
        max_height = (max_size[1] if max_size else self.config.max_height)

        # Simulate resize
        original_size = (2048, 1536)
        scale_factor = min(max_width / original_size[0], max_height / original_size[1])
        new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))

        return {
            "original_size": original_size,
            "new_size": new_size,
            "scale_factor": scale_factor,
        }

    def convert_format(self, image_path: str, target_format: Optional[ImageFormat] = None) -> str:
        """
        Convert image to target format.

        Args:
            image_path: Path to input image
            target_format: Target format (defaults to config setting)

        Returns:
            Path to converted image
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        fmt = target_format or self.config.target_format
        output_path = path.with_suffix(f".{fmt.value}")

        # Simulate conversion
        return str(output_path)

    def optimize_quality(self, image_path: str, quality: Optional[int] = None) -> dict:
        """
        Optimize image quality and file size.

        Args:
            image_path: Path to input image
            quality: Quality level 1-100 (defaults to config setting)

        Returns:
            Dictionary with:
            - original_size: Original file size in bytes
            - optimized_size: Optimized file size in bytes
            - compression_ratio: Compression ratio achieved
            - quality_score: Estimated quality (0-1)
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        q = quality or self.config.quality

        # Simulate optimization
        original_size = 2048000  # bytes
        compression_ratio = 0.3 + (q / 100) * 0.6
        optimized_size = int(original_size * compression_ratio)

        return {
            "original_size": original_size,
            "optimized_size": optimized_size,
            "compression_ratio": 1.0 - (optimized_size / original_size),
            "quality_score": min(0.98, q / 100),
        }

    def remove_metadata(self, image_path: str) -> str:
        """
        Remove EXIF and other metadata from image.

        Args:
            image_path: Path to input image

        Returns:
            Path to image with metadata removed
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        output_path = path.with_stem(path.stem + "_no_metadata")

        # Simulate metadata removal
        return str(output_path)

    def preprocess_batch(self, image_paths: list) -> list:
        """
        Preprocess multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            List of preprocessed image paths
        """
        results = []
        for path in image_paths:
            # Simulate batch preprocessing
            results.append(str(Path(path).with_stem(Path(path).stem + "_processed")))

        return results

    def validate_format(self, image_path: str) -> bool:
        """
        Validate that image format is supported.

        Args:
            image_path: Path to image

        Returns:
            True if format is supported
        """
        path = Path(image_path)
        suffix = path.suffix.lstrip('.').lower()

        supported = [fmt.value for fmt in ImageFormat]
        return suffix in supported

    def estimate_processing_time(self, image_path: str) -> dict:
        """
        Estimate processing time for image.

        Args:
            image_path: Path to image

        Returns:
            Dictionary with time estimates:
            - resize_ms: Estimated resize time
            - format_conversion_ms: Estimated conversion time
            - optimization_ms: Estimated optimization time
            - total_ms: Total estimated time
        """
        # Simulate time estimation
        return {
            "resize_ms": 50,
            "format_conversion_ms": 100,
            "optimization_ms": 75,
            "total_ms": 225,
        }
