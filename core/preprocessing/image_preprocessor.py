"""OpenCV-based image preprocessing orchestration.

The preprocessor accepts NumPy image arrays and applies operation strings in
order. Color images are expected to use OpenCV's BGR channel order. Operations
that keep image intensity as byte images return ``np.uint8``; normalization
operations return ``np.float32``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.preprocessing.image_operations import ImageOperations
from core.preprocessing.operation_parser import OperationParser

ImageArray = NDArray[np.uint8] | NDArray[np.float32]


class ImagePreprocessor:
    """Apply common preprocessing operations to OpenCV image arrays."""

    def __init__(self, parser: OperationParser | None = None) -> None:
        """Create a preprocessor with an operation parser."""
        self._parser = parser or OperationParser()
        self._operations = ImageOperations(self._parser)

    def process(self, image: np.ndarray, ops: list[str]) -> np.ndarray:
        """Apply preprocessing operations in order.

        Args:
            image: Input image as a NumPy array. Color images are documented as
                BGR, matching OpenCV convention. Grayscale images may be 2-D.
            ops: Ordered operation strings. Supported formats are:
                ``resize:224x224`` for direct resize,
                ``resize_keep:224x224`` for aspect-ratio fit,
                ``resize_pad:224x224`` for aspect-ratio fit with black padding,
                ``normalize`` or ``normalize:0_1`` for [0, 1] float32 scaling,
                ``normalize:zscore`` for standard score normalization,
                ``gray`` for BGR/RGB-like image to grayscale conversion,
                ``crop:center:224x224`` for center crop,
                ``crop:100,100,300,300`` for coordinate crop,
                ``blur:5`` or ``gaussian_blur:5`` for Gaussian blur,
                ``equalize`` for histogram equalization,
                ``equalize:clahe`` for CLAHE enhancement.

        Returns:
            The processed image. Normalization returns ``np.float32``; other
            operations preserve or convert to OpenCV-compatible image arrays.
        """
        processed = self._validate_image(image).copy()
        for operation in ops:
            processed = self._apply_operation(processed, operation)
        return processed

    def _apply_operation(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Dispatch one operation string to its implementation."""
        spec = self._parser.split_operation(operation)
        if spec.name == "resize":
            return self._operations.resize(
                image,
                self._parser.parse_size(spec.argument),
            )
        if spec.name == "resize_keep":
            return self._operations.resize_keep_aspect(
                image,
                self._parser.parse_size(spec.argument),
            )
        if spec.name == "resize_pad":
            return self._operations.resize_with_padding(
                image,
                self._parser.parse_size(spec.argument),
            )
        if spec.name == "normalize":
            return self._operations.normalize(image, spec.argument or "0_1")
        if spec.name == "gray":
            return self._operations.to_gray(image)
        if spec.name == "crop":
            return self._operations.crop(image, spec.argument)
        if spec.name in {"blur", "gaussian_blur"}:
            return self._operations.gaussian_blur(image, spec.argument)
        if spec.name == "equalize":
            return self._operations.equalize(image, spec.argument)
        msg = f"Unsupported preprocessing operation: {operation}"
        raise ValueError(msg)

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validate supported image array shape and dtype."""
        if not isinstance(image, np.ndarray):
            msg = "image must be a NumPy ndarray."
            raise TypeError(msg)
        if image.ndim not in {2, 3}:
            msg = "image must be a 2-D grayscale or 3-D color array."
            raise ValueError(msg)
        if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
            msg = "color image must have 1, 3, or 4 channels."
            raise ValueError(msg)
        if image.size == 0:
            msg = "image must not be empty."
            raise ValueError(msg)
        if not np.issubdtype(image.dtype, np.number):
            msg = "image dtype must be numeric."
            raise TypeError(msg)
        return image


def preprocess(image: np.ndarray, operations: list[str]) -> np.ndarray:
    """Convenience function for applying image preprocessing operations."""
    return ImagePreprocessor().process(image, operations)