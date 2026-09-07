"""Concrete OpenCV image preprocessing operations."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
from numpy.typing import NDArray

from core.preprocessing.operation_parser import CropBox, OperationParser, ResizeSize


class NormalizeMode(str, Enum):
    """Supported normalization strategies."""

    ZERO_ONE = "0_1"
    Z_SCORE = "zscore"


class ImageOperations:
    """Implement concrete image transformations used by the preprocessor."""

    def __init__(self, parser: OperationParser) -> None:
        """Create image operations with shared parsing helpers."""
        self._parser = parser

    def resize(self, image: np.ndarray, size: ResizeSize) -> np.ndarray:
        """Resize directly to width x height using area/linear interpolation."""
        interpolation = (
            cv2.INTER_AREA
            if self._is_downscaling(image, size)
            else cv2.INTER_LINEAR
        )
        return cv2.resize(image, (size.width, size.height), interpolation=interpolation)

    def resize_keep_aspect(self, image: np.ndarray, size: ResizeSize) -> np.ndarray:
        """Resize while preserving aspect ratio within target dimensions."""
        height, width = image.shape[:2]
        scale = min(size.width / width, size.height / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return self.resize(image, ResizeSize(width=new_width, height=new_height))

    def resize_with_padding(self, image: np.ndarray, size: ResizeSize) -> np.ndarray:
        """Resize preserving aspect ratio and pad to exact target dimensions."""
        resized = self.resize_keep_aspect(image, size)
        canvas_shape = (
            (size.height, size.width, resized.shape[2])
            if resized.ndim == 3
            else (size.height, size.width)
        )
        canvas = np.zeros(canvas_shape, dtype=resized.dtype)
        top = (size.height - resized.shape[0]) // 2
        left = (size.width - resized.shape[1]) // 2
        canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        return canvas

    def normalize(self, image: np.ndarray, argument: str) -> NDArray[np.float32]:
        """Normalize image pixels to [0, 1] or z-score float32 values."""
        normalized = image.astype(np.float32)
        mode = NormalizeMode(argument.lower())
        if mode == NormalizeMode.ZERO_ONE:
            return normalized / 255.0
        mean = float(np.mean(normalized))
        std = float(np.std(normalized))
        return normalized - mean if std == 0.0 else (normalized - mean) / std

    def to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert a color image to grayscale; 2-D images are returned unchanged."""
        if image.ndim == 2:
            return image
        if image.shape[2] == 1:
            return image[:, :, 0]
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def crop(self, image: np.ndarray, argument: str) -> np.ndarray:
        """Crop by center size or explicit left,top,right,bottom coordinates."""
        if not argument:
            msg = "crop operation requires an argument."
            raise ValueError(msg)
        if argument.startswith("center:"):
            crop_box = self._center_crop_box(
                image,
                self._parser.parse_size(argument.removeprefix("center:")),
            )
        else:
            crop_box = self._parser.parse_crop_box(argument)
        self._validate_crop_box(image, crop_box)
        return image[
            crop_box.top : crop_box.bottom,
            crop_box.left : crop_box.right,
        ].copy()

    def gaussian_blur(self, image: np.ndarray, argument: str) -> np.ndarray:
        """Apply Gaussian blur with an odd positive kernel size."""
        kernel_size = self._parser.parse_positive_int(argument, "blur kernel size")
        if kernel_size % 2 == 0:
            msg = "Gaussian blur kernel size must be odd."
            raise ValueError(msg)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)

    def equalize(self, image: np.ndarray, argument: str) -> np.ndarray:
        """Apply histogram equalization or CLAHE to grayscale/luminance channel."""
        method = (argument or "hist").lower()
        working = (
            image
            if image.dtype == np.uint8
            else np.clip(image, 0, 255).astype(np.uint8)
        )
        if working.ndim == 2:
            return self._equalize_gray(working, method)

        ycrcb = cv2.cvtColor(working, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = self._equalize_gray(ycrcb[:, :, 0], method)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def _equalize_gray(self, image: np.ndarray, method: str) -> np.ndarray:
        """Equalize a single-channel uint8 image."""
        if method in {"hist", "histogram", ""}:
            return cv2.equalizeHist(image)
        if method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        msg = f"Unsupported equalize method: {method}"
        raise ValueError(msg)

    def _center_crop_box(self, image: np.ndarray, size: ResizeSize) -> CropBox:
        """Build a center crop box for the requested size."""
        height, width = image.shape[:2]
        left = (width - size.width) // 2
        top = (height - size.height) // 2
        return CropBox(
            left=left,
            top=top,
            right=left + size.width,
            bottom=top + size.height,
        )

    def _validate_crop_box(self, image: np.ndarray, crop_box: CropBox) -> None:
        """Validate that a crop box is inside image bounds and non-empty."""
        height, width = image.shape[:2]
        if crop_box.left < 0 or crop_box.top < 0:
            msg = "Crop coordinates must not be negative."
            raise ValueError(msg)
        if crop_box.right > width or crop_box.bottom > height:
            msg = "Crop coordinates exceed image bounds."
            raise ValueError(msg)
        if crop_box.right <= crop_box.left or crop_box.bottom <= crop_box.top:
            msg = "Crop coordinates must define a non-empty rectangle."
            raise ValueError(msg)

    def _is_downscaling(self, image: np.ndarray, size: ResizeSize) -> bool:
        """Return whether resize target is smaller than current dimensions."""
        height, width = image.shape[:2]
        return size.width < width or size.height < height