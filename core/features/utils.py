"""Shared utilities for feature extraction."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

FloatVector = NDArray[np.float32]


def validate_image(image: np.ndarray) -> np.ndarray:
    """Validate a grayscale or BGR image array."""
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


def ensure_uint8(image: np.ndarray) -> NDArray[np.uint8]:
    """Convert numeric image arrays to uint8 for OpenCV/skimage feature APIs."""
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating) and float(np.max(image)) <= 1.0:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def ensure_bgr(image: np.ndarray) -> NDArray[np.uint8]:
    """Return a 3-channel BGR uint8 image from grayscale or color input."""
    uint8_image = ensure_uint8(validate_image(image))
    if uint8_image.ndim == 2:
        return cv2.cvtColor(uint8_image, cv2.COLOR_GRAY2BGR)
    if uint8_image.shape[2] == 1:
        return cv2.cvtColor(uint8_image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if uint8_image.shape[2] == 4:
        return cv2.cvtColor(uint8_image, cv2.COLOR_BGRA2BGR)
    return uint8_image


def ensure_gray(image: np.ndarray) -> NDArray[np.uint8]:
    """Return a single-channel grayscale uint8 image from any supported input."""
    uint8_image = ensure_uint8(validate_image(image))
    if uint8_image.ndim == 2:
        return uint8_image
    if uint8_image.shape[2] == 1:
        return uint8_image[:, :, 0]
    if uint8_image.shape[2] == 4:
        bgr_image = cv2.cvtColor(uint8_image, cv2.COLOR_BGRA2BGR)
    else:
        bgr_image = uint8_image
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)


def l1_normalize(vector: np.ndarray) -> FloatVector:
    """Return a float32 vector normalized by its L1 sum."""
    values = vector.astype(np.float32).ravel()
    total = float(np.sum(values))
    if total == 0.0:
        return values
    return values / total


def l2_normalize(vector: np.ndarray) -> FloatVector:
    """Return a float32 vector normalized by its L2 norm."""
    values = vector.astype(np.float32).ravel()
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        return values
    return values / norm