"""Shape and edge feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.feature import hog

from core.features.utils import FloatVector, ensure_gray, l2_normalize


@dataclass(frozen=True)
class HOGConfig:
    """Configuration for Histogram of Oriented Gradients extraction."""

    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (8, 8)
    cells_per_block: tuple[int, int] = (2, 2)
    resize_to: tuple[int, int] = (128, 128)


class ShapeFeatureExtractor:
    """Extract edge and shape descriptors such as HOG."""

    def __init__(self, hog_config: HOGConfig | None = None) -> None:
        """Create a shape feature extractor."""
        self._hog_config = hog_config or HOGConfig()

    def hog_features(self, image: np.ndarray) -> FloatVector:
        """Compute HOG features from a grayscale resized image.

        The image is converted to grayscale and resized to ``resize_to`` before
        HOG extraction so the returned vector has stable length across images.
        The output is L2-normalized and stored as ``np.float32``.
        """
        import cv2

        config = self._hog_config
        gray_image = ensure_gray(image)
        resized = cv2.resize(gray_image, config.resize_to, interpolation=cv2.INTER_AREA)
        features = hog(
            resized,
            orientations=config.orientations,
            pixels_per_cell=config.pixels_per_cell,
            cells_per_block=config.cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return l2_normalize(features)