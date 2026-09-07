"""Image-to-feature vectorization pipeline for prediction workflows."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from core.features import FeatureExtractor, FeatureExtractorConfig
from core.preprocessing import ImagePreprocessor


class FeatureVectorizer(Protocol):
    """Protocol for objects that convert images to feature vectors."""

    def vectorize(
        self,
        image: np.ndarray,
        preprocessing_ops: list[str],
        feature_methods: list[str],
    ) -> np.ndarray:
        """Return a feature vector for an input image."""


class ImageFeatureVectorizer:
    """Apply preprocessing and feature extraction to uploaded images."""

    def __init__(
        self,
        preprocessor: ImagePreprocessor | None = None,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        """Create an image vectorizer with optional reusable components."""
        self._preprocessor = preprocessor or ImagePreprocessor()
        self._extractor = extractor or FeatureExtractor(FeatureExtractorConfig())

    def vectorize(
        self,
        image: np.ndarray,
        preprocessing_ops: list[str],
        feature_methods: list[str],
    ) -> np.ndarray:
        """Convert an image to the configured feature vector.

        Args:
            image: Input image as an OpenCV-style NumPy array.
            preprocessing_ops: Ordered preprocessing operations.
            feature_methods: Ordered feature extraction methods.

        Returns:
            A one-dimensional NumPy feature vector.
        """
        if not feature_methods:
            msg = "feature_methods must contain at least one method."
            raise ValueError(msg)
        processed = self._preprocessor.process(image, preprocessing_ops)
        return self._extractor.extract(processed, feature_methods)