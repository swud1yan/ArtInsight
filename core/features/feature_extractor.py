"""Unified image feature extraction interface."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from core.features.color_features import ColorFeatureExtractor
from core.features.shape_features import HOGConfig, ShapeFeatureExtractor
from core.features.texture_features import (
    GLCMConfig,
    LBPConfig,
    TextureFeatureExtractor,
)
from core.features.utils import FloatVector, l2_normalize, validate_image

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureExtractorConfig:
    """Configuration for the unified feature extractor."""

    rgb_bins: int = 32
    hsv_bins: int = 32
    lbp: LBPConfig = LBPConfig()
    glcm: GLCMConfig = GLCMConfig()
    hog: HOGConfig = HOGConfig()
    normalize_output: bool = True


class FeatureExtractor:
    """Extract and concatenate configured image feature vectors."""

    def __init__(self, config: FeatureExtractorConfig | None = None) -> None:
        """Create a feature extractor with configurable method parameters."""
        self._config = config or FeatureExtractorConfig()
        self._color_extractor = ColorFeatureExtractor()
        self._texture_extractor = TextureFeatureExtractor(
            self._config.lbp,
            self._config.glcm,
        )
        self._shape_extractor = ShapeFeatureExtractor(self._config.hog)

    def extract(self, image: np.ndarray, methods: list[str]) -> FloatVector:
        """Extract selected features and concatenate them into one vector.

        Args:
            image: Grayscale or BGR NumPy image array. BGR follows OpenCV
                convention; grayscale input is automatically promoted when a
                color method needs three channels.
            methods: Ordered feature methods. Supported strings are ``rgb_hist``
                or ``rgb_hist:64``, ``hsv_hist`` or ``hsv_hist:64``, ``lbp``,
                ``hog``, and ``glcm``.

        Returns:
            A one-dimensional ``np.float32`` vector. Individual sub-features are
            normalized, and the final concatenated vector is L2-normalized by
            default.
        """
        validate_image(image)
        if not methods:
            msg = "methods must contain at least one feature method."
            raise ValueError(msg)

        cache: dict[str, FloatVector] = {}
        feature_parts: list[FloatVector] = []
        for method in methods:
            method_name, argument = self._split_method(method)
            cache_key = f"{method_name}:{argument}"
            if cache_key not in cache:
                cache[cache_key] = self._extract_one(image, method_name, argument)
                LOGGER.debug(
                    "Feature %s length=%s",
                    cache_key,
                    cache[cache_key].shape[0],
                )
            feature_parts.append(cache[cache_key])

        combined = np.concatenate(feature_parts).astype(np.float32)
        if self._config.normalize_output:
            return l2_normalize(combined)
        return combined

    def _extract_one(
        self,
        image: np.ndarray,
        method_name: str,
        argument: str,
    ) -> FloatVector:
        """Extract one feature vector by method name."""
        if method_name == "rgb_hist":
            return self._color_extractor.rgb_histogram(
                image,
                self._parse_bins(argument, self._config.rgb_bins),
            )
        if method_name == "hsv_hist":
            return self._color_extractor.hsv_histogram(
                image,
                self._parse_bins(argument, self._config.hsv_bins),
            )
        if method_name == "lbp":
            return self._texture_extractor.lbp_histogram(image)
        if method_name == "glcm":
            return self._texture_extractor.glcm_statistics(image)
        if method_name == "hog":
            return self._shape_extractor.hog_features(image)
        msg = f"Unsupported feature method: {method_name}"
        raise ValueError(msg)

    def _split_method(self, method: str) -> tuple[str, str]:
        """Split a method string into method name and optional argument."""
        if not method or not method.strip():
            msg = "Feature method must be a non-empty string."
            raise ValueError(msg)
        name, separator, argument = method.strip().partition(":")
        return name.lower(), argument if separator else ""

    def _parse_bins(self, argument: str, default_bins: int) -> int:
        """Parse optional histogram bin count."""
        if not argument:
            return default_bins
        try:
            bins = int(argument)
        except ValueError as exc:
            msg = "Histogram bins argument must be an integer."
            raise ValueError(msg) from exc
        if bins <= 0:
            msg = "Histogram bins must be positive."
            raise ValueError(msg)
        return bins


def extract_features(image: np.ndarray, methods: list[str]) -> FloatVector:
    """Convenience function for extracting concatenated image features."""
    return FeatureExtractor().extract(image, methods)