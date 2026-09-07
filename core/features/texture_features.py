"""Texture feature extraction with LBP and GLCM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from core.features.utils import FloatVector, ensure_gray, l1_normalize, l2_normalize


@dataclass(frozen=True)
class LBPConfig:
    """Configuration for Local Binary Pattern extraction."""

    points: int = 8
    radius: int = 1
    method: str = "uniform"


@dataclass(frozen=True)
class GLCMConfig:
    """Configuration for gray-level co-occurrence matrix statistics."""

    distances: tuple[int, ...] = (1,)
    angles: tuple[float, ...] = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
    levels: int = 32
    properties: tuple[str, ...] = (
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    )


class TextureFeatureExtractor:
    """Extract LBP and GLCM texture descriptors."""

    def __init__(
        self,
        lbp_config: LBPConfig | None = None,
        glcm_config: GLCMConfig | None = None,
    ) -> None:
        """Create a texture extractor with optional configuration."""
        self._lbp_config = lbp_config or LBPConfig()
        self._glcm_config = glcm_config or GLCMConfig()

    def lbp_histogram(self, image: np.ndarray) -> FloatVector:
        """Compute an LBP texture histogram.

        The image is converted to grayscale. Uniform LBP with default 8 points
        and radius 1 produces ``points + 2`` bins. The output is L1-normalized.
        """
        gray_image = ensure_gray(image)
        lbp = local_binary_pattern(
            gray_image,
            P=self._lbp_config.points,
            R=self._lbp_config.radius,
            method=self._lbp_config.method,
        )
        bin_count = self._lbp_config.points + 2
        histogram, _ = np.histogram(lbp.ravel(), bins=bin_count, range=(0, bin_count))
        return l1_normalize(histogram)

    def glcm_statistics(self, image: np.ndarray) -> FloatVector:
        """Compute GLCM statistical texture features.

        The grayscale image is quantized to ``levels`` gray values before GLCM
        calculation. For each configured property, values across distances and
        angles are flattened and concatenated, then L2-normalized.
        """
        config = self._glcm_config
        gray_image = ensure_gray(image)
        quantized = self._quantize_gray(gray_image, config.levels)
        matrix = graycomatrix(
            quantized,
            distances=config.distances,
            angles=config.angles,
            levels=config.levels,
            symmetric=True,
            normed=True,
        )
        features = [graycoprops(matrix, prop).ravel() for prop in config.properties]
        return l2_normalize(np.concatenate(features))

    def _quantize_gray(self, image: np.ndarray, levels: int) -> np.ndarray:
        """Quantize a uint8 grayscale image to the configured number of levels."""
        if levels <= 1 or levels > 256:
            msg = "GLCM levels must be in the interval [2, 256]."
            raise ValueError(msg)
        scale = 256 // levels
        return np.clip(image // scale, 0, levels - 1).astype(np.uint8)