"""Color histogram feature extraction."""

from __future__ import annotations

import cv2
import numpy as np

from core.features.utils import FloatVector, ensure_bgr, l1_normalize


class ColorFeatureExtractor:
    """Extract RGB/BGR and HSV color distribution features."""

    def rgb_histogram(self, image: np.ndarray, bins: int) -> FloatVector:
        """Compute a per-channel color histogram and concatenate channels.

        The input image is treated as OpenCV BGR and converted to RGB before
        histogram extraction. Histograms are concatenated in R, G, B order and
        L1-normalized as one vector. The vector length is ``3 * bins``.
        """
        self._validate_bins(bins)
        rgb_image = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2RGB)
        histograms = [
            cv2.calcHist([rgb_image], [channel], None, [bins], [0, 256]).ravel()
            for channel in range(3)
        ]
        return l1_normalize(np.concatenate(histograms))

    def hsv_histogram(self, image: np.ndarray, bins: int) -> FloatVector:
        """Compute HSV channel histograms and concatenate them.

        The image is converted from BGR to HSV. H uses range [0, 180], while S
        and V use [0, 256]. The vector length is ``3 * bins``.
        """
        self._validate_bins(bins)
        hsv_image = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2HSV)
        ranges = [(0, 180), (0, 256), (0, 256)]
        histograms = [
            cv2.calcHist(
                [hsv_image],
                [channel],
                None,
                [bins],
                list(value_range),
            ).ravel()
            for channel, value_range in enumerate(ranges)
        ]
        return l1_normalize(np.concatenate(histograms))

    def _validate_bins(self, bins: int) -> None:
        """Validate histogram bin count."""
        if bins <= 0:
            msg = "Histogram bins must be positive."
            raise ValueError(msg)