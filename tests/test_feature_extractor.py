"""Tests for computer vision feature extraction."""

from __future__ import annotations

import unittest

import numpy as np

from core.features import FeatureExtractor, FeatureExtractorConfig, extract_features


class FeatureExtractorTestCase(unittest.TestCase):
    """Validate color, texture, and shape feature extraction."""

    def setUp(self) -> None:
        """Create a deterministic synthetic BGR image."""
        height, width = 96, 128
        y = np.arange(height, dtype=np.uint8)[:, None]
        x = np.arange(width, dtype=np.uint8)[None, :]
        self.image = np.stack(
            [
                np.tile(x, (height, 1)),
                np.tile(y, (1, width)),
                (
                    (np.tile(x, (height, 1)) // 2)
                    + (np.tile(y, (1, width)) // 2)
                ).astype(np.uint8),
            ],
            axis=2,
        )
        self.gray_image = self.image[:, :, 0]
        self.extractor = FeatureExtractor(
            FeatureExtractorConfig(rgb_bins=16, hsv_bins=16),
        )

    def test_rgb_histogram_length_and_type(self) -> None:
        """RGB/BGR histogram should have 3 * bins values."""
        feature = self.extractor.extract(self.image, ["rgb_hist"])

        self.assertIsInstance(feature, np.ndarray)
        self.assertEqual(feature.dtype, np.float32)
        self.assertEqual(feature.shape, (48,))

    def test_rgb_histogram_uses_rgb_channel_order(self) -> None:
        """RGB histogram should concatenate R, G, B channel histograms."""
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :, 2] = 255
        extractor = FeatureExtractor(
            FeatureExtractorConfig(rgb_bins=2, normalize_output=False),
        )

        feature = extractor.extract(image, ["rgb_hist"])

        self.assertEqual(int(np.argmax(feature[:2])), 1)
        self.assertEqual(int(np.argmax(feature[2:4])), 0)
        self.assertEqual(int(np.argmax(feature[4:6])), 0)
    def test_hsv_histogram_custom_bins_length(self) -> None:
        """HSV histogram should accept method-level bin count."""
        feature = self.extractor.extract(self.image, ["hsv_hist:8"])

        self.assertEqual(feature.shape, (24,))
        self.assertEqual(feature.dtype, np.float32)

    def test_lbp_length_and_gray_support(self) -> None:
        """LBP should work with grayscale images and return fixed length."""
        feature = self.extractor.extract(self.gray_image, ["lbp"])

        self.assertEqual(feature.shape, (10,))
        self.assertEqual(feature.dtype, np.float32)

    def test_glcm_length_and_type(self) -> None:
        """GLCM should return properties across configured angles/distances."""
        feature = self.extractor.extract(self.image, ["glcm"])

        self.assertEqual(feature.shape, (24,))
        self.assertEqual(feature.dtype, np.float32)

    def test_hog_returns_stable_vector(self) -> None:
        """HOG should resize internally and return a stable-length vector."""
        feature = self.extractor.extract(self.image, ["hog"])

        self.assertIsInstance(feature, np.ndarray)
        self.assertEqual(feature.dtype, np.float32)
        self.assertGreater(feature.shape[0], 0)

    def test_combined_features_are_concatenated(self) -> None:
        """Multiple methods should concatenate into one normalized vector."""
        feature = self.extractor.extract(
            self.image,
            ["rgb_hist", "hsv_hist:8", "lbp", "glcm"],
        )

        self.assertEqual(feature.shape, (48 + 24 + 10 + 24,))
        self.assertAlmostEqual(float(np.linalg.norm(feature)), 1.0, places=5)

    def test_repeated_method_reuses_same_result(self) -> None:
        """Repeated methods should produce repeated identical vector sections."""
        feature = self.extractor.extract(self.image, ["rgb_hist:8", "rgb_hist:8"])

        self.assertEqual(feature.shape, (48,))
        self.assertTrue(np.allclose(feature[:24], feature[24:]))

    def test_convenience_function(self) -> None:
        """Top-level convenience function should return a feature vector."""
        feature = extract_features(self.image, ["rgb_hist:4"])

        self.assertEqual(feature.shape, (12,))

    def test_unknown_method_raises_error(self) -> None:
        """Unsupported methods should fail clearly."""
        with self.assertRaises(ValueError):
            self.extractor.extract(self.image, ["flatten"])


if __name__ == "__main__":
    unittest.main()