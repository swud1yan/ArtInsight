"""Tests and usage examples for image preprocessing."""

from __future__ import annotations

import unittest

import numpy as np

from core.preprocessing import ImagePreprocessor, preprocess


class ImagePreprocessorTestCase(unittest.TestCase):
    """Validate operation parsing and image preprocessing behavior."""

    def setUp(self) -> None:
        """Create a deterministic synthetic BGR image."""
        height, width = 80, 120
        y = np.arange(height, dtype=np.uint8)[:, None]
        x = np.arange(width, dtype=np.uint8)[None, :]
        self.image = np.stack(
            [
                np.tile(x, (height, 1)),
                np.tile(y, (1, width)),
                np.full((height, width), 128, dtype=np.uint8),
            ],
            axis=2,
        )
        self.preprocessor = ImagePreprocessor()

    def test_resize_gray_and_normalize_pipeline(self) -> None:
        """Pipeline should follow operation order and return expected shape."""
        result = self.preprocessor.process(
            self.image,
            ["resize:64x32", "gray", "normalize:0_1"],
        )

        self.assertEqual(result.shape, (32, 64))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_center_crop_blur_and_equalize(self) -> None:
        """Center crop, blur, and equalize should preserve requested dimensions."""
        result = self.preprocessor.process(
            self.image,
            ["crop:center:40x30", "blur:5", "equalize:clahe"],
        )

        self.assertEqual(result.shape, (30, 40, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_coordinate_crop_and_resize_padding(self) -> None:
        """Coordinate crop followed by padded resize should hit exact target size."""
        result = preprocess(
            self.image,
            ["crop:10,10,70,50", "resize_pad:64x64"],
        )

        self.assertEqual(result.shape, (64, 64, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_zscore_normalization_has_zero_mean_for_nonconstant_image(self) -> None:
        """Z-score normalization should produce approximately zero mean."""
        result = self.preprocessor.process(self.image, ["normalize:zscore"])

        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(float(result.mean()), 0.0, places=5)

    def test_crop_out_of_bounds_raises_error(self) -> None:
        """Invalid crop coordinates should fail clearly."""
        with self.assertRaises(ValueError):
            self.preprocessor.process(self.image, ["crop:0,0,999,999"])

    def test_even_blur_kernel_raises_error(self) -> None:
        """Gaussian blur requires odd kernel sizes."""
        with self.assertRaises(ValueError):
            self.preprocessor.process(self.image, ["blur:4"])


if __name__ == "__main__":
    unittest.main()