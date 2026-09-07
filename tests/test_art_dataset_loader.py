"""Tests for publication-grade art dataset loading and compliance checks."""

from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from experiments.art_dataset_loader import (
    ArtDatasetLoader,
    ArtDatasetLoaderConfig,
    DatasetFormat,
)
from experiments.dataset_compliance import DatasetStatus, check_dataset_compliance
from tests.temp_utils import ProjectTempDirectory


class ArtDatasetLoaderTestCase(unittest.TestCase):
    """Validate real-dataset loader behavior for supported layouts."""

    def setUp(self) -> None:
        """Create a project-local temporary dataset root."""
        self.temp_dir = ProjectTempDirectory("art_dataset_loader")
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up temporary dataset files."""
        self.temp_dir.cleanup()

    def test_wikiart_loader_reads_style_folders_and_standardizes_images(self) -> None:
        """WikiArt-style folders should produce labels and RGB resized images."""
        image_path = self._write_image(
            self.root / "Impressionism" / "Claude Monet" / "water_lilies.jpg",
        )

        loader = ArtDatasetLoader(
            ArtDatasetLoaderConfig(
                dataset_root=self.root,
                dataset_format=DatasetFormat.WIKIART,
                resize_to=(32, 24),
            ),
        )
        records = list(loader.iter_records())
        loaded = list(loader.iter_images())

        self.assertEqual(records[0].image_path, image_path)
        self.assertEqual(records[0].style_label, "Impressionism")
        self.assertEqual(records[0].artist_name, "Claude Monet")
        self.assertEqual(loaded[0].rgb_image.shape, (24, 32, 3))
        self.assertEqual(loaded[0].normalized_image.dtype, np.float32)
        self.assertGreaterEqual(float(loaded[0].normalized_image.min()), 0.0)
        self.assertLessEqual(float(loaded[0].normalized_image.max()), 1.0)

    def test_kaggle_loader_reads_metadata_csv(self) -> None:
        """Kaggle Painter by Numbers metadata should map images to style labels."""
        train_dir = self.root / "train"
        image_path = self._write_image(train_dir / "abc123.jpg")
        metadata = self.root / "train_info.csv"
        metadata.write_text(
            "filename,artist,style,title\n"
            "abc123.jpg,Test Artist,Cubism,Demo Painting\n",
            encoding="utf-8",
        )

        loader = ArtDatasetLoader(
            ArtDatasetLoaderConfig(
                dataset_root=self.root,
                dataset_format=DatasetFormat.KAGGLE_PAINTER_BY_NUMBERS,
                metadata_file=metadata,
            ),
        )
        records = list(loader.iter_records())

        self.assertEqual(records[0].image_path, image_path)
        self.assertEqual(records[0].style_label, "Cubism")
        self.assertEqual(records[0].artist_name, "Test Artist")
        self.assertEqual(records[0].title, "Demo Painting")

    def test_compliance_requires_real_data_thresholds(self) -> None:
        """Compliance should distinguish missing, real, and debug-only data."""
        missing = check_dataset_compliance(None)
        debug = check_dataset_compliance(None, debug_synthetic=True)
        for class_name in ("A", "B", "C"):
            for index in range(2):
                self._write_image(self.root / class_name / f"{index}.jpg")
        valid = check_dataset_compliance(
            self.root,
            min_classes=3,
            min_images_per_class=2,
        )

        self.assertEqual(missing.dataset_status, DatasetStatus.INSUFFICIENT_DATA)
        self.assertEqual(debug.dataset_status, DatasetStatus.DEBUG_ONLY)
        self.assertEqual(valid.dataset_status, DatasetStatus.VALID_REAL_DATA)
        self.assertTrue(valid.is_publication_ready)

    def _write_image(self, image_path: Path) -> Path:
        """Write a small valid image file for loader tests."""
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((12, 10, 3), 128, dtype=np.uint8)
        if not cv2.imwrite(str(image_path), image):
            msg = f"Failed to write test image: {image_path}"
            raise RuntimeError(msg)
        return image_path


if __name__ == "__main__":
    unittest.main()
