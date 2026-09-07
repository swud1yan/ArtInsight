"""Tests for the research-grade real dataset pipeline modules."""

from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from experiments.art_dataset import ArtDataset
from experiments.art_dataset_loader import ArtDatasetLoader, ArtDatasetLoaderConfig
from experiments.data_cleaner import CleaningConfig, DataCleaner
from experiments.dataset_compliance import DatasetStatus, check_dataset_validity
from experiments.dataset_downloader import (
    DatasetDownloadRequest,
    DatasetDownloader,
    DatasetSource,
)
from experiments.dataset_stats import DatasetStatsCollector
from experiments.dataset_structuring import DatasetStructurer, StructuringConfig
from tests.temp_utils import ProjectTempDirectory


class DatasetPipelineTestCase(unittest.TestCase):
    """Validate acquisition, cleaning, structuring, stats, and loading."""

    def setUp(self) -> None:
        """Create project-local folders for pipeline tests."""
        self.temp_dir = ProjectTempDirectory("dataset_pipeline")
        self.root = Path(self.temp_dir.name)
        self.raw_source = self.root / "raw_source"
        self.managed_raw = self.root / "managed_raw"
        self.cleaned = self.root / "cleaned"
        self.structured = self.root / "structured"

    def tearDown(self) -> None:
        """Clean up temporary pipeline files."""
        self.temp_dir.cleanup()

    def test_full_dataset_pipeline_creates_structured_split_and_description(self) -> None:
        """The real-data pipeline should produce train/test folders and stats."""
        class_colors = {
            "impressionism": (30, 80, 160),
            "baroque": (90, 120, 40),
            "cubism": (150, 40, 110),
        }
        for class_name, base_color in class_colors.items():
            for index in range(5):
                self._write_color_image(
                    self.raw_source / class_name / f"{index}.jpg",
                    color=(base_color[0] + index, base_color[1], base_color[2]),
                )
        self._write_color_image(
            self.raw_source / "impressionism" / "duplicate.jpg",
            color=(30, 80, 160),
        )
        self._write_color_image(
            self.raw_source / "baroque" / "lowres.jpg",
            color=(90, 90, 140),
            size=(20, 20),
        )
        (self.raw_source / "cubism" / "broken.jpg").write_text(
            "not an image",
            encoding="utf-8",
        )

        metadata = DatasetDownloader().acquire(
            DatasetDownloadRequest(
                source=DatasetSource.WIKIART,
                local_source=self.raw_source,
                target_dir=self.managed_raw,
            ),
        )
        records = list(
            ArtDatasetLoader(
                ArtDatasetLoaderConfig(dataset_root=self.managed_raw),
            ).iter_records(),
        )
        cleaning_report = DataCleaner(
            CleaningConfig(
                output_root=self.cleaned,
                resize_to=(64, 64),
                min_resolution=(32, 32),
                remove_grayscale_anomalies=False,
            ),
        ).clean_records(records)
        split_report = DatasetStructurer(
            StructuringConfig(
                cleaned_root=self.cleaned,
                output_root=self.structured,
                train_ratio=0.8,
                random_seed=7,
            ),
        ).create_split()
        stats = DatasetStatsCollector().collect(self.structured)
        description_path = DatasetStatsCollector().write_description(
            stats,
            self.root / "dataset_description.txt",
            source_name="Temporary real-image fixture",
            preprocessing_steps=["resize to 64x64", "RGB standardization"],
            filtering_rules=["remove corrupted", "remove duplicates"],
        )
        dataset = ArtDataset(self.structured)
        split = dataset.get_train_test_split()
        validity = check_dataset_validity(
            self.structured,
            min_classes=3,
            min_images_per_class=4,
        )

        self.assertTrue(metadata.complete)
        self.assertGreaterEqual(cleaning_report.corrupted_count, 1)
        self.assertGreaterEqual(cleaning_report.duplicate_count, 1)
        self.assertGreaterEqual(cleaning_report.low_resolution_count, 1)
        self.assertEqual(sum(split_report.train_counts.values()), 12)
        self.assertEqual(sum(split_report.test_counts.values()), 3)
        self.assertEqual(stats.total_images, 15)
        self.assertTrue(description_path.exists())
        self.assertEqual(len(split.train_paths), 12)
        self.assertEqual(len(split.test_paths), 3)
        self.assertEqual(len(dataset.load_images("train")), 12)
        self.assertEqual(validity.dataset_status, DatasetStatus.VALID_REAL_DATA)

    def test_structured_validity_detects_corrupted_images(self) -> None:
        """Validity check should mark unreadable structured images as corrupted."""
        target = self.structured / "train" / "impressionism"
        target.mkdir(parents=True)
        (target / "broken.jpg").write_text("not an image", encoding="utf-8")
        (self.structured / "test" / "impressionism").mkdir(parents=True)

        validity = check_dataset_validity(
            self.structured,
            min_classes=1,
            min_images_per_class=1,
        )

        self.assertEqual(validity.dataset_status, DatasetStatus.CORRUPTED_DATA)

    def _write_color_image(
        self,
        image_path: Path,
        *,
        color: tuple[int, int, int],
        size: tuple[int, int] = (80, 80),
    ) -> Path:
        """Write a valid BGR image for pipeline tests."""
        image_path.parent.mkdir(parents=True, exist_ok=True)
        x_values = np.arange(size[0], dtype=np.uint8)[None, :]
        y_values = np.arange(size[1], dtype=np.uint8)[:, None]
        image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        image[:, :, 0] = (color[0] + x_values) % 255
        image[:, :, 1] = (color[1] + y_values) % 255
        image[:, :, 2] = (color[2] + x_values + y_values) % 255
        if not cv2.imwrite(str(image_path), image):
            msg = f"Failed to write test image: {image_path}"
            raise RuntimeError(msg)
        return image_path


if __name__ == "__main__":
    unittest.main()
