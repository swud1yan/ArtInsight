"""Image cleaning utilities for publication-grade artwork datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import logging
import shutil

import cv2
import numpy as np

from experiments.art_dataset_loader import ArtImageRecord

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleaningConfig:
    """Configuration for image standardization and filtering."""

    output_root: Path
    resize_to: tuple[int, int] = (224, 224)
    min_resolution: tuple[int, int] = (128, 128)
    remove_grayscale_anomalies: bool = True
    copy_original_on_failure: bool = False


@dataclass
class CleaningReport:
    """Counters produced by the dataset cleaning step."""

    processed_count: int = 0
    kept_count: int = 0
    corrupted_count: int = 0
    duplicate_count: int = 0
    low_resolution_count: int = 0
    grayscale_anomaly_count: int = 0
    written_paths: list[Path] = field(default_factory=list)


class DataCleaner:
    """Clean, deduplicate, and standardize real artwork images."""

    def __init__(self, config: CleaningConfig) -> None:
        """Create a cleaner with a managed output root."""
        self._config = config
        self._seen_hashes: set[str] = set()
        self._config.output_root.mkdir(parents=True, exist_ok=True)

    def clean_records(self, records: list[ArtImageRecord]) -> CleaningReport:
        """Clean artwork records and write standardized RGB images."""
        report = CleaningReport()
        for index, record in enumerate(records):
            report.processed_count += 1
            cleaned = self._clean_one(record, index, report)
            if cleaned is not None:
                report.kept_count += 1
                report.written_paths.append(cleaned)
        return report

    def _clean_one(
        self,
        record: ArtImageRecord,
        index: int,
        report: CleaningReport,
    ) -> Path | None:
        """Clean one image and return the written target path."""
        bgr_image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if bgr_image is None:
            report.corrupted_count += 1
            return None
        if self._is_low_resolution(bgr_image):
            report.low_resolution_count += 1
            return None
        if self._config.remove_grayscale_anomalies and self._is_grayscale_like(
            bgr_image,
        ):
            report.grayscale_anomaly_count += 1
            return None

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image_hash = self._image_hash(rgb_image)
        if image_hash in self._seen_hashes:
            report.duplicate_count += 1
            return None
        self._seen_hashes.add(image_hash)

        resized = cv2.resize(
            rgb_image,
            self._config.resize_to,
            interpolation=cv2.INTER_AREA,
        )
        output_path = self._target_path(record, index)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bgr_output = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), bgr_output):
            report.corrupted_count += 1
            if self._config.copy_original_on_failure:
                shutil.copy2(record.image_path, output_path)
            else:
                return None
        return output_path

    def _target_path(self, record: ArtImageRecord, index: int) -> Path:
        """Return a deterministic cleaned image output path."""
        safe_label = _safe_name(record.style_label)
        stem = _safe_name(record.title or record.image_path.stem)
        return self._config.output_root / safe_label / f"{index:08d}_{stem}.jpg"

    def _is_low_resolution(self, image: np.ndarray) -> bool:
        """Return whether the input image is below configured resolution."""
        min_width, min_height = self._config.min_resolution
        height, width = image.shape[:2]
        return width < min_width or height < min_height

    def _is_grayscale_like(self, image: np.ndarray) -> bool:
        """Return whether a color image has near-identical channels."""
        if image.ndim != 3 or image.shape[2] != 3:
            return True
        channel_std = np.std(image.astype(np.float32), axis=2)
        return float(np.mean(channel_std)) < 1.0

    def _image_hash(self, rgb_image: np.ndarray) -> str:
        """Return a stable content hash after resizing for duplicate detection."""
        resized = cv2.resize(rgb_image, (64, 64), interpolation=cv2.INTER_AREA)
        return hashlib.sha256(resized.tobytes()).hexdigest()


def _safe_name(value: str) -> str:
    """Normalize labels and titles for filesystem paths."""
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    return "_".join(part for part in cleaned.split("_") if part) or "unknown"
