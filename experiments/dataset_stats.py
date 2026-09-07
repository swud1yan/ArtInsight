"""Statistics and method-section descriptions for structured art datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from experiments.dataset_compliance import SUPPORTED_IMAGE_SUFFIXES


@dataclass(frozen=True)
class ClassDistribution:
    """Train/test counts for one class."""

    label: str
    train_count: int
    test_count: int

    @property
    def total_count(self) -> int:
        """Return total images for this class."""
        return self.train_count + self.test_count


@dataclass(frozen=True)
class DatasetStatistics:
    """Statistics for a structured train/test dataset."""

    dataset_root: Path
    total_images: int
    class_distribution: list[ClassDistribution]
    resolution_min: tuple[int, int] | None
    resolution_max: tuple[int, int] | None
    resolution_mean: tuple[float, float] | None
    imbalance_ratio: float | None
    corrupted_count: int = 0
    notes: list[str] = field(default_factory=list)


class DatasetStatsCollector:
    """Collect image counts, split distributions, and resolution statistics."""

    def collect(self, dataset_root: Path) -> DatasetStatistics:
        """Collect statistics from ``dataset/train`` and ``dataset/test``."""
        root = dataset_root.expanduser().resolve()
        train_counts = self._count_split(root / "train")
        test_counts = self._count_split(root / "test")
        if not train_counts and not test_counts:
            train_counts = self._count_raw_classes(root)
        labels = sorted(set(train_counts) | set(test_counts))
        distribution = [
            ClassDistribution(
                label=label,
                train_count=train_counts.get(label, 0),
                test_count=test_counts.get(label, 0),
            )
            for label in labels
        ]
        resolutions, corrupted_count = self._collect_resolutions(root)
        totals = [item.total_count for item in distribution if item.total_count > 0]
        imbalance_ratio = (
            float(max(totals) / min(totals)) if totals and min(totals) > 0 else None
        )
        return DatasetStatistics(
            dataset_root=root,
            total_images=sum(totals),
            class_distribution=distribution,
            resolution_min=_resolution_min(resolutions),
            resolution_max=_resolution_max(resolutions),
            resolution_mean=_resolution_mean(resolutions),
            imbalance_ratio=imbalance_ratio,
            corrupted_count=corrupted_count,
            notes=[],
        )

    def write_description(
        self,
        stats: DatasetStatistics,
        output_path: Path,
        *,
        source_name: str,
        preprocessing_steps: list[str],
        filtering_rules: list[str],
    ) -> Path:
        """Write a Method-section-ready dataset description text file."""
        lines = [
            "ArtInsight Dataset Description",
            "",
            f"Dataset source: {source_name}",
            f"Structured dataset root: {stats.dataset_root}",
            f"Total images: {stats.total_images}",
            f"Corrupted images after structuring: {stats.corrupted_count}",
            "",
            "Class distribution:",
        ]
        for item in stats.class_distribution:
            lines.append(
                f"- {item.label}: train={item.train_count}, "
                f"test={item.test_count}, total={item.total_count}",
            )
        lines.extend(
            [
                "",
                f"Resolution min: {stats.resolution_min}",
                f"Resolution max: {stats.resolution_max}",
                f"Resolution mean: {stats.resolution_mean}",
                f"Class imbalance ratio: {stats.imbalance_ratio}",
                "",
                "Preprocessing steps:",
            ],
        )
        lines.extend(f"- {step}" for step in preprocessing_steps)
        lines.append("")
        lines.append("Filtering rules:")
        lines.extend(f"- {rule}" for rule in filtering_rules)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    def _count_split(self, split_dir: Path) -> dict[str, int]:
        """Count images under a split directory by class label."""
        if not split_dir.exists():
            return {}
        return {
            class_dir.name: len(
                [
                    path
                    for path in class_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
                ],
            )
            for class_dir in split_dir.iterdir()
            if class_dir.is_dir()
        }

    def _count_raw_classes(self, root: Path) -> dict[str, int]:
        """Count raw WikiArt-style images by first-level class directory."""
        counts: dict[str, int] = {}
        for image_path in root.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            relative_parts = image_path.relative_to(root).parts
            if len(relative_parts) < 2:
                continue
            label = relative_parts[0]
            counts[label] = counts.get(label, 0) + 1
        return counts

    def _collect_resolutions(self, root: Path) -> tuple[list[tuple[int, int]], int]:
        """Collect image resolutions and count unreadable files."""
        resolutions: list[tuple[int, int]] = []
        corrupted_count = 0
        for image_path in root.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                corrupted_count += 1
                continue
            height, width = image.shape[:2]
            resolutions.append((width, height))
        return resolutions, corrupted_count


class DatasetStats:
    """Compatibility facade for automatic dataset-stat generation."""

    def __init__(self, dataset_root: Path) -> None:
        """Create a statistics generator for a structured dataset root."""
        self._dataset_root = dataset_root

    def generate(self) -> DatasetStatistics:
        """Generate dataset statistics using the project collector."""
        return DatasetStatsCollector().collect(self._dataset_root)


def _resolution_min(resolutions: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Return minimum width and height."""
    if not resolutions:
        return None
    values = np.asarray(resolutions)
    return int(values[:, 0].min()), int(values[:, 1].min())


def _resolution_max(resolutions: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Return maximum width and height."""
    if not resolutions:
        return None
    values = np.asarray(resolutions)
    return int(values[:, 0].max()), int(values[:, 1].max())


def _resolution_mean(resolutions: list[tuple[int, int]]) -> tuple[float, float] | None:
    """Return mean width and height."""
    if not resolutions:
        return None
    values = np.asarray(resolutions, dtype=np.float64)
    return float(values[:, 0].mean()), float(values[:, 1].mean())
