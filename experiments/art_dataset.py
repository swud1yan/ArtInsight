"""Unified experiment-facing dataset interface for ArtInsight."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    """Train/test image paths and labels."""

    train_paths: list[Path]
    train_labels: list[str]
    test_paths: list[Path]
    test_labels: list[str]


class ArtDataset:
    """Load structured artwork datasets for feature/PCA experiments."""

    def __init__(self, structured_root: Path) -> None:
        """Create an interface for a ``train``/``test`` dataset root."""
        self._root = structured_root.expanduser().resolve()
        if not self._root.exists():
            msg = f"structured_root does not exist: {self._root}"
            raise FileNotFoundError(msg)

    def load_images(self, split: str = "train") -> list[np.ndarray]:
        """Load all images for a split as RGB uint8 arrays."""
        images: list[np.ndarray] = []
        for image_path in self._iter_split_paths(split):
            bgr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr_image is None:
                continue
            images.append(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        return images

    def get_labels(self, split: str = "train") -> list[str]:
        """Return class labels for all images in a split."""
        return [path.parent.name for path in self._iter_split_paths(split)]

    def get_train_test_split(self) -> DatasetSplit:
        """Return train/test image paths and labels."""
        train_paths = self._iter_split_paths("train")
        test_paths = self._iter_split_paths("test")
        return DatasetSplit(
            train_paths=train_paths,
            train_labels=[path.parent.name for path in train_paths],
            test_paths=test_paths,
            test_labels=[path.parent.name for path in test_paths],
        )

    def _iter_split_paths(self, split: str) -> list[Path]:
        """Return sorted image paths for a split."""
        split_dir = self._root / split
        if not split_dir.exists():
            msg = f"Dataset split does not exist: {split_dir}"
            raise FileNotFoundError(msg)
        return sorted(split_dir.glob("*/*.jpg"))
