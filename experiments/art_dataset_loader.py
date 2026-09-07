"""Publication-grade real art dataset loading utilities.

The loader supports WikiArt-style folder datasets and Kaggle Painter by Numbers
metadata layouts. It standardizes image size, channel order, and optional pixel
normalization for downstream experiments.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from experiments.dataset_compliance import SUPPORTED_IMAGE_SUFFIXES

LOGGER = logging.getLogger(__name__)


class DatasetFormat(str, Enum):
    """Supported real art dataset layouts."""

    WIKIART = "wikiart"
    KAGGLE_PAINTER_BY_NUMBERS = "kaggle_painter_by_numbers"
    AUTO = "auto"


@dataclass(frozen=True)
class ArtImageRecord:
    """Metadata for one real artwork image."""

    image_path: Path
    style_label: str
    artist_name: str | None = None
    title: str | None = None
    dataset_name: str | None = None


@dataclass(frozen=True)
class LoadedArtworkImage:
    """Standardized image and metadata ready for feature extraction."""

    record: ArtImageRecord
    rgb_image: np.ndarray
    normalized_image: np.ndarray

    @property
    def bgr_uint8(self) -> np.ndarray:
        """Return the RGB image converted to OpenCV BGR uint8 format."""
        return cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)


@dataclass(frozen=True)
class ArtDatasetLoaderConfig:
    """Configuration for real art dataset loading."""

    dataset_root: Path
    dataset_format: DatasetFormat = DatasetFormat.AUTO
    resize_to: tuple[int, int] = (128, 128)
    metadata_file: Path | None = None
    max_images_per_style: int | None = None
    normalize_pixels: bool = True


class ArtDatasetLoader:
    """Load real artwork records and standardized images."""

    def __init__(self, config: ArtDatasetLoaderConfig) -> None:
        """Create a loader for one real dataset root."""
        self._config = config
        self._root = config.dataset_root.expanduser().resolve()
        if not self._root.exists() or not self._root.is_dir():
            msg = f"dataset_root must be an existing directory: {self._root}"
            raise FileNotFoundError(msg)

    def iter_records(self) -> Iterator[ArtImageRecord]:
        """Yield artwork image records from the configured dataset."""
        dataset_format = self._resolve_format()
        if dataset_format == DatasetFormat.WIKIART:
            yield from self._iter_wikiart_records()
            return
        if dataset_format == DatasetFormat.KAGGLE_PAINTER_BY_NUMBERS:
            yield from self._iter_kaggle_records()
            return
        msg = f"Unsupported dataset format: {dataset_format}"
        raise ValueError(msg)

    def load_image(self, record: ArtImageRecord) -> LoadedArtworkImage | None:
        """Read and standardize one artwork image.

        Images are returned as RGB uint8 arrays resized to ``resize_to``.
        ``normalized_image`` is float32 in [0, 1] when normalization is enabled.
        """
        bgr_image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if bgr_image is None:
            LOGGER.warning("Skipping unreadable image: %s", record.image_path)
            return None
        width, height = self._config.resize_to
        resized = cv2.resize(
            bgr_image,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (
            rgb_image.astype(np.float32) / 255.0
            if self._config.normalize_pixels
            else rgb_image.astype(np.float32)
        )
        return LoadedArtworkImage(
            record=record,
            rgb_image=rgb_image,
            normalized_image=normalized,
        )

    def iter_images(self) -> Iterator[LoadedArtworkImage]:
        """Yield standardized images, skipping unreadable files."""
        for record in self.iter_records():
            loaded = self.load_image(record)
            if loaded is not None:
                yield loaded

    def load(self) -> list[ArtImageRecord]:
        """Load all artwork records without reading full image pixels."""
        return list(self.iter_records())

    def _resolve_format(self) -> DatasetFormat:
        """Infer the dataset format when AUTO is configured."""
        if self._config.dataset_format != DatasetFormat.AUTO:
            return self._config.dataset_format
        if self._find_kaggle_metadata_file() is not None:
            return DatasetFormat.KAGGLE_PAINTER_BY_NUMBERS
        return DatasetFormat.WIKIART

    def _iter_wikiart_records(self) -> Iterator[ArtImageRecord]:
        """Yield records from ``root/style/.../image`` folder layouts."""
        seen_by_style: dict[str, int] = {}
        for image_path in sorted(self._root.rglob("*")):
            if not self._is_supported_image(image_path):
                continue
            parts = image_path.relative_to(self._root).parts
            if len(parts) < 2:
                continue
            style_label = parts[0]
            if self._style_limit_reached(style_label, seen_by_style):
                continue
            artist_name = parts[1] if len(parts) > 2 else None
            seen_by_style[style_label] = seen_by_style.get(style_label, 0) + 1
            yield ArtImageRecord(
                image_path=image_path,
                style_label=style_label,
                artist_name=artist_name,
                title=image_path.stem,
                dataset_name=self._root.name,
            )

    def _iter_kaggle_records(self) -> Iterator[ArtImageRecord]:
        """Yield records from Kaggle Painter by Numbers CSV metadata."""
        metadata_path = self._find_kaggle_metadata_file()
        if metadata_path is None:
            msg = "Kaggle metadata file was not found."
            raise FileNotFoundError(msg)

        seen_by_style: dict[str, int] = {}
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                style_label = _first_present(row, ("style", "genre", "Style", "Genre"))
                image_name = _first_present(
                    row,
                    ("filename", "image", "new_filename", "id"),
                )
                if not style_label or not image_name:
                    continue
                if self._style_limit_reached(style_label, seen_by_style):
                    continue
                image_path = self._resolve_kaggle_image_path(image_name)
                if image_path is None:
                    continue
                seen_by_style[style_label] = seen_by_style.get(style_label, 0) + 1
                yield ArtImageRecord(
                    image_path=image_path,
                    style_label=style_label,
                    artist_name=_first_present(row, ("artist", "Artist")),
                    title=_first_present(row, ("title", "Title")),
                    dataset_name=self._root.name,
                )

    def _find_kaggle_metadata_file(self) -> Path | None:
        """Return configured or conventional Kaggle metadata CSV path."""
        if self._config.metadata_file is not None:
            metadata_path = self._config.metadata_file.expanduser().resolve()
            return metadata_path if metadata_path.exists() else None
        for name in ("train_info.csv", "all_data_info.csv", "test_info.csv"):
            candidate = self._root / name
            if candidate.exists():
                return candidate
        return None

    def _resolve_kaggle_image_path(self, image_name: str) -> Path | None:
        """Resolve a Kaggle metadata image name to an existing image path."""
        raw_name = image_name.strip()
        candidates = [
            self._root / raw_name,
            self._root / "train" / raw_name,
            self._root / "images" / raw_name,
        ]
        if not Path(raw_name).suffix:
            candidates.extend(
                self._root / "train" / f"{raw_name}{suffix}"
                for suffix in SUPPORTED_IMAGE_SUFFIXES
            )
        for candidate in candidates:
            if self._is_supported_image(candidate):
                return candidate
        return None

    def _style_limit_reached(
        self,
        style_label: str,
        seen_by_style: dict[str, int],
    ) -> bool:
        """Return whether the configured per-style limit has been reached."""
        limit = self._config.max_images_per_style
        return limit is not None and seen_by_style.get(style_label, 0) >= limit

    def _is_supported_image(self, image_path: Path) -> bool:
        """Return whether the path is a supported image file."""
        return (
            image_path.exists()
            and image_path.is_file()
            and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )


def _first_present(row: dict[str, str], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty value among possible CSV column names."""
    for key in keys:
        value = row.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None
