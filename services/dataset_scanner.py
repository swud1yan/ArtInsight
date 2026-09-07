"""Filesystem scanning helpers for dataset import."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
UNKNOWN_STYLE_NAME = "Unknown Style"
UNKNOWN_ARTIST_NAME = "Unknown Artist"


@dataclass(frozen=True)
class ImageCandidate:
    """Import-ready image path and inferred metadata labels."""

    path: Path
    title: str
    style_name: str
    artist_name: str


class DatasetScanner:
    """Scan a dataset directory for supported readable image files."""

    def scan(self, root_path: Path) -> tuple[list[ImageCandidate], int]:
        """Return readable image candidates and skipped unreadable count."""
        candidates: list[ImageCandidate] = []
        skipped_count = 0
        for image_path in self.iter_image_paths(root_path):
            if not self.is_readable_image(image_path):
                skipped_count += 1
                LOGGER.warning("Skipping unreadable image file: %s", image_path)
                continue
            style_name, artist_name = self.extract_labels(root_path, image_path)
            candidates.append(
                ImageCandidate(
                    path=image_path,
                    title=image_path.stem,
                    style_name=style_name,
                    artist_name=artist_name,
                ),
            )
        return candidates, skipped_count

    def iter_image_paths(self, root_path: Path) -> list[Path]:
        """Return supported image paths under a dataset root."""
        return sorted(
            path
            for path in root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

    def extract_labels(self, root_path: Path, image_path: Path) -> tuple[str, str]:
        """Extract style and artist labels from relative directory names."""
        relative_parts = image_path.relative_to(root_path).parts
        style_name = (
            relative_parts[0]
            if len(relative_parts) >= 2
            else UNKNOWN_STYLE_NAME
        )
        artist_name = (
            relative_parts[1]
            if len(relative_parts) >= 3
            else UNKNOWN_ARTIST_NAME
        )
        return style_name, artist_name

    def is_readable_image(self, image_path: Path) -> bool:
        """Check whether an image file can be opened and is not empty."""
        try:
            with image_path.open("rb") as image_file:
                header = image_file.read(32)
            return bool(header)
        except OSError as exc:
            LOGGER.warning("Failed to read image file %s: %s", image_path, exc)
            return False