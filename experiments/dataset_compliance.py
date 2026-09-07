"""Dataset compliance checks for publication-grade experiments.

This module separates real publication datasets from synthetic debug data.
Synthetic data is allowed only for pipeline validation and must never be used
as paper evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class DatasetStatus(str, Enum):
    """Publication-compliance status for an experiment dataset."""

    VALID_REAL_DATA = "VALID_REAL_DATA"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    CORRUPTED_DATA = "CORRUPTED_DATA"
    DEBUG_ONLY = "DEBUG_ONLY"


@dataclass(frozen=True)
class DatasetComplianceReport:
    """Result returned by dataset compliance checks."""

    dataset_status: DatasetStatus
    dataset_root: Path | None
    image_count: int
    class_count: int
    min_images_per_class: int
    message: str

    @property
    def is_publication_ready(self) -> bool:
        """Return whether the dataset can be used for paper experiments."""
        return self.dataset_status == DatasetStatus.VALID_REAL_DATA


def check_dataset_compliance(
    dataset_root: Path | None,
    *,
    debug_synthetic: bool = False,
    min_classes: int = 3,
    min_images_per_class: int = 30,
) -> DatasetComplianceReport:
    """Check whether a dataset is valid for SCI/IET-style experiments.

    Args:
        dataset_root: Folder dataset root. The expected structure is
            ``root/style_name/.../image.jpg``.
        debug_synthetic: Whether the caller explicitly requested synthetic
            debug data.
        min_classes: Minimum style/category count required for real experiments.
        min_images_per_class: Minimum image count per class required for real
            experiments.

    Returns:
        A ``DatasetComplianceReport`` with status:
        ``VALID_REAL_DATA``, ``INSUFFICIENT_DATA``, or ``DEBUG_ONLY``.
    """
    if debug_synthetic:
        return DatasetComplianceReport(
            dataset_status=DatasetStatus.DEBUG_ONLY,
            dataset_root=dataset_root,
            image_count=0,
            class_count=0,
            min_images_per_class=0,
            message=(
                "Synthetic data is enabled only for debug mode. Results are "
                "not publication-grade and must not be reported as paper metrics."
            ),
        )

    if dataset_root is None:
        return DatasetComplianceReport(
            dataset_status=DatasetStatus.INSUFFICIENT_DATA,
            dataset_root=None,
            image_count=0,
            class_count=0,
            min_images_per_class=0,
            message=(
                "REQUIRED DOWNLOAD STEP: provide a real art dataset root such as "
                "WikiArt, Kaggle Painter by Numbers, or BAM before running "
                "publication experiments."
            ),
        )

    root = dataset_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return DatasetComplianceReport(
            dataset_status=DatasetStatus.INSUFFICIENT_DATA,
            dataset_root=root,
            image_count=0,
            class_count=0,
            min_images_per_class=0,
            message=f"Dataset root does not exist or is not a directory: {root}",
        )

    counts = _count_images_by_class(root)
    if not counts:
        counts = _count_kaggle_metadata_by_class(root)
    valid_counts = {
        label: count for label, count in counts.items() if count >= min_images_per_class
    }
    image_count = sum(counts.values())
    class_count = len(valid_counts)
    observed_min = min(valid_counts.values(), default=0)
    if class_count < min_classes:
        return DatasetComplianceReport(
            dataset_status=DatasetStatus.INSUFFICIENT_DATA,
            dataset_root=root,
            image_count=image_count,
            class_count=class_count,
            min_images_per_class=observed_min,
            message=(
                "Insufficient real art data for publication experiments: "
                f"requires at least {min_classes} classes with "
                f"{min_images_per_class} images per class."
            ),
        )

    return DatasetComplianceReport(
        dataset_status=DatasetStatus.VALID_REAL_DATA,
        dataset_root=root,
        image_count=image_count,
        class_count=class_count,
        min_images_per_class=observed_min,
        message="Real art dataset satisfies the minimum experiment threshold.",
    )


def check_dataset_validity(
    structured_root: Path,
    *,
    min_classes: int = 3,
    min_images_per_class: int = 30,
) -> DatasetComplianceReport:
    """Validate a structured ``train``/``test`` dataset for IET experiments."""
    root = structured_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return DatasetComplianceReport(
            dataset_status=DatasetStatus.INSUFFICIENT_DATA,
            dataset_root=root,
            image_count=0,
            class_count=0,
            min_images_per_class=0,
            message=f"Structured dataset root does not exist: {root}",
        )
    train_dir = root / "train"
    test_dir = root / "test"
    if not train_dir.exists() or not test_dir.exists():
        return DatasetComplianceReport(
            dataset_status=DatasetStatus.CORRUPTED_DATA,
            dataset_root=root,
            image_count=0,
            class_count=0,
            min_images_per_class=0,
            message="Structured dataset must contain train/ and test/ directories.",
        )

    counts = _count_structured_images(root)
    corrupted_count = _count_corrupted_images(root)
    if corrupted_count > 0:
        return DatasetComplianceReport(
            dataset_status=DatasetStatus.CORRUPTED_DATA,
            dataset_root=root,
            image_count=sum(counts.values()),
            class_count=len(counts),
            min_images_per_class=min(counts.values(), default=0),
            message=f"Structured dataset contains {corrupted_count} unreadable images.",
        )

    valid_counts = {
        label: count for label, count in counts.items() if count >= min_images_per_class
    }
    if len(valid_counts) < min_classes:
        return DatasetComplianceReport(
            dataset_status=DatasetStatus.INSUFFICIENT_DATA,
            dataset_root=root,
            image_count=sum(counts.values()),
            class_count=len(valid_counts),
            min_images_per_class=min(valid_counts.values(), default=0),
            message=(
                "Structured dataset is too small for publication experiments: "
                f"requires at least {min_classes} classes with "
                f"{min_images_per_class} images per class."
            ),
        )
    return DatasetComplianceReport(
        dataset_status=DatasetStatus.VALID_REAL_DATA,
        dataset_root=root,
        image_count=sum(counts.values()),
        class_count=len(valid_counts),
        min_images_per_class=min(valid_counts.values()),
        message="Structured train/test dataset is valid for real-data experiments.",
    )


def _count_images_by_class(root: Path) -> dict[str, int]:
    """Count supported images grouped by first-level class/style directory."""
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


def _count_kaggle_metadata_by_class(root: Path) -> dict[str, int]:
    """Count images from Kaggle Painter by Numbers metadata when available."""
    import csv

    metadata_path = _find_kaggle_metadata(root)
    if metadata_path is None:
        return {}

    counts: dict[str, int] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            style_label = _first_present(row, ("style", "genre", "Style", "Genre"))
            image_name = _first_present(row, ("filename", "image", "new_filename", "id"))
            if style_label is None or image_name is None:
                continue
            if _resolve_metadata_image(root, image_name) is None:
                continue
            counts[style_label] = counts.get(style_label, 0) + 1
    return counts


def _find_kaggle_metadata(root: Path) -> Path | None:
    """Return the first conventional Kaggle metadata CSV path."""
    for name in ("train_info.csv", "all_data_info.csv", "test_info.csv"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _resolve_metadata_image(root: Path, image_name: str) -> Path | None:
    """Resolve a metadata image name to an existing image path."""
    raw_name = image_name.strip()
    candidates = [
        root / raw_name,
        root / "train" / raw_name,
        root / "images" / raw_name,
    ]
    if not Path(raw_name).suffix:
        candidates.extend(
            root / "train" / f"{raw_name}{suffix}"
            for suffix in SUPPORTED_IMAGE_SUFFIXES
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _first_present(row: dict[str, str], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty metadata value for possible column names."""
    for key in keys:
        value = row.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def _count_structured_images(root: Path) -> dict[str, int]:
    """Count images across train/test structured class folders."""
    counts: dict[str, int] = {}
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            counts[class_dir.name] = counts.get(class_dir.name, 0) + len(
                [
                    path
                    for path in class_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
                ],
            )
    return counts


def _count_corrupted_images(root: Path) -> int:
    """Count unreadable image files in a structured dataset."""
    import cv2

    corrupted_count = 0
    for image_path in root.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
            corrupted_count += 1
    return corrupted_count
