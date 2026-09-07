"""Dataset structuring and stratified train/test splitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
import shutil


@dataclass(frozen=True)
class StructuringConfig:
    """Configuration for train/test directory creation."""

    cleaned_root: Path
    output_root: Path
    train_ratio: float = 0.8
    random_seed: int = 42


@dataclass
class SplitReport:
    """Counters produced by stratified dataset structuring."""

    train_counts: dict[str, int] = field(default_factory=dict)
    test_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        """Return total copied image count."""
        return sum(self.train_counts.values()) + sum(self.test_counts.values())


class DatasetStructurer:
    """Create stratified ``train`` and ``test`` directory structures."""

    def __init__(self, config: StructuringConfig) -> None:
        """Create a structurer with deterministic splitting settings."""
        if not 0.0 < config.train_ratio < 1.0:
            msg = "train_ratio must be in the interval (0, 1)."
            raise ValueError(msg)
        self._config = config

    def create_split(self) -> SplitReport:
        """Create train/test split directories from cleaned class folders."""
        report = SplitReport()
        rng = random.Random(self._config.random_seed)
        for class_dir in sorted(path for path in self._config.cleaned_root.iterdir() if path.is_dir()):
            images = sorted(
                path for path in class_dir.iterdir() if path.suffix.lower() == ".jpg"
            )
            rng.shuffle(images)
            split_index = max(1, int(len(images) * self._config.train_ratio))
            split_index = min(split_index, max(1, len(images) - 1)) if len(images) > 1 else len(images)
            train_images = images[:split_index]
            test_images = images[split_index:]
            report.train_counts[class_dir.name] = self._copy_images(
                train_images,
                self._config.output_root / "train" / class_dir.name,
            )
            report.test_counts[class_dir.name] = self._copy_images(
                test_images,
                self._config.output_root / "test" / class_dir.name,
            )
        return report

    def _copy_images(self, images: list[Path], target_dir: Path) -> int:
        """Copy images into a split/class directory."""
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in images:
            shutil.copy2(image_path, target_dir / image_path.name)
        return len(images)
