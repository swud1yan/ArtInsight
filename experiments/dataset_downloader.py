"""Dataset acquisition and import helpers for real art datasets.

The module supports local imports and Kaggle CLI downloads when credentials are
configured by the user. It does not fabricate data and never falls back to
synthetic samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from pathlib import Path
import shutil
import subprocess
import zipfile

LOGGER = logging.getLogger(__name__)


class DatasetSource(str, Enum):
    """Supported real dataset sources."""

    WIKIART = "wikiart"
    KAGGLE_PAINTER_BY_NUMBERS = "kaggle_painter_by_numbers"


@dataclass(frozen=True)
class DatasetDownloadRequest:
    """Request describing how to acquire a real dataset."""

    source: DatasetSource
    target_dir: Path
    local_source: Path | None = None
    kaggle_competition: str | None = None
    kaggle_dataset: str | None = None
    force: bool = False


@dataclass(frozen=True)
class DatasetMetadata:
    """Traceable acquisition metadata written after import/download."""

    source: DatasetSource
    target_dir: Path
    acquisition_mode: str
    acquired_at_utc: str
    file_count: int
    byte_count: int
    complete: bool
    notes: str


class DatasetDownloader:
    """Acquire real datasets from local folders or Kaggle CLI."""

    def acquire(self, request: DatasetDownloadRequest) -> DatasetMetadata:
        """Acquire the requested dataset and write acquisition metadata."""
        target = request.target_dir.expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        if request.local_source is not None:
            metadata = self._import_local(request, target)
        else:
            metadata = self._download_with_kaggle(request, target)
        self._write_metadata(metadata)
        return metadata

    def check_completeness(self, dataset_dir: Path) -> bool:
        """Return whether a dataset directory appears non-empty and structured."""
        root = dataset_dir.expanduser().resolve()
        if not root.exists() or not root.is_dir():
            return False
        image_count = sum(
            1
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        has_metadata = any(
            (root / name).exists()
            for name in ("train_info.csv", "all_data_info.csv", "dataset_metadata.json")
        )
        return image_count > 0 or has_metadata

    def _import_local(
        self,
        request: DatasetDownloadRequest,
        target: Path,
    ) -> DatasetMetadata:
        """Import a local real dataset folder into the managed data area."""
        assert request.local_source is not None
        source = request.local_source.expanduser().resolve()
        if not source.exists() or not source.is_dir():
            msg = f"local_source must be an existing directory: {source}"
            raise FileNotFoundError(msg)
        if any(target.iterdir()) and not request.force:
            msg = f"target_dir is not empty; use force=True to overwrite: {target}"
            raise FileExistsError(msg)
        if request.force:
            self._clear_directory(target)
        self._copy_directory(source, target)
        return self._build_metadata(
            request.source,
            target,
            "local_import",
            f"Imported from local source: {source}",
        )

    def _download_with_kaggle(
        self,
        request: DatasetDownloadRequest,
        target: Path,
    ) -> DatasetMetadata:
        """Download a Kaggle dataset or competition archive with Kaggle CLI."""
        if request.source != DatasetSource.KAGGLE_PAINTER_BY_NUMBERS:
            msg = (
                "Direct remote download is currently supported only for Kaggle "
                "Painter by Numbers. WikiArt should be imported from a licensed "
                "local snapshot."
            )
            raise ValueError(msg)
        if request.kaggle_competition is None and request.kaggle_dataset is None:
            msg = "Provide kaggle_competition or kaggle_dataset for Kaggle download."
            raise ValueError(msg)

        archive_dir = target / "_archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        command = self._build_kaggle_command(request, archive_dir)
        LOGGER.info("Running Kaggle download command: %s", " ".join(command))
        subprocess.run(command, check=True)
        self._extract_archives(archive_dir, target)
        return self._build_metadata(
            request.source,
            target,
            "kaggle_cli",
            "Downloaded with Kaggle CLI; credentials must be user-provided.",
        )

    def _build_kaggle_command(
        self,
        request: DatasetDownloadRequest,
        archive_dir: Path,
    ) -> list[str]:
        """Build the Kaggle CLI command for a dataset or competition."""
        if request.kaggle_competition is not None:
            return [
                "kaggle",
                "competitions",
                "download",
                "-c",
                request.kaggle_competition,
                "-p",
                str(archive_dir),
            ]
        assert request.kaggle_dataset is not None
        return [
            "kaggle",
            "datasets",
            "download",
            "-d",
            request.kaggle_dataset,
            "-p",
            str(archive_dir),
        ]

    def _extract_archives(self, archive_dir: Path, target: Path) -> None:
        """Extract downloaded zip archives without re-downloading existing files."""
        for archive_path in archive_dir.glob("*.zip"):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(target)

    def _build_metadata(
        self,
        source: DatasetSource,
        target: Path,
        acquisition_mode: str,
        notes: str,
    ) -> DatasetMetadata:
        """Build traceable metadata from a dataset directory."""
        files = [path for path in target.rglob("*") if path.is_file()]
        return DatasetMetadata(
            source=source,
            target_dir=target,
            acquisition_mode=acquisition_mode,
            acquired_at_utc=datetime.now(timezone.utc).isoformat(),
            file_count=len(files),
            byte_count=sum(path.stat().st_size for path in files),
            complete=self.check_completeness(target),
            notes=notes,
        )

    def _write_metadata(self, metadata: DatasetMetadata) -> None:
        """Write acquisition metadata next to the dataset."""
        output_path = metadata.target_dir / "dataset_metadata.json"
        payload = asdict(metadata)
        payload["source"] = metadata.source.value
        payload["target_dir"] = str(metadata.target_dir)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _copy_directory(self, source: Path, target: Path) -> None:
        """Copy a directory tree while preserving existing target root."""
        for path in source.rglob("*"):
            relative_path = path.relative_to(source)
            destination = target / relative_path
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

    def _clear_directory(self, target: Path) -> None:
        """Remove all children inside a managed target directory."""
        for child in target.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
