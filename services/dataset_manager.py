"""Dataset import, deletion, and management workflows."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from app.config import ConfigManager
from database.initializer import initialize_database
from database.session import DatabaseSessionManager
from models import Dataset, History
from repositories import (
    ArtistRepository,
    DatasetRepository,
    HistoryRepository,
    StyleRepository,
)
from repositories.exceptions import EntityConflictError, EntityNotFoundError
from services.dataset_importer import DatasetArtworkImporter
from services.dataset_scanner import DatasetScanner
from services.dataset_statistics import DatasetStatistics, DatasetStatisticsQuery

LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100


@dataclass(frozen=True)
class DatasetImportResult:
    """Summary returned after importing a dataset."""

    dataset_id: int
    dataset_name: str
    imported_count: int
    skipped_count: int
    style_count: int
    artist_count: int


class DatasetManager:
    """Manage dataset import, deletion, listing, and statistics."""

    def __init__(
        self,
        session_manager: DatabaseSessionManager,
        scanner: DatasetScanner | None = None,
    ) -> None:
        """Create a dataset manager using the given database session manager."""
        self._session_manager = session_manager
        self._scanner = scanner or DatasetScanner()

    @classmethod
    def from_config(cls) -> "DatasetManager":
        """Create a dataset manager from the default application configuration."""
        config = ConfigManager().load()
        session_manager = DatabaseSessionManager.from_url(config.database.url)
        initialize_database(session_manager.engine)
        return cls(session_manager)

    def import_dataset(
        self,
        dataset_path: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> DatasetImportResult:
        """Import supported images under a dataset folder into the database."""
        root_path = self._validate_dataset_path(dataset_path)
        if batch_size <= 0:
            msg = "batch_size must be positive."
            raise ValueError(msg)

        image_candidates, skipped_count = self._scanner.scan(root_path)
        if not image_candidates:
            LOGGER.warning("No supported readable images found under: %s", root_path)

        with self._session_manager.session_scope() as session:
            dataset_repository = DatasetRepository(session)
            self._raise_if_dataset_exists(dataset_repository, root_path)
            dataset = dataset_repository.add(
                Dataset(name=root_path.name, path=str(root_path)),
            )
            importer = DatasetArtworkImporter(
                session=session,
                style_repository=StyleRepository(session),
                artist_repository=ArtistRepository(session),
            )
            imported_count = importer.add_artworks_in_batches(
                dataset_id=dataset.id,
                image_candidates=image_candidates,
                batch_size=batch_size,
                dataset_name=dataset.name,
            )
            HistoryRepository(session).add(
                History(
                    action="dataset_imported",
                    dataset_id=dataset.id,
                    details=(
                        f"Imported {imported_count} images from {root_path}; "
                        f"skipped {skipped_count} files."
                    ),
                ),
            )
            LOGGER.info(
                "Completed dataset import: dataset_id=%s, imported=%s, skipped=%s.",
                dataset.id,
                imported_count,
                skipped_count,
            )
            return DatasetImportResult(
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                imported_count=imported_count,
                skipped_count=skipped_count,
                style_count=len(
                    {candidate.style_name for candidate in image_candidates},
                ),
                artist_count=len(
                    {candidate.artist_name for candidate in image_candidates},
                ),
            )

    def delete_dataset(self, dataset_id: int, confirm: bool = False) -> None:
        """Delete a dataset, its artwork records, and related image files."""
        if not confirm:
            msg = "Dataset deletion requires confirm=True."
            raise PermissionError(msg)

        with self._session_manager.session_scope() as session:
            dataset_repository = DatasetRepository(session)
            dataset = self._get_existing_dataset(dataset_repository, dataset_id)
            image_paths = [Path(artwork.image_path) for artwork in dataset.artworks]
            dataset_repository.delete(dataset)
            session.flush()

        self._delete_image_files(image_paths)
        LOGGER.info(
            "Deleted dataset %s and %s image files.",
            dataset_id,
            len(image_paths),
        )

    def list_datasets(self) -> list[Dataset]:
        """Return all imported datasets."""
        with self._session_manager.session_scope() as session:
            return DatasetRepository(session).list_all()

    def get_statistics(self, dataset_id: int) -> DatasetStatistics:
        """Return image, style, and artist counts for a dataset."""
        with self._session_manager.session_scope() as session:
            dataset_repository = DatasetRepository(session)
            dataset = self._get_existing_dataset(dataset_repository, dataset_id)
            return DatasetStatisticsQuery(session).build(dataset)

    def _validate_dataset_path(self, dataset_path: str) -> Path:
        """Validate and resolve a dataset root path."""
        root_path = Path(dataset_path).expanduser().resolve()
        if not root_path.exists():
            msg = f"Dataset path does not exist: {root_path}"
            raise FileNotFoundError(msg)
        if not root_path.is_dir():
            msg = f"Dataset path must be a directory: {root_path}"
            raise NotADirectoryError(msg)
        return root_path

    def _raise_if_dataset_exists(
        self,
        repository: DatasetRepository,
        root_path: Path,
    ) -> None:
        """Raise an error if a dataset path already exists."""
        existing_dataset = repository.query_by(path=str(root_path))
        if existing_dataset:
            msg = f"Dataset already exists and will not be imported twice: {root_path}"
            LOGGER.warning(msg)
            raise EntityConflictError(msg)

    def _get_existing_dataset(
        self,
        repository: DatasetRepository,
        dataset_id: int,
    ) -> Dataset:
        """Return an existing dataset or raise a domain-level not-found error."""
        dataset = repository.get_by_id(dataset_id)
        if dataset is None:
            msg = f"Dataset does not exist: {dataset_id}"
            raise EntityNotFoundError(msg)
        return dataset

    def _delete_image_files(self, image_paths: list[Path]) -> None:
        """Delete image files from disk, logging failures without aborting."""
        for image_path in image_paths:
            try:
                if image_path.exists() and image_path.is_file():
                    image_path.unlink()
            except OSError as exc:
                LOGGER.warning("Failed to delete image file %s: %s", image_path, exc)
