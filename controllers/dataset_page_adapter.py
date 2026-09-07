"""Dataset page adapter for GUI controllers."""

from __future__ import annotations

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from controllers.gui_dtos import DatasetImportViewResult, DatasetRow
from database.session import DatabaseSessionManager
from models import Artwork, Dataset
from repositories import DatasetRepository
from services.dataset_manager import DatasetManager


class DatasetPageAdapter:
    """Adapt dataset services into GUI-friendly DTOs."""

    def __init__(self, session_manager: DatabaseSessionManager) -> None:
        """Create the adapter from a shared session manager."""
        self._session_manager = session_manager
        self._dataset_manager = DatasetManager(session_manager)

    def list_datasets(self) -> list[DatasetRow]:
        """Return imported datasets with aggregate statistics."""
        with self._session_manager.session_scope() as session:
            datasets = DatasetRepository(session).list_all()
            summary_by_dataset = self._load_dataset_summaries(session)
            return [
                DatasetRow(
                    dataset_id=dataset.id,
                    is_enabled=bool(dataset.is_enabled),
                    name=dataset.name,
                    path=dataset.path,
                    image_count=summary_by_dataset.get(dataset.id, (0, 0, 0))[0],
                    style_count=summary_by_dataset.get(dataset.id, (0, 0, 0))[1],
                    artist_count=summary_by_dataset.get(dataset.id, (0, 0, 0))[2],
                    imported_at=dataset.date.strftime("%Y-%m-%d %H:%M"),
                )
                for dataset in datasets
            ]

    def import_dataset(self, dataset_path: str) -> DatasetImportViewResult:
        """Import a dataset folder through the dataset service."""
        result = self._dataset_manager.import_dataset(dataset_path)
        return DatasetImportViewResult(
            dataset_id=result.dataset_id,
            dataset_name=result.dataset_name,
            imported_count=result.imported_count,
            skipped_count=result.skipped_count,
            style_count=result.style_count,
            artist_count=result.artist_count,
        )

    def delete_dataset(self, dataset_id: int) -> None:
        """Delete a dataset after confirmation in the GUI layer."""
        self._dataset_manager.delete_dataset(dataset_id, confirm=True)

    def set_dataset_enabled(self, dataset_id: int, enabled: bool) -> None:
        """Enable or disable one dataset for the current experiment scope."""
        with self._session_manager.session_scope() as session:
            dataset = session.get(Dataset, dataset_id)
            if dataset is None:
                msg = f"Dataset does not exist: {dataset_id}"
                raise ValueError(msg)
            dataset.is_enabled = enabled

    def set_dataset_enabled_many(self, dataset_ids: list[int], enabled: bool) -> None:
        """Enable or disable multiple datasets for the current experiment scope."""
        if not dataset_ids:
            return
        target_ids = set(dataset_ids)
        with self._session_manager.session_scope() as session:
            datasets = session.scalars(
                select(Dataset).where(Dataset.id.in_(target_ids)),
            )
            for dataset in datasets:
                dataset.is_enabled = enabled

    def set_exclusive_enabled_datasets(self, dataset_ids: list[int]) -> None:
        """Make exactly the provided datasets participate in the experiment."""
        target_ids = set(dataset_ids)
        with self._session_manager.session_scope() as session:
            datasets = session.scalars(select(Dataset))
            for dataset in datasets:
                dataset.is_enabled = dataset.id in target_ids

    def _load_dataset_summaries(
        self,
        session: Session,
    ) -> dict[int, tuple[int, int, int]]:
        """Load image, style, and artist counts for all datasets at once."""
        statement = (
            select(
                Artwork.dataset_id,
                func.count(Artwork.id),
                func.count(distinct(Artwork.style_id)),
                func.count(distinct(Artwork.artist_id)),
            )
            .group_by(Artwork.dataset_id)
        )
        return {
            int(dataset_id): (int(image_count), int(style_count), int(artist_count))
            for dataset_id, image_count, style_count, artist_count
            in session.execute(statement)
        }
