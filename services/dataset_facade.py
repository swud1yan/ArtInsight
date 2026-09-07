"""Convenience facade functions for dataset management scripts."""

from __future__ import annotations

from models import Dataset
from services.dataset_manager import DatasetManager

_default_manager: DatasetManager | None = None


def get_default_dataset_manager() -> DatasetManager:
    """Return a lazily-created dataset manager using application config."""
    global _default_manager
    if _default_manager is None:
        _default_manager = DatasetManager.from_config()
    return _default_manager


def import_dataset(dataset_path: str) -> None:
    """Import a dataset path using the default manager."""
    get_default_dataset_manager().import_dataset(dataset_path)


def delete_dataset(dataset_id: int) -> None:
    """Delete a dataset using the default manager after explicit confirmation."""
    get_default_dataset_manager().delete_dataset(dataset_id, confirm=True)


def list_datasets() -> list[Dataset]:
    """List datasets using the default manager."""
    return get_default_dataset_manager().list_datasets()


def get_statistics(dataset_id: int) -> dict[str, object]:
    """Return dataset statistics as a dictionary for script-level callers."""
    statistics = get_default_dataset_manager().get_statistics(dataset_id)
    return {
        "dataset_id": statistics.dataset_id,
        "dataset_name": statistics.dataset_name,
        "image_count": statistics.image_count,
        "style_count": statistics.style_count,
        "artist_count": statistics.artist_count,
        "styles": statistics.styles,
        "artists": statistics.artists,
    }