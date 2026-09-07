"""Dataset statistics query helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from models import Artist, Artwork, Dataset, Style
from repositories.exceptions import RepositoryError


@dataclass(frozen=True)
class DatasetStatistics:
    """Aggregated statistics for one dataset."""

    dataset_id: int
    dataset_name: str
    image_count: int
    style_count: int
    artist_count: int
    styles: dict[str, int]
    artists: dict[str, int]


class DatasetStatisticsQuery:
    """Query aggregate dataset statistics from ORM tables."""

    def __init__(self, session: Any) -> None:
        """Create a statistics query helper for an active session."""
        self._session = session

    def build(self, dataset: Dataset) -> DatasetStatistics:
        """Build statistics for a dataset entity."""
        style_counts = self._count_by_related_name(
            Style.name,
            Artwork.style_id == Style.id,
            dataset.id,
        )
        artist_counts = self._count_by_related_name(
            Artist.name,
            Artwork.artist_id == Artist.id,
            dataset.id,
        )
        image_count = sum(style_counts.values())
        return DatasetStatistics(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            image_count=image_count,
            style_count=len(style_counts),
            artist_count=len(artist_counts),
            styles=style_counts,
            artists=artist_counts,
        )

    def _count_by_related_name(
        self,
        name_column: Any,
        join_condition: Any,
        dataset_id: int,
    ) -> dict[str, int]:
        """Count artworks grouped by a related name column."""
        try:
            statement = (
                select(name_column, func.count(Artwork.id))
                .select_from(Artwork)
                .join_from(Artwork, name_column.class_, join_condition)
                .where(Artwork.dataset_id == dataset_id)
                .group_by(name_column)
            )
            return {
                str(name): int(count)
                for name, count in self._session.execute(statement)
            }
        except SQLAlchemyError as exc:
            raise RepositoryError(str(exc)) from exc