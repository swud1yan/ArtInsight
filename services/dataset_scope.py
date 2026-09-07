"""Helpers for applying the active dataset scope to SQLAlchemy queries."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.sql import Select

from models import Artwork, Dataset, Feature


def enabled_artwork_ids_query() -> Select[tuple[int]]:
    """Return artwork ids belonging to datasets enabled for the current experiment."""
    return (
        select(Artwork.id)
        .join(Dataset, Artwork.dataset_id == Dataset.id)
        .where(Dataset.is_enabled.is_(True))
    )


def apply_enabled_dataset_scope(statement: Select[tuple[object]]) -> Select[tuple[object]]:
    """Restrict an Artwork-based select statement to enabled datasets."""
    return statement.join(Dataset, Artwork.dataset_id == Dataset.id).where(
        Dataset.is_enabled.is_(True),
    )


def apply_enabled_feature_scope(statement: Select[tuple[Feature]]) -> Select[tuple[Feature]]:
    """Restrict a Feature-based select statement to enabled datasets."""
    return (
        statement.join(Artwork, Feature.artwork_id == Artwork.id)
        .join(Dataset, Artwork.dataset_id == Dataset.id)
        .where(Dataset.is_enabled.is_(True))
    )


def delete_enabled_features_statement(feature_type: str) -> object:
    """Build a delete statement for one feature type in enabled datasets only."""
    return delete(Feature).where(
        Feature.feature_type == feature_type,
        Feature.artwork_id.in_(enabled_artwork_ids_query()),
    )
