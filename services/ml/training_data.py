"""Database-backed loading of feature matrices for model training."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from core.features.feature_codec import deserialize_feature_vector
from models import Artwork, Feature
from services.dataset_scope import apply_enabled_feature_scope

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingDataset:
    """Feature matrix and labels used by classification and retrieval."""

    feature_matrix: np.ndarray
    style_labels: list[str]
    artwork_ids: list[int]
    feature_type: str

    @property
    def sample_count(self) -> int:
        """Return the number of training samples."""
        return int(self.feature_matrix.shape[0])

    @property
    def feature_count(self) -> int:
        """Return the number of feature columns."""
        return int(self.feature_matrix.shape[1])


class TrainingDataLoader:
    """Load serialized feature vectors and style labels from the database."""

    def __init__(self, session: Session) -> None:
        """Create a loader bound to a SQLAlchemy session."""
        self._session = session

    def load(self, feature_type: str = "combined") -> TrainingDataset:
        """Load a complete feature matrix for a feature type.

        Args:
            feature_type: Feature vector family to load from the ``features``
                table. The default matches the existing combined feature type.

        Returns:
            A ``TrainingDataset`` with shape ``(n_samples, n_features)``.
        """
        statement = (
            select(Feature)
            .options(joinedload(Feature.artwork).joinedload(Artwork.style))
            .where(Feature.feature_type == feature_type)
            .order_by(Feature.artwork_id)
        )
        statement = apply_enabled_feature_scope(statement)
        feature_rows = list(self._session.scalars(statement))
        return self._build_dataset(feature_rows, feature_type)

    def _build_dataset(
        self,
        feature_rows: list[Feature],
        feature_type: str,
    ) -> TrainingDataset:
        """Convert ORM feature rows into validated NumPy training data."""
        vectors: list[np.ndarray] = []
        labels: list[str] = []
        artwork_ids: list[int] = []
        expected_length: int | None = None

        for feature in feature_rows:
            artwork = feature.artwork
            if artwork is None or artwork.style is None:
                LOGGER.warning("Skipping feature %s without style label.", feature.id)
                continue

            vector = deserialize_feature_vector(feature.vector)
            if expected_length is None:
                expected_length = int(vector.shape[0])
            elif vector.shape[0] != expected_length:
                msg = (
                    "All feature vectors must have the same length; "
                    f"expected {expected_length}, got {vector.shape[0]}."
                )
                raise ValueError(msg)

            vectors.append(vector)
            labels.append(artwork.style.name)
            artwork_ids.append(artwork.id)

        if not vectors:
            msg = f"No training features found for feature_type={feature_type!r}."
            raise ValueError(msg)

        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        return TrainingDataset(
            feature_matrix=matrix,
            style_labels=labels,
            artwork_ids=artwork_ids,
            feature_type=feature_type,
        )
