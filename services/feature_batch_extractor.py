"""Batch feature extraction service for imported artwork records."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.features.feature_codec import serialize_feature_vector
from database.session import DatabaseSessionManager
from models import Artwork, Feature
from services.dataset_scope import (
    apply_enabled_dataset_scope,
    delete_enabled_features_statement,
)
from services.ml.image_pipeline import ImageFeatureVectorizer
from services.ml.training_config import (
    DEFAULT_FEATURE_METHODS,
    DEFAULT_PREPROCESSING_OPS,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureBatchExtractionConfig:
    """Configuration for database-backed feature extraction."""

    feature_type: str = "combined"
    feature_methods: list[str] | None = None
    preprocessing_ops: list[str] | None = None
    batch_size: int = 100
    replace_existing: bool = False


@dataclass(frozen=True)
class FeatureBatchExtractionResult:
    """Summary returned after batch feature extraction."""

    processed_count: int
    skipped_count: int
    failed_count: int
    feature_type: str
    feature_methods: list[str]


class FeatureBatchExtractionService:
    """Extract feature vectors for imported artworks and persist them."""

    def __init__(
        self,
        session_manager: DatabaseSessionManager,
        vectorizer: ImageFeatureVectorizer | None = None,
    ) -> None:
        """Create a feature extraction service."""
        self._session_manager = session_manager
        self._vectorizer = vectorizer or ImageFeatureVectorizer()

    def extract_features(
        self,
        config: FeatureBatchExtractionConfig | None = None,
    ) -> FeatureBatchExtractionResult:
        """Extract and store features for all imported artwork images.

        Args:
            config: Extraction options. Defaults match the training pipeline.

        Returns:
            A summary containing processed, skipped, and failed counts.
        """
        extraction_config = config or FeatureBatchExtractionConfig()
        feature_methods = (
            extraction_config.feature_methods or DEFAULT_FEATURE_METHODS.copy()
        )
        preprocessing_ops = (
            extraction_config.preprocessing_ops or DEFAULT_PREPROCESSING_OPS.copy()
        )
        self._validate_config(extraction_config, feature_methods)

        processed_count = 0
        skipped_count = 0
        failed_count = 0

        with self._session_manager.session_scope() as session:
            artworks = list(
                session.scalars(
                    apply_enabled_dataset_scope(select(Artwork)).order_by(Artwork.id),
                ),
            )
            if not artworks:
                msg = (
                    "No enabled artworks found. Please enable at least one real "
                    "dataset before feature extraction."
                )
                raise ValueError(msg)

            if extraction_config.replace_existing:
                session.execute(
                    delete_enabled_features_statement(extraction_config.feature_type),
                )
                session.flush()

            existing_artwork_ids = self._existing_artwork_ids(
                session,
                extraction_config.feature_type,
            )
            pending_features: list[Feature] = []

            for artwork in artworks:
                if artwork.id in existing_artwork_ids:
                    skipped_count += 1
                    continue
                image_path = Path(artwork.image_path)
                image = cv2.imread(str(image_path))
                if image is None:
                    failed_count += 1
                    LOGGER.warning("Skipping unreadable artwork image: %s", image_path)
                    continue
                vector = self._vectorizer.vectorize(
                    image,
                    preprocessing_ops,
                    feature_methods,
                )
                pending_features.append(
                    Feature(
                        artwork_id=artwork.id,
                        feature_type=extraction_config.feature_type,
                        vector=serialize_feature_vector(vector),
                    ),
                )
                processed_count += 1
                if len(pending_features) >= extraction_config.batch_size:
                    self._flush_features(session, pending_features)
                    LOGGER.info("Extracted %s feature vectors.", processed_count)

            if pending_features:
                self._flush_features(session, pending_features)

        return FeatureBatchExtractionResult(
            processed_count=processed_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
            feature_type=extraction_config.feature_type,
            feature_methods=feature_methods,
        )

    def _existing_artwork_ids(self, session: Session, feature_type: str) -> set[int]:
        """Return artwork ids that already have the requested feature type."""
        statement = select(Feature.artwork_id).where(Feature.feature_type == feature_type)
        return {int(artwork_id) for artwork_id in session.scalars(statement)}

    def _flush_features(self, session: Session, features: list[Feature]) -> None:
        """Persist pending features in one database round trip."""
        session.add_all(features)
        session.flush()
        features.clear()

    def _validate_config(
        self,
        config: FeatureBatchExtractionConfig,
        feature_methods: list[str],
    ) -> None:
        """Validate extraction settings."""
        if config.batch_size < 1:
            msg = "batch_size must be positive."
            raise ValueError(msg)
        if not config.feature_type.strip():
            msg = "feature_type must not be empty."
            raise ValueError(msg)
        if not feature_methods:
            msg = "feature_methods must contain at least one method."
            raise ValueError(msg)
