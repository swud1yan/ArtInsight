"""Tests for database-backed batch feature extraction behavior."""

from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np
from sqlalchemy import select

from core.features.feature_codec import deserialize_feature_vector
from database.initializer import initialize_database
from database.session import DatabaseSessionManager
from models import Artist, Artwork, Dataset, Feature, Style
from repositories import ArtistRepository, ArtworkRepository, DatasetRepository
from repositories import StyleRepository
from services.feature_batch_extractor import (
    FeatureBatchExtractionConfig,
    FeatureBatchExtractionService,
)
from tests.temp_utils import ProjectTempDirectory


class ConstantVectorizer:
    """Lightweight vectorizer used to test persistence policy only."""

    def __init__(self) -> None:
        """Create a deterministic vectorizer."""
        self.call_count = 0

    def vectorize(
        self,
        image: np.ndarray,
        preprocessing_ops: list[str],
        feature_methods: list[str],
    ) -> np.ndarray:
        """Return a different vector each time to reveal recomputation."""
        del preprocessing_ops, feature_methods
        if image.size == 0:
            msg = "image must not be empty."
            raise ValueError(msg)
        self.call_count += 1
        return np.asarray([float(self.call_count), 1.0], dtype=np.float32)


class FeatureBatchExtractionTestCase(unittest.TestCase):
    """Validate skip-existing and explicit replacement behavior."""

    def setUp(self) -> None:
        """Create a temporary database with one readable artwork image."""
        self.temp_dir = ProjectTempDirectory("feature_batch")
        self.root = Path(self.temp_dir.name)
        self.database_path = self.root / "artinsight_feature_batch.sqlite3"
        self.image_path = self.root / "sample.jpg"
        cv2.imwrite(str(self.image_path), np.full((12, 12, 3), 128, dtype=np.uint8))
        self.session_manager = DatabaseSessionManager.from_url(
            f"sqlite:///{self.database_path.as_posix()}",
        )
        initialize_database(self.session_manager.engine)
        self._insert_artwork()

    def tearDown(self) -> None:
        """Dispose database resources and clean test files."""
        self.session_manager.engine.dispose()
        self.temp_dir.cleanup()

    def test_default_extraction_skips_existing_features(self) -> None:
        """Second extraction should resume instead of recomputing existing rows."""
        vectorizer = ConstantVectorizer()
        service = FeatureBatchExtractionService(self.session_manager, vectorizer)
        config = FeatureBatchExtractionConfig(
            feature_type="combined_quick",
            feature_methods=["rgb_hist"],
        )

        first = service.extract_features(config)
        second = service.extract_features(config)

        self.assertEqual(first.processed_count, 1)
        self.assertEqual(first.skipped_count, 0)
        self.assertEqual(second.processed_count, 0)
        self.assertEqual(second.skipped_count, 1)
        self.assertEqual(vectorizer.call_count, 1)
        self.assertEqual(self._feature_count("combined_quick"), 1)

    def test_replace_existing_recomputes_selected_feature_type(self) -> None:
        """Explicit replacement should delete and rebuild the selected mode."""
        vectorizer = ConstantVectorizer()
        service = FeatureBatchExtractionService(self.session_manager, vectorizer)
        config = FeatureBatchExtractionConfig(
            feature_type="combined_quick",
            feature_methods=["rgb_hist"],
        )
        service.extract_features(config)

        replaced = service.extract_features(
            FeatureBatchExtractionConfig(
                feature_type="combined_quick",
                feature_methods=["rgb_hist"],
                replace_existing=True,
            ),
        )
        vector = self._latest_vector("combined_quick")

        self.assertEqual(replaced.processed_count, 1)
        self.assertEqual(replaced.skipped_count, 0)
        self.assertEqual(vectorizer.call_count, 2)
        np.testing.assert_allclose(vector, np.asarray([2.0, 1.0], dtype=np.float32))

    def _insert_artwork(self) -> None:
        """Insert one artwork row backed by a real image file."""
        with self.session_manager.session_scope() as session:
            artist = ArtistRepository(session).add(Artist(name="Test Artist"))
            style = StyleRepository(session).add(Style(name="Impressionism"))
            dataset = DatasetRepository(session).add(
                Dataset(name="Real Validation", path=str(self.root)),
            )
            ArtworkRepository(session).add(
                Artwork(
                    title="Readable Sample",
                    dataset_id=dataset.id,
                    style_id=style.id,
                    artist_id=artist.id,
                    image_path=str(self.image_path),
                ),
            )

    def _feature_count(self, feature_type: str) -> int:
        """Return persisted feature row count for a type."""
        with self.session_manager.session_scope() as session:
            statement = select(Feature).where(Feature.feature_type == feature_type)
            return len(list(session.scalars(statement)))

    def _latest_vector(self, feature_type: str) -> np.ndarray:
        """Return the newest stored feature vector for a type."""
        with self.session_manager.session_scope() as session:
            statement = (
                select(Feature)
                .where(Feature.feature_type == feature_type)
                .order_by(Feature.id.desc())
                .limit(1)
            )
            feature = session.scalars(statement).one()
            return deserialize_feature_vector(feature.vector)


if __name__ == "__main__":
    unittest.main()
