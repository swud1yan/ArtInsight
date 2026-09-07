"""Tests for the ArtInsight database access layer."""

from __future__ import annotations

import unittest

from sqlalchemy import inspect, text

from database.initializer import initialize_database
from database.session import DatabaseSessionManager
from models import Artist, Artwork, Dataset, Feature, History, PCAResult, Style
from repositories import (
    ArtistRepository,
    ArtworkRepository,
    DatasetRepository,
    EntityConflictError,
    FeatureRepository,
    HistoryRepository,
    PCAResultRepository,
    StyleRepository,
)


class DatabaseLayerTestCase(unittest.TestCase):
    """Validate ORM table creation and repository CRUD behavior."""

    def setUp(self) -> None:
        """Create a fresh in-memory database for each test."""
        self.session_manager = DatabaseSessionManager.from_url("sqlite:///:memory:")
        initialize_database(self.session_manager.engine)

    def tearDown(self) -> None:
        """Dispose database connections after each test."""
        self.session_manager.engine.dispose()

    def test_repositories_insert_query_and_load_relationships(self) -> None:
        """Repositories should insert, query, and load related ORM entities."""
        with self.session_manager.session_scope() as session:
            artist = ArtistRepository(session).add(Artist(name="Claude Monet"))
            style = StyleRepository(session).add(Style(name="Impressionism"))
            dataset = DatasetRepository(session).add(
                Dataset(name="Sample Dataset", path="data/datasets/sample"),
            )
            artwork_repository = ArtworkRepository(session)
            artwork = artwork_repository.add(
                Artwork(
                    title="Water Lilies",
                    dataset_id=dataset.id,
                    style_id=style.id,
                    artist_id=artist.id,
                    image_path="data/datasets/sample/water_lilies.jpg",
                ),
            )
            FeatureRepository(session).add(
                Feature(
                    artwork_id=artwork.id,
                    feature_type="color_histogram",
                    vector=b"sample-feature-vector",
                ),
            )
            PCAResultRepository(session).add(
                PCAResult(
                    artwork_id=artwork.id,
                    components=b"sample-components",
                    explained_variance=b"sample-explained-variance",
                ),
            )
            HistoryRepository(session).add(
                History(
                    action="database_test_created",
                    dataset_id=dataset.id,
                    details="Created sample relational data.",
                ),
            )

            queried = artwork_repository.get_by_id(artwork.id)

            self.assertIsNotNone(queried)
            assert queried is not None
            self.assertEqual(queried.artist.name, "Claude Monet")
            self.assertEqual(queried.style.name, "Impressionism")
            self.assertEqual(queried.dataset.name, "Sample Dataset")
            self.assertEqual(len(queried.features), 1)
            self.assertEqual(len(queried.pca_results), 1)

    def test_repository_reports_unique_constraint_conflict(self) -> None:
        """Repository should convert uniqueness failures to domain errors."""
        with self.assertRaises(EntityConflictError):
            with self.session_manager.session_scope() as session:
                repository = ArtistRepository(session)
                repository.add(Artist(name="Claude Monet"))
                repository.add(Artist(name="Claude Monet"))

    def test_repository_reports_foreign_key_conflict(self) -> None:
        """Repository should convert foreign key failures to domain errors."""
        with self.assertRaises(EntityConflictError):
            with self.session_manager.session_scope() as session:
                FeatureRepository(session).add(
                    Feature(
                        artwork_id=999,
                        feature_type="color_histogram",
                        vector=b"invalid-feature-vector",
                    ),
                )

    def test_performance_indexes_exist_for_frequent_queries(self) -> None:
        """High-frequency filter columns should be indexed."""
        artwork_indexes = {
            column.name
            for index in Artwork.__table__.indexes
            for column in index.columns
        }
        feature_indexes = {
            column.name
            for index in Feature.__table__.indexes
            for column in index.columns
        }

        self.assertIn("dataset_id", artwork_indexes)
        self.assertIn("style_id", artwork_indexes)
        self.assertIn("artist_id", artwork_indexes)
        self.assertIn("artwork_id", feature_indexes)
        self.assertIn("feature_type", feature_indexes)

    def test_initializer_adds_dataset_scope_column_to_legacy_database(self) -> None:
        """Older SQLite files should receive the dataset scope column."""
        legacy_manager = DatabaseSessionManager.from_url("sqlite:///:memory:")
        try:
            with legacy_manager.engine.begin() as connection:
                connection.execute(
                    text(
                        "CREATE TABLE datasets ("
                        "id INTEGER PRIMARY KEY, "
                        "name VARCHAR(160) NOT NULL, "
                        "path VARCHAR(500) NOT NULL UNIQUE, "
                        "date DATETIME NOT NULL"
                        ")",
                    ),
                )

            initialize_database(legacy_manager.engine)

            columns = {
                column["name"]
                for column in inspect(legacy_manager.engine).get_columns("datasets")
            }
            self.assertIn("is_enabled", columns)
        finally:
            legacy_manager.engine.dispose()


if __name__ == "__main__":
    unittest.main()
