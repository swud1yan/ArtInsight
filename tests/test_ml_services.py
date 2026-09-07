"""Tests for model training, style classification, and retrieval services."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from sqlalchemy import select

from core.features.feature_codec import (
    deserialize_feature_vector,
    serialize_feature_vector,
)
from database.initializer import initialize_database
from database.session import DatabaseSessionManager
from models import Artist, Artwork, Dataset, Feature, Style
from repositories import ArtistRepository, ArtworkRepository, DatasetRepository
from repositories import FeatureRepository, StyleRepository
from services.ml import (
    ModelTrainer,
    SimilarArtworkRetriever,
    StyleClassifier,
    TrainingConfig,
)
from services.ml.model_artifact import ModelArtifactStore
from services.ml.prediction_pipeline import PredictionPipeline
from tests.temp_utils import ProjectTempDirectory


class StubVectorizer:
    """Return a fixed feature vector for image-path service tests."""

    def __init__(self, feature_vector: np.ndarray) -> None:
        """Create a vectorizer with a deterministic output vector."""
        self._feature_vector = feature_vector.astype(np.float32)

    def vectorize(
        self,
        image: np.ndarray,
        preprocessing_ops: list[str],
        feature_methods: list[str],
    ) -> np.ndarray:
        """Return the configured vector after validating service metadata."""
        if image.size == 0:
            msg = "image must not be empty."
            raise ValueError(msg)
        if not preprocessing_ops:
            msg = "preprocessing_ops must not be empty."
            raise ValueError(msg)
        if not feature_methods:
            msg = "feature_methods must not be empty."
            raise ValueError(msg)
        return self._feature_vector.copy()


class MLServicesTestCase(unittest.TestCase):
    """Validate the training, classification, and retrieval workflow."""

    def setUp(self) -> None:
        """Create a temporary database and deterministic training vectors."""
        self.temp_dir = ProjectTempDirectory("ml")
        self.root = Path(self.temp_dir.name)
        self.database_path = self.root / "artinsight_ml_test.sqlite3"
        self.model_path = self.root / "models" / "style_model.joblib"
        self.session_manager = DatabaseSessionManager.from_url(
            f"sqlite:///{self.database_path.as_posix()}",
        )
        initialize_database(self.session_manager.engine)
        self._insert_training_data()

    def tearDown(self) -> None:
        """Dispose database connections and remove temporary files."""
        self.session_manager.engine.dispose()
        self.temp_dir.cleanup()

    def test_feature_vector_codec_round_trips_numpy_arrays(self) -> None:
        """Feature vectors should serialize and deserialize without shape loss."""
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        restored = deserialize_feature_vector(serialize_feature_vector(vector))

        self.assertEqual(restored.dtype, np.float32)
        np.testing.assert_allclose(restored, vector)

    def test_train_classify_and_retrieve_from_features(self) -> None:
        """Training should persist a model usable for classification and retrieval."""
        store = ModelArtifactStore(self.model_path)
        trainer = ModelTrainer(self.session_manager, store)
        config = TrainingConfig(
            n_components=2,
            n_neighbors=1,
            cross_validation_folds=2,
            model_path=self.model_path,
        )

        result = trainer.train_model(config)
        pipeline = PredictionPipeline(store)
        classifier = StyleClassifier(pipeline)
        retriever = SimilarArtworkRetriever(pipeline)
        classification = classifier.classify_feature(
            np.array([1.05, 0.95, 1.0, 1.1], dtype=np.float32),
        )
        similar = retriever.retrieve_feature(
            np.array([1.05, 0.95, 1.0, 1.1], dtype=np.float32),
            top_n=3,
        )

        self.assertTrue(result.model_path.exists())
        self.assertEqual(result.sample_count, 8)
        self.assertEqual(result.feature_count, 4)
        self.assertEqual(result.style_count, 2)
        self.assertIsNotNone(result.cross_validation_accuracy)
        assert result.cross_validation_accuracy is not None
        self.assertGreaterEqual(result.cross_validation_accuracy, 0.75)
        self.assertEqual(classification.style_label, "Impressionism")
        self.assertGreaterEqual(classification.confidence, 0.0)
        self.assertLessEqual(classification.confidence, 1.0)
        self.assertEqual(len(similar), 3)
        self.assertEqual(similar[0].style_label, "Impressionism")
        self.assertLessEqual(similar[0].distance, similar[-1].distance)

    def test_auto_component_selection_records_candidate_results(self) -> None:
        """Automatic PCA dimension search should pick and persist a valid K."""
        store = ModelArtifactStore(self.model_path)
        trainer = ModelTrainer(self.session_manager, store)
        config = TrainingConfig(
            n_components=4,
            n_neighbors=1,
            cross_validation_folds=2,
            model_path=self.model_path,
            auto_select_components=True,
            component_candidates=(1, 2, 4),
        )

        result = trainer.train_model(config)
        artifact = store.load()

        self.assertIn(result.n_components, {1, 2, 4})
        self.assertEqual(artifact.metadata.n_components, result.n_components)
        self.assertGreaterEqual(len(result.component_search_rows), 1)
        self.assertIsNotNone(result.component_selection_note)

    def test_image_entrypoints_use_configured_vectorizer(self) -> None:
        """Image classification and retrieval should share trained metadata."""
        store = ModelArtifactStore(self.model_path)
        trainer = ModelTrainer(self.session_manager, store)
        trainer.train_model(
            TrainingConfig(
                n_components=2,
                n_neighbors=1,
                cross_validation_folds=2,
                model_path=self.model_path,
            ),
        )
        pipeline = PredictionPipeline(store)
        vectorizer = StubVectorizer(
            np.array([4.95, 5.05, 5.0, 4.9], dtype=np.float32),
        )
        image = np.ones((8, 8, 3), dtype=np.uint8)
        classifier = StyleClassifier(pipeline, vectorizer=vectorizer)
        retriever = SimilarArtworkRetriever(pipeline, vectorizer=vectorizer)

        classification = classifier.classify(image)
        similar = retriever.retrieve_similar(image, top_n=2)

        self.assertEqual(classification.style_label, "Cubism")
        self.assertEqual(len(similar), 2)
        self.assertEqual(similar[0].style_label, "Cubism")

    def test_retrieval_pagination_skips_nearest_results(self) -> None:
        """Retriever should support offset-based pagination."""
        store = ModelArtifactStore(self.model_path)
        trainer = ModelTrainer(self.session_manager, store)
        trainer.train_model(
            TrainingConfig(
                n_components=2,
                n_neighbors=1,
                cross_validation_folds=2,
                model_path=self.model_path,
            ),
        )
        retriever = SimilarArtworkRetriever(PredictionPipeline(store))
        feature = np.array([4.95, 5.05, 5.0, 4.9], dtype=np.float32)

        first_page = retriever.retrieve_feature(feature, top_n=2)
        second_page = retriever.retrieve_feature(feature, top_n=2, offset=2)

        self.assertEqual(len(first_page), 2)
        self.assertEqual(len(second_page), 2)
        self.assertNotEqual(first_page[0].artwork_id, second_page[0].artwork_id)
        self.assertEqual(first_page[0].style_label, "Cubism")

    def test_training_ignores_disabled_datasets(self) -> None:
        """Disabled datasets should not leak into scoped model training."""
        with self.session_manager.session_scope() as session:
            artist = session.scalars(select(Artist).limit(1)).one()
            style = session.scalars(select(Style).limit(1)).one()
            disabled = DatasetRepository(session).add(
                Dataset(
                    name="Disabled Scope",
                    path=str(self.root / "disabled"),
                    is_enabled=False,
                ),
            )
            artwork = ArtworkRepository(session).add(
                Artwork(
                    title="Disabled Artwork",
                    dataset_id=disabled.id,
                    style_id=style.id,
                    artist_id=artist.id,
                    image_path=str(self.root / "disabled.jpg"),
                ),
            )
            FeatureRepository(session).add(
                Feature(
                    artwork_id=artwork.id,
                    feature_type="combined",
                    vector=serialize_feature_vector(
                        np.array([99.0, 99.0, 99.0, 99.0], dtype=np.float32),
                    ),
                ),
            )

        result = ModelTrainer(
            self.session_manager,
            ModelArtifactStore(self.model_path),
        ).train_model(
            TrainingConfig(
                n_components=2,
                n_neighbors=1,
                cross_validation_folds=2,
                model_path=self.model_path,
            ),
        )

        self.assertEqual(result.sample_count, 8)

    def _insert_training_data(self) -> None:
        """Insert two separable style clusters with serialized features."""
        impressionism_vectors = [
            [1.0, 1.0, 1.0, 1.0],
            [1.1, 0.9, 1.0, 1.2],
            [0.9, 1.2, 0.8, 1.1],
            [1.2, 1.0, 1.1, 0.9],
        ]
        cubism_vectors = [
            [5.0, 5.0, 5.0, 5.0],
            [4.9, 5.1, 5.0, 4.8],
            [5.2, 4.8, 5.1, 5.0],
            [4.8, 5.0, 4.9, 5.2],
        ]
        with self.session_manager.session_scope() as session:
            artist = ArtistRepository(session).add(Artist(name="Test Artist"))
            impressionism = StyleRepository(session).add(Style(name="Impressionism"))
            cubism = StyleRepository(session).add(Style(name="Cubism"))
            dataset = DatasetRepository(session).add(
                Dataset(name="Synthetic", path=str(self.root / "synthetic")),
            )
            artwork_repository = ArtworkRepository(session)
            feature_repository = FeatureRepository(session)
            for index, vector in enumerate(impressionism_vectors):
                artwork = artwork_repository.add(
                    Artwork(
                        title=f"Impressionism {index}",
                        dataset_id=dataset.id,
                        style_id=impressionism.id,
                        artist_id=artist.id,
                        image_path=str(self.root / f"imp_{index}.jpg"),
                    ),
                )
                feature_repository.add(
                    Feature(
                        artwork_id=artwork.id,
                        feature_type="combined",
                        vector=serialize_feature_vector(np.asarray(vector)),
                    ),
                )
            for index, vector in enumerate(cubism_vectors):
                artwork = artwork_repository.add(
                    Artwork(
                        title=f"Cubism {index}",
                        dataset_id=dataset.id,
                        style_id=cubism.id,
                        artist_id=artist.id,
                        image_path=str(self.root / f"cub_{index}.jpg"),
                    ),
                )
                feature_repository.add(
                    Feature(
                        artwork_id=artwork.id,
                        feature_type="combined",
                        vector=serialize_feature_vector(np.asarray(vector)),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
