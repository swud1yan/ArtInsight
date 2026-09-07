"""End-to-end integration tests for the ArtInsight workflow."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from pypdf import PdfReader

from core.features.feature_codec import serialize_feature_vector
from database.initializer import initialize_database
from database.session import DatabaseSessionManager
from models import Artist, Artwork, Dataset, Feature, Style
from repositories import ArtistRepository, ArtworkRepository, DatasetRepository
from repositories import FeatureRepository, StyleRepository
from services.ml import ModelTrainer, SimilarArtworkRetriever, StyleClassifier
from services.ml.model_artifact import ModelArtifactStore
from services.ml.prediction_pipeline import PredictionPipeline
from services.ml.trainer import TrainingConfig
from services.report_data_provider import ReportDataProvider
from services.report_generator import ReportExportOptions, ReportGenerator
from tests.temp_utils import ProjectTempDirectory


class EndToEndWorkflowTestCase(unittest.TestCase):
    """Validate the training, prediction, retrieval, and report workflow."""

    def setUp(self) -> None:
        """Create a temporary project database and synthetic features."""
        self.temp_dir = ProjectTempDirectory("e2e")
        self.root = Path(self.temp_dir.name)
        self.database_path = self.root / "artinsight_e2e.sqlite3"
        self.model_path = self.root / "models" / "style_model.joblib"
        self.export_dir = self.root / "exports"
        self.session_manager = DatabaseSessionManager.from_url(
            f"sqlite:///{self.database_path.as_posix()}",
        )
        initialize_database(self.session_manager.engine)
        self._insert_training_data()

    def tearDown(self) -> None:
        """Dispose database connections and remove temporary files."""
        self.session_manager.engine.dispose()
        self.temp_dir.cleanup()

    def test_train_predict_retrieve_and_export_report(self) -> None:
        """A complete backend workflow should run without interface breaks."""
        store = ModelArtifactStore(self.model_path)
        trainer = ModelTrainer(self.session_manager, store)
        result = trainer.train_model(
            TrainingConfig(
                n_components=2,
                n_neighbors=1,
                cross_validation_folds=2,
                model_path=self.model_path,
            ),
        )
        pipeline = PredictionPipeline(store)
        classification = StyleClassifier(pipeline).classify_feature(
            np.array([1.02, 0.98, 1.0, 1.1], dtype=np.float32),
        )
        similar = SimilarArtworkRetriever(pipeline).retrieve_feature(
            np.array([1.02, 0.98, 1.0, 1.1], dtype=np.float32),
            top_n=3,
        )
        report = ReportGenerator(
            ReportDataProvider(self.session_manager, store),
            self.export_dir,
        ).export_report(ReportExportOptions("EndToEnd_Report", "pdf"))

        self.assertEqual(result.sample_count, 8)
        self.assertEqual(classification.style_label, "Impressionism")
        self.assertEqual(len(similar), 3)
        self.assertTrue(report.output_path.exists())
        self.assertGreaterEqual(len(PdfReader(str(report.output_path)).pages), 3)

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
            self._insert_vectors(
                artwork_repository,
                feature_repository,
                dataset.id,
                artist.id,
                impressionism.id,
                "Impressionism",
                impressionism_vectors,
            )
            self._insert_vectors(
                artwork_repository,
                feature_repository,
                dataset.id,
                artist.id,
                cubism.id,
                "Cubism",
                cubism_vectors,
            )

    def _insert_vectors(
        self,
        artwork_repository: ArtworkRepository,
        feature_repository: FeatureRepository,
        dataset_id: int,
        artist_id: int,
        style_id: int,
        style_name: str,
        vectors: list[list[float]],
    ) -> None:
        """Insert artwork rows and their feature vectors."""
        for index, vector in enumerate(vectors):
            artwork = artwork_repository.add(
                Artwork(
                    title=f"{style_name} {index}",
                    dataset_id=dataset_id,
                    style_id=style_id,
                    artist_id=artist_id,
                    image_path=str(self.root / f"{style_name}_{index}.jpg"),
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
