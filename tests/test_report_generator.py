"""Tests for ArtInsight report generation."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from PIL import Image
from pypdf import PdfReader

from core.features.feature_codec import serialize_feature_vector
from database.initializer import initialize_database
from database.session import DatabaseSessionManager
from models import Artist, Artwork, Dataset, Feature, Style
from repositories import ArtistRepository, ArtworkRepository, DatasetRepository
from repositories import FeatureRepository, StyleRepository
from services.ml.model_artifact import ModelArtifactStore
from services.report_data_provider import ReportDataProvider
from services.report_generator import ReportExportOptions, ReportGenerator
from tests.temp_utils import ProjectTempDirectory


class ReportGeneratorTestCase(unittest.TestCase):
    """Validate PDF and HTML report generation from database state."""

    def setUp(self) -> None:
        """Create a temporary project database with one artwork."""
        self.temp_dir = ProjectTempDirectory("report")
        self.root = Path(self.temp_dir.name)
        self.database_path = self.root / "report.sqlite3"
        self.export_dir = self.root / "exports"
        self.session_manager = DatabaseSessionManager.from_url(
            f"sqlite:///{self.database_path.as_posix()}",
        )
        initialize_database(self.session_manager.engine)
        self.image_path = self.root / "artwork.png"
        Image.new("RGB", (32, 32), color=(48, 111, 115)).save(self.image_path)
        self._insert_report_data()

    def tearDown(self) -> None:
        """Dispose database connections and remove temporary files."""
        self.session_manager.engine.dispose()
        self.temp_dir.cleanup()

    def test_export_pdf_and_html_reports(self) -> None:
        """Report generator should produce non-empty PDF and HTML files."""
        generator = ReportGenerator(
            ReportDataProvider(
                self.session_manager,
                ModelArtifactStore(self.root / "missing_model.joblib"),
            ),
            self.export_dir,
        )

        pdf = generator.export_report(
            ReportExportOptions("答辩报告", "pdf"),
        )
        html = generator.export_report(
            ReportExportOptions("答辩报告", "html"),
        )

        self.assertTrue(pdf.output_path.exists())
        self.assertGreater(pdf.output_path.stat().st_size, 1000)
        self.assertGreaterEqual(len(PdfReader(str(pdf.output_path)).pages), 3)
        self.assertTrue(html.output_path.exists())
        html_text = html.output_path.read_text(encoding="utf-8")
        self.assertIn("ArtInsight", html_text)
        self.assertIn("答辩报告", html_text)
        self.assertIn("Impressionism", html_text)

    def test_unsupported_format_raises_error(self) -> None:
        """Unsupported formats should fail clearly."""
        generator = ReportGenerator(
            ReportDataProvider(
                self.session_manager,
                ModelArtifactStore(self.root / "missing_model.joblib"),
            ),
            self.export_dir,
        )

        with self.assertRaises(ValueError):
            generator.export_report(ReportExportOptions("Report", "docx"))

    def _insert_report_data(self) -> None:
        """Insert deterministic report records."""
        with self.session_manager.session_scope() as session:
            artist = ArtistRepository(session).add(Artist(name="Claude Test"))
            style = StyleRepository(session).add(Style(name="Impressionism"))
            dataset = DatasetRepository(session).add(
                Dataset(name="Synthetic WikiArt", path=str(self.root / "dataset")),
            )
            artwork = ArtworkRepository(session).add(
                Artwork(
                    title="Synthetic Artwork",
                    dataset_id=dataset.id,
                    style_id=style.id,
                    artist_id=artist.id,
                    image_path=str(self.image_path),
                ),
            )
            FeatureRepository(session).add(
                Feature(
                    artwork_id=artwork.id,
                    feature_type="combined",
                    vector=serialize_feature_vector(
                        np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
