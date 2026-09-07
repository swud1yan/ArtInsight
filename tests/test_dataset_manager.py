"""Tests for dataset management workflows."""

from __future__ import annotations

from pathlib import Path
import unittest

from database.initializer import initialize_database
from database.session import DatabaseSessionManager
from repositories.exceptions import EntityConflictError, EntityNotFoundError
from services.dataset_manager import DatasetManager
from tests.temp_utils import ProjectTempDirectory


class DatasetManagerTestCase(unittest.TestCase):
    """Validate dataset import, statistics, and deletion behavior."""

    def setUp(self) -> None:
        """Create a temporary database and image dataset."""
        self.temp_dir = ProjectTempDirectory("dataset")
        self.root = Path(self.temp_dir.name)
        self.database_path = self.root / "artinsight_test.sqlite3"
        self.dataset_root = self.root / "WikiArt"
        self.dataset_root.mkdir()
        self.session_manager = DatabaseSessionManager.from_url(
            f"sqlite:///{self.database_path.as_posix()}",
        )
        initialize_database(self.session_manager.engine)
        self.manager = DatasetManager(self.session_manager)

    def tearDown(self) -> None:
        """Remove temporary test files."""
        self.session_manager.engine.dispose()
        self.temp_dir.cleanup()

    def test_import_dataset_creates_records_and_statistics(self) -> None:
        """Import should create dataset, artworks, styles, artists, and stats."""
        self._write_image("Impressionism/Claude Monet/water_lilies.jpg")
        self._write_image("Impressionism/Claude Monet/garden.png")
        self._write_image("Cubism/Pablo Picasso/guitar.webp")
        self._write_text_file("notes.txt", "not an image")

        result = self.manager.import_dataset(str(self.dataset_root), batch_size=2)
        statistics = self.manager.get_statistics(result.dataset_id)
        datasets = self.manager.list_datasets()

        self.assertEqual(result.imported_count, 3)
        self.assertEqual(result.skipped_count, 0)
        self.assertEqual(result.style_count, 2)
        self.assertEqual(result.artist_count, 2)
        self.assertEqual(len(datasets), 1)
        self.assertEqual(statistics.image_count, 3)
        self.assertEqual(statistics.styles["Impressionism"], 2)
        self.assertEqual(statistics.styles["Cubism"], 1)
        self.assertEqual(statistics.artists["Claude Monet"], 2)
        self.assertEqual(statistics.artists["Pablo Picasso"], 1)

    def test_import_dataset_rejects_existing_path(self) -> None:
        """The same dataset path should not be imported twice."""
        self._write_image("Impressionism/Claude Monet/water_lilies.jpg")

        self.manager.import_dataset(str(self.dataset_root))

        with self.assertLogs("services.dataset_manager", level="WARNING"):
            with self.assertRaises(EntityConflictError):
                self.manager.import_dataset(str(self.dataset_root))

    def test_delete_dataset_requires_confirmation(self) -> None:
        """Deletion should require explicit confirmation."""
        self._write_image("Impressionism/Claude Monet/water_lilies.jpg")
        result = self.manager.import_dataset(str(self.dataset_root))

        with self.assertRaises(PermissionError):
            self.manager.delete_dataset(result.dataset_id)

    def test_delete_dataset_removes_database_records_and_files(self) -> None:
        """Deletion should remove artwork records and image files."""
        image_path = self._write_image("Impressionism/Claude Monet/water_lilies.jpg")
        result = self.manager.import_dataset(str(self.dataset_root))

        self.manager.delete_dataset(result.dataset_id, confirm=True)

        self.assertFalse(image_path.exists())
        self.assertEqual(self.manager.list_datasets(), [])
        with self.assertRaises(EntityNotFoundError):
            self.manager.get_statistics(result.dataset_id)

    def test_flat_dataset_uses_unknown_labels(self) -> None:
        """Images directly under the dataset root should receive fallback labels."""
        self._write_image("untitled.jpeg")

        result = self.manager.import_dataset(str(self.dataset_root))
        statistics = self.manager.get_statistics(result.dataset_id)

        self.assertEqual(statistics.styles["Unknown Style"], 1)
        self.assertEqual(statistics.artists["Unknown Artist"], 1)

    def _write_image(self, relative_path: str) -> Path:
        """Write a tiny image-like binary file for import tests."""
        image_path = self.dataset_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nartinsight-test-image")
        return image_path

    def _write_text_file(self, relative_path: str, content: str) -> Path:
        """Write a non-image file under the dataset root."""
        text_path = self.dataset_root / relative_path
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(content, encoding="utf-8")
        return text_path


if __name__ == "__main__":
    unittest.main()
