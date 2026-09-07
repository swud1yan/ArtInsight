"""Tests for GUI dataset page adapter queries."""

from __future__ import annotations

from pathlib import Path
import unittest

from controllers.dataset_page_adapter import DatasetPageAdapter
from database.initializer import initialize_database
from database.session import DatabaseSessionManager
from services.dataset_manager import DatasetManager
from tests.temp_utils import ProjectTempDirectory


class DatasetPageAdapterTestCase(unittest.TestCase):
    """Validate dataset list DTOs used by the GUI table."""

    def setUp(self) -> None:
        """Create a temporary database and two dataset roots."""
        self.temp_dir = ProjectTempDirectory("dataset_adapter")
        self.root = Path(self.temp_dir.name)
        self.database_path = self.root / "artinsight_dataset_adapter.sqlite3"
        self.session_manager = DatabaseSessionManager.from_url(
            f"sqlite:///{self.database_path.as_posix()}",
        )
        initialize_database(self.session_manager.engine)
        self.manager = DatasetManager(self.session_manager)
        self.adapter = DatasetPageAdapter(self.session_manager)

    def tearDown(self) -> None:
        """Dispose database connections and remove temporary files."""
        self.session_manager.engine.dispose()
        self.temp_dir.cleanup()

    def test_list_datasets_returns_aggregate_counts(self) -> None:
        """Dataset rows should include image, style, and artist counts."""
        first_root = self.root / "WikiArtA"
        second_root = self.root / "WikiArtB"
        self._write_image(first_root, "Impressionism/Claude Monet/a.jpg")
        self._write_image(first_root, "Impressionism/Claude Monet/b.png")
        self._write_image(first_root, "Cubism/Pablo Picasso/c.webp")
        self._write_image(second_root, "Baroque/Caravaggio/d.jpeg")

        self.manager.import_dataset(str(first_root))
        self.manager.import_dataset(str(second_root))

        rows = {row.name: row for row in self.adapter.list_datasets()}

        self.assertEqual(rows["WikiArtA"].image_count, 3)
        self.assertEqual(rows["WikiArtA"].style_count, 2)
        self.assertEqual(rows["WikiArtA"].artist_count, 2)
        self.assertEqual(rows["WikiArtB"].image_count, 1)
        self.assertEqual(rows["WikiArtB"].style_count, 1)
        self.assertEqual(rows["WikiArtB"].artist_count, 1)

    def _write_image(self, dataset_root: Path, relative_path: str) -> Path:
        """Write a tiny image-like file under a dataset root."""
        image_path = dataset_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nartinsight-test-image")
        return image_path


if __name__ == "__main__":
    unittest.main()
