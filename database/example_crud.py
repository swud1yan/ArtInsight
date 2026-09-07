"""Example CRUD operations for validating the database layer."""

from __future__ import annotations

import logging

from app.config import ConfigManager, PROJECT_ROOT
from database.initializer import initialize_database
from database.session import DatabaseSessionManager
from models import Artist, Artwork, Dataset, Feature, History, PCAResult, Style
from repositories import (
    ArtistRepository,
    ArtworkRepository,
    DatasetRepository,
    FeatureRepository,
    HistoryRepository,
    PCAResultRepository,
    StyleRepository,
)

LOGGER = logging.getLogger(__name__)


def run_example(database_url: str | None = None) -> Artwork:
    """Create tables, insert sample records, and query one artwork."""
    config = ConfigManager().load()
    selected_url = database_url or config.database.url
    session_manager = DatabaseSessionManager.from_url(selected_url)
    initialize_database(session_manager.engine)

    with session_manager.session_scope() as session:
        artist_repository = ArtistRepository(session)
        style_repository = StyleRepository(session)
        dataset_repository = DatasetRepository(session)
        artwork_repository = ArtworkRepository(session)
        feature_repository = FeatureRepository(session)
        pca_repository = PCAResultRepository(session)
        history_repository = HistoryRepository(session)

        artist = artist_repository.add(Artist(name="Claude Monet"))
        style = style_repository.add(Style(name="Impressionism"))
        dataset = dataset_repository.add(
            Dataset(name="Sample Dataset", path="data/datasets/sample"),
        )
        artwork = artwork_repository.add(
            Artwork(
                title="Water Lilies",
                dataset_id=dataset.id,
                style_id=style.id,
                artist_id=artist.id,
                image_path="data/datasets/sample/water_lilies.jpg",
            ),
        )
        feature_repository.add(
            Feature(
                artwork_id=artwork.id,
                feature_type="color_histogram",
                vector=b"sample-feature-vector",
            ),
        )
        pca_repository.add(
            PCAResult(
                artwork_id=artwork.id,
                components=b"sample-components",
                explained_variance=b"sample-explained-variance",
            ),
        )
        history_repository.add(
            History(
                action="database_example_created",
                dataset_id=dataset.id,
                details=(
                    "Inserted sample artist, style, dataset, artwork, "
                    "feature, and PCA result."
                ),
            ),
        )
        queried_artwork = artwork_repository.get_by_id(artwork.id)
        if queried_artwork is None:
            msg = "Example artwork was not found after insertion."
            raise RuntimeError(msg)
        return queried_artwork


def main() -> int:
    """Run the CRUD example against a disposable local database file."""
    logging.basicConfig(level=logging.INFO)
    example_path = PROJECT_ROOT / "data" / "artinsight_example.sqlite3"
    if example_path.exists():
        example_path.unlink()
    artwork = run_example(f"sqlite:///{example_path.as_posix()}")
    LOGGER.info("Created and queried artwork: %s", artwork.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())