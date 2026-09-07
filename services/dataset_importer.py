"""Batch artwork import helpers for dataset management."""

from __future__ import annotations

import logging
from typing import Any

from models import Artist, Artwork, Style
from repositories import ArtistRepository, StyleRepository
from services.dataset_scanner import ImageCandidate

LOGGER = logging.getLogger(__name__)


class DatasetArtworkImporter:
    """Create artwork rows while caching style and artist entities."""

    def __init__(
        self,
        session: Any,
        style_repository: StyleRepository,
        artist_repository: ArtistRepository,
    ) -> None:
        """Create an importer for an active database session."""
        self._session = session
        self._style_repository = style_repository
        self._artist_repository = artist_repository
        self._style_cache = self._load_named_entities(style_repository)
        self._artist_cache = self._load_named_entities(artist_repository)

    def add_artworks_in_batches(
        self,
        *,
        dataset_id: int,
        image_candidates: list[ImageCandidate],
        batch_size: int,
        dataset_name: str,
    ) -> int:
        """Add artwork rows in flush batches without committing per image."""
        batch: list[Artwork] = []
        imported_count = 0
        for candidate in image_candidates:
            style = self._get_or_create_style(candidate.style_name)
            artist = self._get_or_create_artist(candidate.artist_name)
            batch.append(
                Artwork(
                    title=candidate.title,
                    dataset_id=dataset_id,
                    style_id=style.id,
                    artist_id=artist.id,
                    image_path=str(candidate.path),
                ),
            )
            if len(batch) >= batch_size:
                imported_count += self._flush_artwork_batch(batch, dataset_name)
                batch.clear()

        if batch:
            imported_count += self._flush_artwork_batch(batch, dataset_name)
        return imported_count

    def _flush_artwork_batch(self, batch: list[Artwork], dataset_name: str) -> int:
        """Flush one batch of artwork rows and log progress."""
        self._session.add_all(batch)
        self._session.flush()
        LOGGER.info("Imported %s more images for dataset %s.", len(batch), dataset_name)
        return len(batch)

    def _load_named_entities(
        self,
        repository: ArtistRepository | StyleRepository,
    ) -> dict[str, Artist | Style]:
        """Load existing named entities into a cache."""
        return {entity.name: entity for entity in repository.list_all()}

    def _get_or_create_style(self, name: str) -> Style:
        """Return an existing style or create it."""
        cached = self._style_cache.get(name)
        if isinstance(cached, Style):
            return cached
        style = self._style_repository.add(Style(name=name))
        self._style_cache[name] = style
        return style

    def _get_or_create_artist(self, name: str) -> Artist:
        """Return an existing artist or create it."""
        cached = self._artist_cache.get(name)
        if isinstance(cached, Artist):
            return cached
        artist = self._artist_repository.add(Artist(name=name))
        self._artist_cache[name] = artist
        return artist