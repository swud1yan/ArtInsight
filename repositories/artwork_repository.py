"""Repository for Artwork entities."""

from sqlalchemy.orm import Session

from models.artwork import Artwork
from repositories.base import BaseRepository


class ArtworkRepository(BaseRepository[Artwork]):
    """CRUD repository for artworks."""

    def __init__(self, session: Session) -> None:
        """Create an artwork repository."""
        super().__init__(session, Artwork)