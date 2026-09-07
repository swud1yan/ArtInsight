"""Repository for Artist entities."""

from sqlalchemy.orm import Session

from models.artist import Artist
from repositories.base import BaseRepository


class ArtistRepository(BaseRepository[Artist]):
    """CRUD repository for artists."""

    def __init__(self, session: Session) -> None:
        """Create an artist repository."""
        super().__init__(session, Artist)