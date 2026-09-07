"""Repository for Style entities."""

from sqlalchemy.orm import Session

from models.style import Style
from repositories.base import BaseRepository


class StyleRepository(BaseRepository[Style]):
    """CRUD repository for styles."""

    def __init__(self, session: Session) -> None:
        """Create a style repository."""
        super().__init__(session, Style)