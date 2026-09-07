"""Repository for History entities."""

from sqlalchemy.orm import Session

from models.history import History
from repositories.base import BaseRepository


class HistoryRepository(BaseRepository[History]):
    """CRUD repository for history records."""

    def __init__(self, session: Session) -> None:
        """Create a history repository."""
        super().__init__(session, History)