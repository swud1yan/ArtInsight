"""Repository for Feature entities."""

from sqlalchemy.orm import Session

from models.feature import Feature
from repositories.base import BaseRepository


class FeatureRepository(BaseRepository[Feature]):
    """CRUD repository for feature vectors."""

    def __init__(self, session: Session) -> None:
        """Create a feature repository."""
        super().__init__(session, Feature)