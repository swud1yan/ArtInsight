"""Repository for Dataset entities."""

from sqlalchemy.orm import Session

from models.dataset import Dataset
from repositories.base import BaseRepository


class DatasetRepository(BaseRepository[Dataset]):
    """CRUD repository for datasets."""

    def __init__(self, session: Session) -> None:
        """Create a dataset repository."""
        super().__init__(session, Dataset)