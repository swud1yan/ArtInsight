"""Repository for PCAResult entities."""

from sqlalchemy.orm import Session

from models.pca_result import PCAResult
from repositories.base import BaseRepository


class PCAResultRepository(BaseRepository[PCAResult]):
    """CRUD repository for PCA results."""

    def __init__(self, session: Session) -> None:
        """Create a PCA result repository."""
        super().__init__(session, PCAResult)