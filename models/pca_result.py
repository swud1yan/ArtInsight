"""PCA result table model.

Stores serialized PCA components and explained variance data for an artwork.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base
from models.timestamps import utc_now


class PCAResult(Base):
    """PCA result entity mapped to the pca_results table."""

    __tablename__ = "pca_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    artwork_id: Mapped[int] = mapped_column(ForeignKey("artworks.id"), nullable=False)
    components: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    explained_variance: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )

    artwork: Mapped["Artwork"] = relationship("Artwork", back_populates="pca_results")

    def __repr__(self) -> str:
        """Return a concise debug representation."""
        return f"PCAResult(id={self.id!r}, artwork_id={self.artwork_id!r})"