"""Feature table model.

Stores serialized feature vectors extracted from artwork images.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base
from models.timestamps import utc_now


class Feature(Base):
    """Feature entity mapped to the features table."""

    __tablename__ = "features"

    id: Mapped[int] = mapped_column(primary_key=True)
    artwork_id: Mapped[int] = mapped_column(
        ForeignKey("artworks.id"),
        nullable=False,
        index=True,
    )
    feature_type: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
        default="combined",
        index=True,
    )
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )

    artwork: Mapped["Artwork"] = relationship("Artwork", back_populates="features")

    def __repr__(self) -> str:
        """Return a concise debug representation."""
        return (
            "Feature("
            f"id={self.id!r}, artwork_id={self.artwork_id!r}, "
            f"feature_type={self.feature_type!r}"
            ")"
        )
