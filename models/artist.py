"""Artist table model.

Stores normalized artist names. One artist can be associated with many artworks.
"""

from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base


class Artist(Base):
    """Artist entity mapped to the artists table."""

    __tablename__ = "artists"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        unique=True,
        index=True,
    )

    artworks: Mapped[list["Artwork"]] = relationship(
        "Artwork",
        back_populates="artist",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""
        return f"Artist(id={self.id!r}, name={self.name!r})"