"""Dataset table model.

Stores imported dataset metadata and the root path used to locate artwork files.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base
from models.timestamps import utc_now


class Dataset(Base):
    """Dataset entity mapped to the datasets table."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    is_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )
    date: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )

    artworks: Mapped[list["Artwork"]] = relationship(
        "Artwork",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    history_records: Mapped[list["History"]] = relationship(
        "History",
        back_populates="dataset",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""
        return (
            "Dataset("
            f"id={self.id!r}, name={self.name!r}, path={self.path!r}, "
            f"is_enabled={self.is_enabled!r}"
            ")"
        )
