"""Artwork table model.

Stores artwork metadata and foreign keys to dataset, artist, and style tables.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base
from models.timestamps import utc_now


class Artwork(Base):
    """Artwork entity mapped to the artworks table."""

    __tablename__ = "artworks"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "image_path",
            name="uq_artwork_dataset_image_path",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"),
        nullable=False,
        index=True,
    )
    style_id: Mapped[int] = mapped_column(
        ForeignKey("styles.id"),
        nullable=False,
        index=True,
    )
    artist_id: Mapped[int] = mapped_column(
        ForeignKey("artists.id"),
        nullable=False,
        index=True,
    )
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="artworks")
    artist: Mapped["Artist"] = relationship("Artist", back_populates="artworks")
    style: Mapped["Style"] = relationship("Style", back_populates="artworks")
    features: Mapped[list["Feature"]] = relationship(
        "Feature",
        back_populates="artwork",
        cascade="all, delete-orphan",
    )
    pca_results: Mapped[list["PCAResult"]] = relationship(
        "PCAResult",
        back_populates="artwork",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""
        return (
            "Artwork("
            f"id={self.id!r}, title={self.title!r}, image_path={self.image_path!r}"
            ")"
        )
