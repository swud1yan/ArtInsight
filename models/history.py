"""History table model.

Stores user or system actions for later auditing and reproducibility.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base
from models.timestamps import utc_now


class History(Base):
    """History entity mapped to the history table."""

    __tablename__ = "history"

    id: Mapped[int] = mapped_column(primary_key=True)
    action: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"))
    details: Mapped[str | None] = mapped_column(Text)

    dataset: Mapped["Dataset | None"] = relationship(
        "Dataset",
        back_populates="history_records",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""
        return f"History(id={self.id!r}, action={self.action!r})"