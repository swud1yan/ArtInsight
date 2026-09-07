"""Artwork preview queries for GUI pages."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from controllers.gui_dtos import ArtworkCard
from database.session import DatabaseSessionManager
from models import Artwork, Dataset, Style


class ArtworkCardQuery:
    """Load artwork metadata as lightweight GUI card DTOs."""

    def __init__(self, session_manager: DatabaseSessionManager) -> None:
        """Create a query object with shared session management."""
        self._session_manager = session_manager

    def list_examples(
        self,
        style_label: str | None = None,
        limit: int = 12,
    ) -> list[ArtworkCard]:
        """Return artwork preview cards, optionally filtered by style."""
        with self._session_manager.session_scope() as session:
            statement = (
                select(Artwork)
                .options(joinedload(Artwork.style), joinedload(Artwork.artist))
                .join(Dataset, Artwork.dataset_id == Dataset.id)
                .where(Dataset.is_enabled.is_(True))
                .order_by(Artwork.id.desc())
                .limit(limit)
            )
            if style_label:
                statement = (
                    select(Artwork)
                    .join(Style, Artwork.style_id == Style.id)
                    .join(Dataset, Artwork.dataset_id == Dataset.id)
                    .options(joinedload(Artwork.style), joinedload(Artwork.artist))
                    .where(Style.name == style_label, Dataset.is_enabled.is_(True))
                    .order_by(Artwork.id.desc())
                    .limit(limit)
                )
            return [
                self.to_card(artwork)
                for artwork in session.scalars(statement)
            ]

    def get_cards(self, artwork_ids: list[int]) -> list[ArtworkCard]:
        """Return artwork preview cards for selected artwork IDs."""
        if not artwork_ids:
            return []
        with self._session_manager.session_scope() as session:
            cards = self.cards_by_id(session, artwork_ids)
        return [cards[artwork_id] for artwork_id in artwork_ids if artwork_id in cards]

    def cards_by_id(
        self,
        session: Session,
        artwork_ids: list[int],
    ) -> dict[int, ArtworkCard]:
        """Load artwork cards indexed by artwork ID inside an active session."""
        statement = (
            select(Artwork)
            .options(joinedload(Artwork.style), joinedload(Artwork.artist))
            .join(Dataset, Artwork.dataset_id == Dataset.id)
            .where(Artwork.id.in_(artwork_ids))
            .where(Dataset.is_enabled.is_(True))
        )
        return {
            artwork.id: self.to_card(artwork)
            for artwork in session.scalars(statement)
        }

    def to_card(self, artwork: Artwork) -> ArtworkCard:
        """Convert an ORM artwork entity to a GUI DTO."""
        return ArtworkCard(
            artwork_id=artwork.id,
            title=artwork.title,
            style_label=artwork.style.name if artwork.style else "未知风格",
            artist_name=artwork.artist.name if artwork.artist else "未知作者",
            image_path=artwork.image_path,
        )
