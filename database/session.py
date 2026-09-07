"""Session factory management for SQLAlchemy ORM access."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from database.connection import create_database_engine


@dataclass(frozen=True)
class DatabaseSessionManager:
    """Own the SQLAlchemy engine and session factory."""

    engine: Engine
    session_factory: sessionmaker[Session]

    @classmethod
    def from_url(
        cls,
        database_url: str,
        echo: bool = False,
    ) -> "DatabaseSessionManager":
        """Create a session manager from a database connection URL."""
        engine = create_database_engine(database_url, echo=echo)
        session_factory = sessionmaker(
            bind=engine,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )
        return cls(engine=engine, session_factory=session_factory)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Provide a transactional session scope with rollback on failure."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()