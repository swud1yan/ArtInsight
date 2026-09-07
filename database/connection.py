"""SQLAlchemy engine creation utilities."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event

from app.config import PROJECT_ROOT


def normalize_sqlite_url(database_url: str) -> str:
    """Normalize a SQLite URL so relative database paths are project-relative."""
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return database_url

    raw_path = database_url.removeprefix(prefix)
    if raw_path in {":memory:", ""}:
        return database_url

    database_path = Path(raw_path)
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{database_path.as_posix()}"


def create_database_engine(database_url: str, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine and enable SQLite foreign key checks."""
    normalized_url = normalize_sqlite_url(database_url)
    if normalized_url.startswith("sqlite"):
        engine = create_engine(
            normalized_url,
            echo=echo,
            future=True,
            connect_args={"check_same_thread": False},
        )
        _enable_sqlite_foreign_keys(engine)
        return engine
    return create_engine(normalized_url, echo=echo, future=True)


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """Ensure SQLite enforces declared foreign keys for every connection."""

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
