"""Database initialization helpers."""

from __future__ import annotations

import logging

from sqlalchemy import Engine, inspect, text

from models import Base

LOGGER = logging.getLogger(__name__)


def initialize_database(engine: Engine) -> None:
    """Create all ORM tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    _ensure_dataset_scope_columns(engine)
    LOGGER.info("Database tables are ready.")


def _ensure_dataset_scope_columns(engine: Engine) -> None:
    """Add dataset-scope columns for databases created by older versions."""
    inspector = inspect(engine)
    if "datasets" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("datasets")}
    if "is_enabled" in columns:
        return
    with engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE datasets ADD COLUMN is_enabled BOOLEAN NOT NULL DEFAULT 1"),
        )
    LOGGER.info("Added datasets.is_enabled compatibility column.")
