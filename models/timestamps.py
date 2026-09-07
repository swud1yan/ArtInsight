"""Timestamp helpers for ORM models."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current UTC time without timezone metadata for SQLite storage."""
    return datetime.now(timezone.utc).replace(tzinfo=None)