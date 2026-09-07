"""Centralized logging setup for ArtInsight."""

from __future__ import annotations

import logging
from pathlib import Path


def initialize_logging(log_dir: Path, level: str, file_name: str) -> None:
    """Initialize root logging with file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    file_handler = logging.FileHandler(log_dir / file_name, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=numeric_level,
        handlers=[file_handler, stream_handler],
        force=True,
    )
