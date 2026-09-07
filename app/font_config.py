"""Qt font configuration helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

LOGGER = logging.getLogger(__name__)

CHINESE_FONT_FILES: tuple[Path, ...] = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
)
FALLBACK_FONT_FAMILIES: tuple[str, ...] = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
)


def configure_application_font(
    qt_app: QApplication,
    preferred_family: str,
    point_size: int,
) -> None:
    """Set a Qt application font with Chinese glyph fallback support."""
    loaded_families = _load_chinese_font_files()
    available_families = set(QFontDatabase.families()) | loaded_families
    family = _select_font_family(preferred_family, available_families)
    qt_app.setFont(QFont(family, point_size))
    LOGGER.info("Configured Qt font family: %s", family)


def _load_chinese_font_files() -> set[str]:
    """Load common local Chinese fonts and return their exposed families."""
    loaded_families: set[str] = set()
    for font_path in CHINESE_FONT_FILES:
        if not font_path.exists():
            continue
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id < 0:
            LOGGER.warning("Failed to load Qt font file: %s", font_path)
            continue
        loaded_families.update(QFontDatabase.applicationFontFamilies(font_id))
    return loaded_families


def _select_font_family(
    preferred_family: str,
    available_families: set[str],
) -> str:
    """Select the preferred configured font or the first available fallback."""
    candidates = (preferred_family, *FALLBACK_FONT_FAMILIES)
    for candidate in candidates:
        if candidate in available_families:
            return candidate
    return preferred_family
