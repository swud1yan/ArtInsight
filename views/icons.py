"""Icon helpers for Qt widgets."""

from __future__ import annotations

from pathlib import Path
import re

from PySide6.QtCore import QByteArray
from PySide6.QtGui import QIcon, QPixmap

ICON_ROOT = Path(__file__).resolve().parents[1] / "resources" / "icons"
DEFAULT_ICON_COLOR = "#2F6F73"


def load_icon(name: str, color: str | None = None) -> QIcon:
    """Load an SVG icon from resources/icons by stem name.

    When ``color`` is provided, stroke/fill colors in the SVG are replaced at
    load time. This keeps primary buttons readable without duplicating icon
    assets.
    """
    icon_path = ICON_ROOT / f"{name}.svg"
    if not icon_path.exists():
        return QIcon()
    if color is None:
        return QIcon(str(icon_path))
    svg_text = _recolor_svg(icon_path.read_text(encoding="utf-8"), color)
    pixmap = QPixmap()
    pixmap.loadFromData(QByteArray(svg_text.encode("utf-8")), "SVG")
    return QIcon(pixmap)


def _recolor_svg(svg_text: str, color: str) -> str:
    """Replace visible SVG stroke/fill colors while preserving ``none`` fills."""
    recolored = re.sub(r'stroke="(?!none)[^"]+"', f'stroke="{color}"', svg_text)
    return re.sub(r'fill="(?!none)[^"]+"', f'fill="{color}"', recolored)
