"""Entry point for the ArtInsight desktop application."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from PySide6.QtCore import QTimer

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.bootstrap import create_application


def main() -> int:
    """Start the PySide6 application event loop."""
    context = create_application(sys.argv)
    if _is_browser_hosted_desktop():
        _show_with_novnc_margin(context)
    elif _should_show_maximized():
        context.main_window.showMaximized()
    else:
        context.main_window.show()
    return context.qt_app.exec()


def _is_browser_hosted_desktop() -> bool:
    """Detect the Xvfb desktop used by the noVNC deployment."""
    return os.getenv("DISPLAY") == ":99"


def _show_with_novnc_margin(context) -> None:
    """Leave room for the noVNC toolbar while fitting the whole sidebar."""
    screen = context.qt_app.primaryScreen()
    if screen is None:
        context.main_window.showMaximized()
        return
    geometry = screen.availableGeometry()
    left_margin = int(os.getenv("ARTINSIGHT_NOVNC_LEFT_MARGIN", "72"))
    width = max(
        context.main_window.minimumWidth(),
        geometry.width() - left_margin,
    )
    def apply_geometry() -> None:
        context.main_window.setGeometry(
            geometry.x() + left_margin,
            geometry.y(),
            width,
            geometry.height(),
        )

    context.main_window.show()
    apply_geometry()
    QTimer.singleShot(250, apply_geometry)


def _should_show_maximized() -> bool:
    """Honor an explicit request to maximize the main window."""
    value = os.getenv("ARTINSIGHT_SHOW_MAXIMIZED")
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
