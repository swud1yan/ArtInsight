"""Toolbar widget for the PCA style map page."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from views.ui_helpers import create_button
from views.widgets import button_row


class StyleMapToolbar(QWidget):
    """Top-level actions used by the interactive style map."""

    load_requested = Signal()
    upload_requested = Signal()
    clear_requested = Signal()
    show_2d_requested = Signal()
    show_3d_requested = Signal()

    def __init__(self) -> None:
        """Create style map action buttons."""
        super().__init__()
        self._load_button = create_button("加载地图", "chart", primary=True)
        self._upload_button = create_button("上传定位", "image")
        self._clear_button = create_button("清除高亮", "refresh")
        self._show_2d_button = create_button("二维视图", "chart")
        self._show_3d_button = create_button("三维视图", "chart")
        self._setup_ui()
        self.set_map_actions_enabled(False)

    def set_loading(self, is_loading: bool) -> None:
        """Disable the load action while map data is loading."""
        self._load_button.setEnabled(not is_loading)

    def set_upload_enabled(self, enabled: bool) -> None:
        """Enable or disable uploaded-image projection."""
        self._upload_button.setEnabled(enabled)

    def set_3d_enabled(self, enabled: bool) -> None:
        """Enable or disable the 3-D map view action."""
        self._show_3d_button.setEnabled(enabled)

    def set_map_actions_enabled(self, enabled: bool) -> None:
        """Enable or disable actions that require loaded map data."""
        for button in (
            self._upload_button,
            self._clear_button,
            self._show_2d_button,
            self._show_3d_button,
        ):
            button.setEnabled(enabled)

    def _setup_ui(self) -> None:
        """Wire button signals and layout the toolbar."""
        self._load_button.clicked.connect(self.load_requested.emit)
        self._upload_button.clicked.connect(self.upload_requested.emit)
        self._clear_button.clicked.connect(self.clear_requested.emit)
        self._show_2d_button.clicked.connect(self.show_2d_requested.emit)
        self._show_3d_button.clicked.connect(self.show_3d_requested.emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            button_row(
                self._load_button,
                self._upload_button,
                self._clear_button,
                self._show_2d_button,
                self._show_3d_button,
            ),
        )
