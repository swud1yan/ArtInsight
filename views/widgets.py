"""Reusable widgets for ArtInsight GUI pages."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from controllers.gui_dtos import ArtworkCard
from services.style_translations import format_style_label


class PageHeader(QWidget):
    """Consistent page title and subtitle block."""

    def __init__(self, title: str, subtitle: str) -> None:
        """Create a header widget."""
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(6)
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("pageSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)


class MetricCard(QFrame):
    """Small dashboard metric card."""

    def __init__(self, title: str, value: str = "0") -> None:
        """Create a metric card with title and value labels."""
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)
        self._value_label = QLabel(value)
        self._value_label.setObjectName("metricValue")
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        layout.addWidget(self._value_label)
        layout.addWidget(title_label)

    def set_value(self, value: object) -> None:
        """Update the metric value text."""
        self._value_label.setText(str(value))


class ImagePreviewLabel(QLabel):
    """QLabel that displays an image scaled to its current size."""

    def __init__(self, placeholder: str = "未选择图片") -> None:
        """Create an adaptive image preview label."""
        super().__init__(placeholder)
        self.setObjectName("imagePreview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setWordWrap(True)
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder

    def set_image_path(self, image_path: Path | str | None) -> None:
        """Load and display an image file path."""
        if image_path is None:
            self._pixmap = None
            self.setPixmap(QPixmap())
            self.setText(self._placeholder)
            return
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._pixmap = None
            self.setPixmap(QPixmap())
            self.setText("无法预览图片")
            return
        self._pixmap = pixmap
        self.setText("")
        self._render_pixmap()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Re-render pixmap when the preview size changes."""
        super().resizeEvent(event)
        self._render_pixmap()

    def sizeHint(self) -> QSize:
        """Return a stable size hint so loading a large image cannot shift layout."""
        return self.minimumSize()

    def _render_pixmap(self) -> None:
        """Scale the current pixmap to fit the label."""
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class ArtworkCardWidget(QFrame):
    """Clickable artwork preview card."""

    selected = Signal(int)

    def __init__(self, card: ArtworkCard, extra_text: str = "") -> None:
        """Create a card for an artwork DTO."""
        super().__init__()
        self._card = card
        self.setObjectName("artworkCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        preview = ImagePreviewLabel("无预览")
        preview.setMinimumSize(120, 90)
        preview.setMaximumHeight(130)
        preview.set_image_path(Path(card.image_path))
        title = QLabel(card.title)
        title.setObjectName("cardTitle")
        title.setWordWrap(True)
        subtitle = QLabel(f"{format_style_label(card.style_label)} · {card.artist_name}")
        subtitle.setObjectName("cardSubtitle")
        subtitle.setWordWrap(True)
        detail = QLabel(extra_text)
        detail.setObjectName("cardSubtitle")
        button = QPushButton("查看详情")
        button.clicked.connect(lambda: self.selected.emit(card.artwork_id))
        layout.addWidget(preview)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        if extra_text:
            layout.addWidget(detail)
        layout.addWidget(button)


class ArtworkGridWidget(QWidget):
    """Grid container for artwork cards."""

    artwork_selected = Signal(int)

    def __init__(self, columns: int = 4, empty_message: str = "请选择图片后开始分析") -> None:
        """Create a grid with a fixed maximum column count."""
        super().__init__()
        self.setObjectName("artworkGrid")
        self._columns = max(1, columns)
        self._empty_message = empty_message
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(14)
        self._layout.setVerticalSpacing(14)
        self.set_cards([])

    def set_cards(self, cards: Sequence[tuple[ArtworkCard, str]]) -> None:
        """Replace all artwork cards in the grid."""
        self.clear()
        if not cards:
            empty = QLabel(self._empty_message)
            empty.setObjectName("emptyLabel")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setMinimumHeight(220)
            self._layout.addWidget(empty, 0, 0)
            return
        for index, (card, extra_text) in enumerate(cards):
            widget = ArtworkCardWidget(card, extra_text)
            widget.selected.connect(self.artwork_selected)
            row = index // self._columns
            column = index % self._columns
            self._layout.addWidget(widget, row, column)

    def clear(self) -> None:
        """Remove all child widgets from the grid."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


def show_artwork_detail(parent: QWidget, card: ArtworkCard) -> None:
    """Open an artwork detail dialog."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(f"作品详情 #{card.artwork_id}")
    dialog.resize(720, 560)
    layout = QVBoxLayout(dialog)
    preview = ImagePreviewLabel("无预览")
    preview.setMinimumSize(520, 360)
    preview.set_image_path(Path(card.image_path))
    title = QLabel(card.title)
    title.setObjectName("pageTitle")
    metadata = QLabel(
        f"ID：{card.artwork_id}\n风格：{format_style_label(card.style_label)}\n"
        f"作者：{card.artist_name}\n路径：{card.image_path}",
    )
    metadata.setWordWrap(True)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(preview)
    layout.addWidget(title)
    layout.addWidget(metadata)
    layout.addWidget(buttons)
    dialog.exec()


def button_row(*buttons: QPushButton) -> QWidget:
    """Create a left-aligned horizontal row of buttons."""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    for button in buttons:
        layout.addWidget(button)
    layout.addStretch(1)
    return container
