"""Reusable image-analysis workflow widgets."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QLabel,
    QFrame,
    QHBoxLayout,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from controllers.gui_dtos import ArtworkCard, StyleScoreView
from services.style_translations import format_style_label
from views.ui_helpers import create_panel, create_scroll_area
from views.widgets import ArtworkGridWidget, ImagePreviewLabel


class ImageAnalysisWorkspace(QFrame):
    """Two-column workspace for image input and analysis results."""

    def __init__(
        self,
        *,
        result_title: str,
        empty_message: str,
        columns: int,
    ) -> None:
        """Create a reusable image-analysis workspace."""
        super().__init__()
        self.setObjectName("panel")
        self.preview = ImagePreviewLabel("请选择图片")
        self.preview.setMinimumSize(360, 360)
        self.status_label = QLabel("请选择图片后开始分析。")
        self.status_label.setObjectName("hintLabel")
        self.status_label.setWordWrap(True)
        self.grid = ArtworkGridWidget(columns=columns, empty_message=empty_message)
        self.grid_container = create_scroll_area(self.grid)
        self._result_layout: QVBoxLayout
        self._setup_ui(result_title)

    def set_image_path(self, image_path: Path | str | None) -> None:
        """Display a selected image or reset the preview."""
        self.preview.set_image_path(image_path)

    def set_status(self, message: str) -> None:
        """Update the current result status text."""
        self.status_label.setText(message)

    def set_cards(self, cards: list[tuple[ArtworkCard, str]]) -> None:
        """Render artwork result cards."""
        self.grid.set_cards(cards)

    def clear_results(self) -> None:
        """Clear artwork results while keeping the configured empty message."""
        self.grid.set_cards([])

    def add_result_widget(self, widget: QWidget) -> None:
        """Insert a page-specific control above the artwork result grid."""
        self._result_layout.insertWidget(2, widget)

    def _setup_ui(self, result_title: str) -> None:
        """Build input and result columns."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(18)
        layout.addWidget(self._create_input_panel(), stretch=2)
        layout.addWidget(self._create_result_panel(result_title), stretch=3)

    def _create_input_panel(self) -> QFrame:
        """Create the input-image panel."""
        panel = create_panel()
        panel.setMinimumWidth(420)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        title = QLabel("输入图像")
        title.setObjectName("cardTitle")
        hint = QLabel("选择一张艺术作品图片后，系统会在右侧显示分析结果。")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(self.preview, stretch=1)
        layout.addWidget(hint)
        return panel

    def _create_result_panel(self, title_text: str) -> QFrame:
        """Create the analysis-result panel."""
        panel = create_panel()
        panel.setMinimumWidth(520)
        self._result_layout = QVBoxLayout(panel)
        self._result_layout.setContentsMargins(16, 16, 16, 16)
        self._result_layout.setSpacing(10)
        title = QLabel(title_text)
        title.setObjectName("cardTitle")
        self._result_layout.addWidget(title)
        self._result_layout.addWidget(self.status_label)
        self._result_layout.addWidget(self.grid_container, stretch=1)
        return panel


class StyleSimilarityChart(QFrame):
    """Compact bar chart for per-style similarity shares."""

    def __init__(self, max_rows: int = 8) -> None:
        """Create an initially empty similarity chart."""
        super().__init__()
        self._max_rows = max(1, max_rows)
        self.setObjectName("panel")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setSpacing(8)
        title = QLabel("风格相似度占比")
        title.setObjectName("cardTitle")
        self._note = QLabel("完成分类后显示各风格的相似度分布。")
        self._note.setObjectName("hintLabel")
        self._note.setWordWrap(True)
        self._layout.addWidget(title)
        self._layout.addWidget(self._note)
        self.set_scores([])

    def set_scores(self, scores: list[StyleScoreView]) -> None:
        """Render ranked style similarity scores."""
        self._clear_rows()
        if not scores:
            empty = QLabel("暂无相似度数据。")
            empty.setObjectName("hintLabel")
            self._layout.addWidget(empty)
            return

        top_scores = sorted(scores, key=lambda item: item.score, reverse=True)[
            : self._max_rows
        ]
        for index, score in enumerate(top_scores):
            self._layout.addWidget(self._create_row(index, score))

    def _create_row(self, index: int, score: StyleScoreView) -> QWidget:
        """Create one labeled progress row."""
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        label = QLabel(f"{index + 1}. {format_style_label(score.style_label)}")
        label.setObjectName("cardSubtitle")
        value = QLabel(f"{score.score * 100:.1f}%")
        value.setObjectName("cardTitle")
        label_row.addWidget(label, stretch=1)
        label_row.addWidget(value)
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setValue(int(round(score.score * 1000)))
        bar.setTextVisible(False)
        layout.addLayout(label_row)
        layout.addWidget(bar)
        return row

    def _clear_rows(self) -> None:
        """Remove rendered score rows while keeping title and note."""
        while self._layout.count() > 2:
            item = self._layout.takeAt(2)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
