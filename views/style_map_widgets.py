"""Small widgets used by the PCA style map page."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
)

from services.style_translations import format_style_label
from views.ui_helpers import create_button

ALL_STYLES_LABEL = "全部风格"


class StyleMapSidebar(QFrame):
    """Sidebar for style filtering, artwork search, and legend text."""

    style_changed = Signal(str)
    search_requested = Signal(str)

    def __init__(self) -> None:
        """Create map sidebar controls."""
        super().__init__()
        self.setObjectName("panel")
        self.setFixedWidth(300)
        self._style_filter = QComboBox()
        self._search_edit = QLineEdit()
        self._legend_label = QLabel("加载地图后显示风格图例。")
        self._setup_ui()

    def set_styles(self, labels: list[str]) -> None:
        """Replace available style labels while preserving current selection."""
        current = self.selected_style()
        self._style_filter.blockSignals(True)
        self._style_filter.clear()
        self._style_filter.addItem(ALL_STYLES_LABEL, None)
        for label in sorted(labels):
            self._style_filter.addItem(format_style_label(label), label)
        index = self._style_filter.findData(current)
        self._style_filter.setCurrentIndex(index if index >= 0 else 0)
        self._style_filter.blockSignals(False)

    def selected_style(self) -> str | None:
        """Return selected style label, or None when all styles are visible."""
        current = self._style_filter.currentData()
        return str(current) if current else None

    def set_legend_counts(self, style_counts: dict[str, int]) -> None:
        """Show textual style counts beside the chart legend."""
        if not style_counts:
            self._legend_label.setText("加载地图后显示风格图例。")
            return
        lines = ["风格图例与数量："]
        lines.extend(
            f"{format_style_label(label)}: {count} 幅"
            for label, count in sorted(style_counts.items())
        )
        self._legend_label.setText("\n".join(lines))

    def _setup_ui(self) -> None:
        """Build sidebar child controls."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._style_filter.addItem(ALL_STYLES_LABEL, None)
        self._style_filter.currentTextChanged.connect(self.style_changed.emit)
        self._search_edit.setPlaceholderText("输入作品ID、标题、作者或风格")
        self._search_edit.returnPressed.connect(self._emit_search)
        search_button = create_button("搜索", "search")
        search_button.clicked.connect(self._emit_search)

        help_label = QLabel("可用鼠标滚轮缩放，工具栏支持平移和保存图像。")
        help_label.setWordWrap(True)
        help_label.setObjectName("hintLabel")
        self._legend_label.setWordWrap(True)
        self._legend_label.setObjectName("hintLabel")
        self._legend_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )

        layout.addWidget(QLabel("风格筛选"))
        layout.addWidget(self._style_filter)
        layout.addWidget(QLabel("作品搜索"))
        layout.addWidget(self._search_edit)
        layout.addWidget(search_button)
        layout.addSpacing(8)
        layout.addWidget(help_label)
        layout.addWidget(self._legend_label, stretch=1)

    def _emit_search(self) -> None:
        """Emit stripped search text."""
        self.search_requested.emit(self._search_edit.text().strip())
