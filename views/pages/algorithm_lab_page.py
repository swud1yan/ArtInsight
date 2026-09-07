"""Standalone PCA algorithm laboratory page for defense demonstrations."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QHBoxLayout, QTableView, QVBoxLayout, QWidget

from controllers.gui_dtos import PCAAlgorithmDemoStep
from controllers.page_data_controller import PageDataController
from views.algorithm_demo_chart import PCAAlgorithmDemoChart
from views.pages.base_page import BasePage
from views.table_models import ReadOnlyTableModel, TableColumn
from views.ui_helpers import configure_table, create_button, create_panel
from views.widgets import button_row
from views.zoom_dialogs import show_enlarged_table


class AlgorithmLabPage(BasePage):
    """Interactive teaching page that walks through the PCA algorithm."""

    page_title = "算法实验室"
    page_subtitle = "用二维示例点逐步演示 PCA 的中心化、协方差、特征向量与投影。"

    def __init__(self, controller: PageDataController) -> None:
        """Create the standalone PCA laboratory page."""
        super().__init__(controller)
        self._steps = self._controller.get_pca_algorithm_demo_steps()
        self._step_index = 0
        self._chart = PCAAlgorithmDemoChart()
        self._title_label = QLabel()
        self._title_label.setObjectName("cardTitle")
        self._description_label = QLabel()
        self._description_label.setWordWrap(True)
        self._description_label.setObjectName("hintLabel")
        self._annotation_label = QLabel()
        self._annotation_label.setWordWrap(True)
        self._annotation_label.setObjectName("hintLabel")
        self._previous_button = create_button("上一步")
        self._next_button = create_button("下一步", primary=True)
        self._zoom_table_button = create_button("放大表格", "search")
        self._table_view = QTableView()
        self._table_model = ReadOnlyTableModel[list[str]]([])
        self._setup_ui()
        self._render_step()

    def _setup_ui(self) -> None:
        """Build demo chart, explanation panel, table, and navigation."""
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        chart_panel = create_panel()
        chart_layout = QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(16, 16, 16, 16)
        chart_layout.addWidget(self._chart)

        detail_panel = create_panel()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(16, 16, 16, 16)
        detail_layout.setSpacing(12)
        configure_table(self._table_view)
        self._table_view.setMinimumHeight(220)
        self._previous_button.clicked.connect(self._show_previous)
        self._next_button.clicked.connect(self._show_next)
        self._zoom_table_button.clicked.connect(
            lambda: show_enlarged_table(self, self._table_view, "PCA算法步骤表"),
        )
        detail_layout.addWidget(self._title_label)
        detail_layout.addWidget(self._description_label)
        detail_layout.addWidget(button_row(self._zoom_table_button))
        detail_layout.addWidget(self._table_view)
        detail_layout.addWidget(self._annotation_label)
        detail_layout.addWidget(button_row(self._previous_button, self._next_button))

        content_layout.addWidget(chart_panel, stretch=3)
        content_layout.addWidget(detail_panel, stretch=2)
        self._root_layout.addWidget(content, stretch=1)

    def _show_previous(self) -> None:
        """Move to the previous teaching step."""
        if self._step_index > 0:
            self._step_index -= 1
            self._render_step()

    def _show_next(self) -> None:
        """Move to the next teaching step."""
        if self._step_index < len(self._steps) - 1:
            self._step_index += 1
            self._render_step()

    def _render_step(self) -> None:
        """Render the current PCA teaching step."""
        step = self._steps[self._step_index]
        self._chart.draw_step(step)
        self._title_label.setText(
            f"{step.title}（{self._step_index + 1}/{len(self._steps)}）",
        )
        self._description_label.setText(step.explanation)
        self._annotation_label.setText(step.annotation)
        self._replace_table(step)
        self._previous_button.setEnabled(self._step_index > 0)
        self._next_button.setEnabled(self._step_index < len(self._steps) - 1)

    def _replace_table(self, step: PCAAlgorithmDemoStep) -> None:
        """Replace the numeric table for the current teaching step."""
        columns = [
            TableColumn(
                header,
                lambda row, column_index=index: row[column_index],
                Qt.AlignmentFlag.AlignRight if index > 0 else Qt.AlignmentFlag.AlignLeft,
            )
            for index, header in enumerate(step.table_headers)
        ]
        self._table_model = ReadOnlyTableModel[list[str]](columns)
        self._table_model.set_rows(step.table_rows)
        self._table_view.setModel(self._table_model)
