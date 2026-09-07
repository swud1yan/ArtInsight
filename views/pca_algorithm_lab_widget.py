"""Step-by-step PCA algorithm laboratory widget."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from controllers.gui_dtos import PCALabStepData
from views.table_models import ReadOnlyTableModel, TableColumn
from views.ui_helpers import configure_table, create_button, create_panel
from views.widgets import button_row
from views.zoom_dialogs import show_enlarged_table


class PCAAlgorithmLabWidget(QWidget):
    """Teach PCA as an ordered sequence of compact matrix previews."""

    def __init__(self) -> None:
        """Create an empty PCA algorithm laboratory."""
        super().__init__()
        self._steps: list[PCALabStepData] = []
        self._step_index = 0
        self._title_label = QLabel("算法实验室")
        self._title_label.setObjectName("cardTitle")
        self._description_label = QLabel("加载PCA结果后，可逐步查看计算过程。")
        self._description_label.setWordWrap(True)
        self._description_label.setObjectName("hintLabel")
        self._note_label = QLabel("")
        self._note_label.setWordWrap(True)
        self._note_label.setObjectName("hintLabel")
        self._previous_button = create_button("上一步")
        self._next_button = create_button("下一步")
        self._zoom_table_button = create_button("放大表格", "search")
        self._table_view = QTableView()
        self._table_model = ReadOnlyTableModel[list[str]]([])
        self._setup_ui()
        self.clear()

    def set_steps(self, steps: list[PCALabStepData]) -> None:
        """Load algorithm lab steps and display the first one."""
        self._steps = list(steps)
        self._step_index = 0
        if not self._steps:
            self.clear()
            return
        self._render_step()

    def clear(self) -> None:
        """Reset the laboratory to its empty state."""
        self._steps = []
        self._step_index = 0
        self._title_label.setText("算法实验室")
        self._description_label.setText("加载PCA结果后，可逐步查看计算过程。")
        self._note_label.setText("")
        self._replace_table([], [])
        self._update_buttons()

    def _setup_ui(self) -> None:
        """Build laboratory labels, table, and navigation controls."""
        panel = create_panel()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(panel)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        configure_table(self._table_view)
        self._table_view.setMinimumHeight(180)
        self._previous_button.clicked.connect(self._show_previous)
        self._next_button.clicked.connect(self._show_next)
        self._zoom_table_button.clicked.connect(
            lambda: show_enlarged_table(self, self._table_view, "PCA算法步骤表"),
        )
        layout.addWidget(self._title_label)
        layout.addWidget(self._description_label)
        layout.addWidget(button_row(self._zoom_table_button))
        layout.addWidget(self._table_view)
        layout.addWidget(self._note_label)
        layout.addWidget(button_row(self._previous_button, self._next_button))

    def _show_previous(self) -> None:
        """Show the previous PCA algorithm step."""
        if self._step_index > 0:
            self._step_index -= 1
            self._render_step()

    def _show_next(self) -> None:
        """Show the next PCA algorithm step."""
        if self._step_index < len(self._steps) - 1:
            self._step_index += 1
            self._render_step()

    def _render_step(self) -> None:
        """Render the currently selected PCA algorithm step."""
        step = self._steps[self._step_index]
        self._title_label.setText(
            f"{step.title}（{self._step_index + 1}/{len(self._steps)}）",
        )
        self._description_label.setText(step.description)
        self._note_label.setText(step.note)
        self._replace_table(step.column_headers, step.rows)
        self._update_buttons()

    def _replace_table(self, headers: list[str], rows: list[list[str]]) -> None:
        """Replace the table model with headers matching the current step."""
        columns = [
            TableColumn(
                header,
                lambda row, column_index=index: row[column_index],
                Qt.AlignmentFlag.AlignRight if index > 0 else Qt.AlignmentFlag.AlignLeft,
            )
            for index, header in enumerate(headers)
        ]
        self._table_model = ReadOnlyTableModel[list[str]](columns)
        self._table_model.set_rows(rows)
        self._table_view.setModel(self._table_model)

    def _update_buttons(self) -> None:
        """Enable navigation buttons for the current step index."""
        has_steps = bool(self._steps)
        self._zoom_table_button.setEnabled(has_steps)
        self._previous_button.setEnabled(has_steps and self._step_index > 0)
        self._next_button.setEnabled(
            has_steps and self._step_index < len(self._steps) - 1,
        )
