"""Reusable enlarged-view dialogs for dense charts and tables."""

from __future__ import annotations

import copy

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHeaderView,
    QScrollArea,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from views.ui_helpers import configure_table


class EnlargedFigureDialog(QDialog):
    """Show a Matplotlib figure in a large independent dialog."""

    def __init__(self, parent: QWidget, source_figure: Figure, title: str) -> None:
        """Create a large copy of the provided Matplotlib figure."""
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1180, 760)
        self.setMinimumSize(900, 620)
        self._figure = self._clone_figure(source_figure)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        width, height = self._figure.get_size_inches()
        dpi = self._figure.get_dpi()
        self._canvas.setMinimumSize(
            max(1100, int(width * dpi)),
            max(680, int(height * dpi)),
        )
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setWidget(self._canvas)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self._toolbar)
        layout.addWidget(scroll_area, stretch=1)

    def _clone_figure(self, source_figure: Figure) -> Figure:
        """Clone a Matplotlib figure, falling back to a rendered snapshot."""
        try:
            figure = copy.deepcopy(source_figure)
            figure.set_size_inches(12, 7)
            return figure
        except Exception:
            figure = Figure(figsize=(12, 7), tight_layout=True)
            self._copy_figure_snapshot(source_figure, figure)
            return figure

    def _copy_figure_snapshot(self, source_figure: Figure, target_figure: Figure) -> None:
        """Copy rendered figure content into the enlarged figure."""
        source_figure.canvas.draw()
        image = np.asarray(source_figure.canvas.buffer_rgba()).copy()
        axis = target_figure.add_subplot(111)
        axis.imshow(image)
        axis.set_axis_off()


class EnlargedTableDialog(QDialog):
    """Show a table model in a large, scrollable dialog."""

    def __init__(self, parent: QWidget, source_table: QTableView, title: str) -> None:
        """Create a large table dialog from an existing table view."""
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1180, 760)
        self.setMinimumSize(900, 620)

        table = QTableView(self)
        configure_table(table)
        table.setModel(source_table.model())
        table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        table.setSortingEnabled(source_table.isSortingEnabled())
        table.setWordWrap(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(True)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(table)


def show_enlarged_figure(parent: QWidget, figure: Figure, title: str) -> None:
    """Open an enlarged copy of a Matplotlib figure."""
    dialog = EnlargedFigureDialog(parent, figure, title)
    dialog.exec()


def show_enlarged_table(parent: QWidget, table: QTableView, title: str) -> None:
    """Open an enlarged copy of a table view."""
    dialog = EnlargedTableDialog(parent, table, title)
    dialog.exec()
