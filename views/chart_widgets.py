"""Matplotlib widgets embedded in PySide6 pages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from visualization.pca_charts import (
    draw_empty_chart,
    draw_explained_variance_plot,
    draw_pca_scatter_2d,
    draw_scree_plot,
)
from visualization.pca_chart_models import PCAScatterPoint
from visualization.matplotlib_style import configure_matplotlib_fonts
from views.ui_helpers import create_button
from views.zoom_dialogs import show_enlarged_figure


configure_matplotlib_fonts()


class MatplotlibChart(QWidget):
    """Reusable Matplotlib canvas wrapper."""

    def __init__(self) -> None:
        """Create an empty chart widget."""
        super().__init__()
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._dialog_title = "图表放大查看"
        self._zoom_redraw: tuple[str, Sequence[PCAScatterPoint], str] | None = None
        self._zoom_button = create_button("放大查看", "search")
        self._zoom_button.clicked.connect(self._show_zoom_dialog)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.addWidget(self.toolbar)
        toolbar_row.addStretch(1)
        toolbar_row.addWidget(self._zoom_button)
        layout.addLayout(toolbar_row)
        layout.addWidget(self.canvas)

    def clear(self) -> Any:
        """Clear the figure and return a fresh axes object."""
        self.figure.clear()
        return self.figure.add_subplot(111)

    def draw_empty(self, title: str, message: str = "暂无数据") -> None:
        """Draw a centered no-data message."""
        self._dialog_title = title
        self._zoom_redraw = None
        self.toolbar.hide()
        draw_empty_chart(self.figure, title, message)
        self.canvas.draw_idle()

    def draw_scree(
        self,
        eigenvalues: Sequence[float],
        max_components: int | None = 20,
    ) -> None:
        """Draw PCA Scree plot with the shared visualization module."""
        self._dialog_title = "Scree Plot"
        self._zoom_redraw = None
        self.toolbar.show()
        draw_scree_plot(
            self.figure,
            eigenvalues,
            max_components=max_components,
        )
        self.canvas.draw_idle()

    def draw_explained_variance(
        self,
        explained_ratio: Sequence[float],
        cumulative_ratio: Sequence[float],
        max_components: int | None = 20,
    ) -> None:
        """Draw explained variance bars plus cumulative contribution line."""
        self._dialog_title = "解释方差与累计贡献率"
        self._zoom_redraw = None
        self.toolbar.show()
        draw_explained_variance_plot(
            self.figure,
            explained_ratio,
            cumulative_ratio,
            max_components=max_components,
        )
        self.canvas.draw_idle()

    def draw_pca_scatter(
        self,
        points: Sequence[PCAScatterPoint],
        title: str,
    ) -> None:
        """Draw a non-interactive PCA scatter plot for analysis views."""
        self._dialog_title = title
        self._zoom_redraw = ("pca_scatter_2d", list(points), title)
        self.toolbar.show()
        draw_pca_scatter_2d(
            self.figure,
            points,
            title=title,
            interactive=False,
        )
        self.canvas.draw_idle()

    def draw_bar(
        self,
        title: str,
        labels: Sequence[str],
        values: Sequence[float],
    ) -> None:
        """Draw a simple bar chart."""
        self._dialog_title = title
        self._zoom_redraw = None
        if len(labels) == 0 or len(values) == 0:
            self.draw_empty(title)
            return
        self.toolbar.show()
        axis = self.clear()
        axis.bar(list(labels), list(values), color="#2F6F73")
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=25)
        axis.grid(axis="y", alpha=0.25)
        self.canvas.draw_idle()

    def draw_line(self, title: str, values: Sequence[float], ylabel: str) -> None:
        """Draw a component-indexed line chart."""
        self._dialog_title = title
        self._zoom_redraw = None
        numeric_values = np.asarray(values, dtype=float)
        if not numeric_values.size:
            self.draw_empty(title)
            return
        self.toolbar.show()
        axis = self.clear()
        x_values = np.arange(1, numeric_values.size + 1)
        axis.plot(x_values, numeric_values, marker="o", color="#C0563D")
        axis.set_title(title)
        axis.set_xlabel("Principal Component")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
        self.canvas.draw_idle()

    def draw_pie(self, title: str, values_by_label: dict[str, int]) -> None:
        """Draw a pie chart for categorical counts."""
        self._dialog_title = title
        self._zoom_redraw = None
        axis = self.clear()
        if not values_by_label or sum(values_by_label.values()) <= 0:
            self.toolbar.hide()
            axis.text(0.5, 0.5, "暂无数据", ha="center", va="center")
        else:
            self.toolbar.show()
            axis.pie(
                list(values_by_label.values()),
                labels=list(values_by_label.keys()),
                autopct="%1.1f%%",
                startangle=90,
            )
        axis.set_title(title)
        self.canvas.draw_idle()

    def _show_zoom_dialog(self) -> None:
        """Open the current chart in a larger dialog."""
        if self._zoom_redraw is None:
            show_enlarged_figure(self, self.figure, self._dialog_title)
            return
        redraw_kind, points, title = self._zoom_redraw
        figure = Figure(figsize=(12, 7), tight_layout=True)
        if redraw_kind == "pca_scatter_2d":
            draw_pca_scatter_2d(
                figure,
                points,
                title=title,
                interactive=False,
                legend_mode="full",
            )
        show_enlarged_figure(self, figure, self._dialog_title)


