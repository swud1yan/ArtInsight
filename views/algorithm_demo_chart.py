"""Matplotlib chart for the standalone PCA algorithm demonstration."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from controllers.gui_dtos import PCAAlgorithmDemoStep
from visualization.matplotlib_style import configure_matplotlib_fonts
from views.ui_helpers import create_button
from views.zoom_dialogs import show_enlarged_figure

configure_matplotlib_fonts()


class PCAAlgorithmDemoChart(QWidget):
    """Render two-dimensional PCA teaching steps."""

    def __init__(self) -> None:
        """Create an empty PCA demo chart."""
        super().__init__()
        self.figure = Figure(figsize=(5.4, 4.0), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self._dialog_title = "PCA算法演示"
        self._zoom_button = create_button("放大查看", "search")
        self._zoom_button.clicked.connect(self._show_zoom_dialog)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.addStretch(1)
        toolbar_row.addWidget(self._zoom_button)
        layout.addLayout(toolbar_row)
        layout.addWidget(self.canvas)

    def draw_step(self, step: PCAAlgorithmDemoStep) -> None:
        """Draw points, optional mean marker, and optional principal vector."""
        self._dialog_title = step.title
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        points = np.asarray(step.points, dtype=float)
        axis.scatter(points[:, 0], points[:, 1], s=64, color="#2F6F73", alpha=0.9)
        for index, (x_value, y_value) in enumerate(points, start=1):
            axis.annotate(f"S{index}", (x_value, y_value), xytext=(5, 5), textcoords="offset points")
        if step.highlight_mean is not None:
            axis.scatter(
                [step.highlight_mean[0]],
                [step.highlight_mean[1]],
                s=110,
                marker="x",
                color="#C0563D",
                linewidths=2.4,
                label="均值/原点",
            )
        if step.vector_start is not None and step.vector_end is not None:
            start_x, start_y = step.vector_start
            end_x, end_y = step.vector_end
            axis.arrow(
                start_x,
                start_y,
                end_x - start_x,
                end_y - start_y,
                head_width=0.18,
                length_includes_head=True,
                color="#C0563D",
                linewidth=2.2,
                label="主成分方向",
            )
        axis.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.45)
        axis.axvline(0.0, color="#777777", linewidth=0.8, alpha=0.45)
        axis.set_title(step.title)
        axis.set_xlabel("Feature 1 / PC axis")
        axis.set_ylabel("Feature 2")
        axis.grid(alpha=0.24)
        axis.set_aspect("equal", adjustable="datalim")
        if step.vector_start is not None or step.highlight_mean is not None:
            axis.legend(loc="best")
        self.canvas.draw_idle()

    def _show_zoom_dialog(self) -> None:
        """Open the current algorithm demo chart in a larger dialog."""
        show_enlarged_figure(self, self.figure, self._dialog_title)
