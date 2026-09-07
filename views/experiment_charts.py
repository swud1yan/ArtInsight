"""Matplotlib charts for the algorithm experiment center."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from services.experiments.experiment_models import (
    ConfusionMatrixData,
    DatasetExperimentInfo,
    ExperimentMetricRow,
)
from services.style_translations import format_style_label
from visualization.matplotlib_style import configure_matplotlib_fonts
from views.ui_helpers import create_button
from views.zoom_dialogs import show_enlarged_figure

configure_matplotlib_fonts()


class ExperimentChart(QWidget):
    """Reusable chart widget for experiment result tabs."""

    def __init__(self) -> None:
        """Create an embedded Matplotlib chart."""
        super().__init__()
        self.figure = Figure(figsize=(6.8, 4.2), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._dialog_title = "实验图表放大查看"
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

    def draw_empty(self, title: str, message: str = "暂无实验数据") -> None:
        """Draw an empty-state chart."""
        self._dialog_title = title
        self.toolbar.hide()
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_title(title)
        axis.text(0.5, 0.5, message, ha="center", va="center")
        axis.set_axis_off()
        self.canvas.draw_idle()

    def draw_metric_bars(
        self,
        rows: Sequence[ExperimentMetricRow],
        title: str,
        metrics: Sequence[tuple[str, str]],
    ) -> None:
        """Draw grouped bars for one or more metrics."""
        self._dialog_title = title
        valid_rows = [row for row in rows if any(getattr(row, attr) is not None for _, attr in metrics)]
        if not valid_rows:
            self.draw_empty(title)
            return
        self.toolbar.show()
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        x_values = np.arange(len(valid_rows))
        width = min(0.18, 0.78 / max(1, len(metrics)))
        for metric_index, (label, attribute) in enumerate(metrics):
            offset = (metric_index - (len(metrics) - 1) / 2) * width
            values = [
                float(getattr(row, attribute) or 0.0)
                for row in valid_rows
            ]
            axis.bar(x_values + offset, values, width=width, label=label)
        axis.set_title(title)
        axis.set_xticks(x_values)
        axis.set_xticklabels([row.method for row in valid_rows], rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.22)
        axis.legend(loc="best")
        self.canvas.draw_idle()

    def draw_pca_sensitivity(self, rows: Sequence[ExperimentMetricRow]) -> None:
        """Draw PCA component count versus accuracy and variance."""
        self._dialog_title = "PCA Sensitivity"
        valid_rows = [
            row
            for row in rows
            if row.pca_components is not None and row.accuracy is not None
        ]
        if not valid_rows:
            self.draw_empty("PCA Sensitivity")
            return
        self.toolbar.show()
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        ordered = sorted(valid_rows, key=lambda row: row.pca_components or 0)
        x_values = [int(row.pca_components or 0) for row in ordered]
        axis.plot(
            x_values,
            [float(row.accuracy or 0.0) for row in ordered],
            marker="o",
            label="Accuracy",
            color="#C0563D",
        )
        if any(row.cumulative_variance is not None for row in ordered):
            twin = axis.twinx()
            twin.plot(
                x_values,
                [float(row.cumulative_variance or 0.0) for row in ordered],
                marker="s",
                label="Cumulative Variance",
                color="#2F6F73",
            )
            twin.set_ylabel("Cumulative Variance")
            twin.set_ylim(0, 1.05)
            handles, labels = axis.get_legend_handles_labels()
            handles_b, labels_b = twin.get_legend_handles_labels()
            axis.legend(handles + handles_b, labels + labels_b, loc="best")
        else:
            axis.legend(loc="best")
        axis.set_title("Accuracy and Variance vs PCA Dimension")
        axis.set_xlabel("Principal Components")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.22)
        self.canvas.draw_idle()

    def draw_confusion_matrix(self, data: ConfusionMatrixData | None) -> None:
        """Draw a confusion matrix heatmap."""
        self._dialog_title = "Confusion Matrix"
        if data is None:
            self.draw_empty("Confusion Matrix")
            return
        self.toolbar.show()
        matrix = np.asarray(data.matrix, dtype=float)
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        image = axis.imshow(matrix, cmap="Blues")
        self.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        display_labels = [format_style_label(label) for label in data.labels]
        axis.set_title(f"Confusion Matrix - {data.method}")
        axis.set_xticks(np.arange(len(data.labels)))
        axis.set_yticks(np.arange(len(data.labels)))
        axis.set_xticklabels(display_labels, rotation=30, ha="right")
        axis.set_yticklabels(display_labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    int(matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        self.canvas.draw_idle()

    def draw_dataset_distribution(self, dataset: DatasetExperimentInfo | None) -> None:
        """Draw class distribution for the paper figure generator tab."""
        self._dialog_title = "Dataset Distribution"
        if dataset is None:
            self.draw_empty("Dataset Distribution")
            return
        labels = [format_style_label(label) for label in dataset.class_distribution]
        values = list(dataset.class_distribution.values())
        if not labels:
            self.draw_empty("Dataset Distribution")
            return
        self.toolbar.show()
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.bar(np.arange(len(labels)), values, color="#2F6F73")
        axis.set_title("Dataset Distribution")
        axis.set_ylabel("Samples")
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.22)
        self.canvas.draw_idle()

    def _show_zoom_dialog(self) -> None:
        """Open the current experiment chart in a larger dialog."""
        show_enlarged_figure(self, self.figure, self._dialog_title)
