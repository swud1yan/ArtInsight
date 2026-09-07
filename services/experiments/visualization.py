"""Paper figure generation for algorithm experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from services.experiments.experiment_models import (
    ConfusionMatrixData,
    DatasetExperimentInfo,
    ExperimentMetricRow,
    PaperFigure,
)
from services.style_translations import format_style_label
from visualization.matplotlib_style import configure_matplotlib_fonts

configure_matplotlib_fonts()


class ExperimentFigureExporter:
    """Export SCI-style experiment figures as 300-dpi PNG images."""

    def __init__(self, output_root: Path) -> None:
        """Create an exporter rooted in the experiment export directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_dir = output_root / timestamp
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        *,
        dataset_info: DatasetExperimentInfo,
        rows: list[ExperimentMetricRow],
        baseline_rows: list[ExperimentMetricRow],
        feature_rows: list[ExperimentMetricRow],
        pca_rows: list[ExperimentMetricRow],
        confusion_data: ConfusionMatrixData | None,
        best_payload: Any | None,
    ) -> list[PaperFigure]:
        """Generate the seven standard experiment figures."""
        figures: list[PaperFigure] = []
        figures.append(
            self._dataset_distribution(dataset_info, "Figure1_Dataset_Distribution.png"),
        )
        figures.append(self._scree_plot(best_payload, "Figure2_Scree_Plot.png"))
        figures.append(self._pca_scatter(best_payload, "Figure3_PCA_Scatter.png"))
        figures.append(
            self._confusion_matrix(confusion_data, "Figure4_Confusion_Matrix.png"),
        )
        figures.append(
            self._metric_bar(
                baseline_rows or rows,
                "Accuracy",
                "accuracy",
                "Figure5_Baseline_Comparison.png",
            ),
        )
        figures.append(
            self._metric_bar(
                feature_rows or rows,
                "Feature Ablation",
                "accuracy",
                "Figure6_Feature_Ablation.png",
            ),
        )
        figures.append(
            self._efficiency(rows, "Figure7_Efficiency_Comparison.png"),
        )
        return figures

    def _dataset_distribution(
        self,
        dataset_info: DatasetExperimentInfo,
        file_name: str,
    ) -> PaperFigure:
        """Render class distribution."""
        figure, axis = plt.subplots(figsize=(7.0, 4.2), dpi=300)
        labels = [format_style_label(label) for label in dataset_info.class_distribution]
        values = list(dataset_info.class_distribution.values())
        axis.bar(np.arange(len(labels)), values, color="#2F6F73")
        axis.set_title("Dataset Distribution")
        axis.set_ylabel("Samples")
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.22)
        return self._save(figure, file_name, "Figure1 Dataset Distribution")

    def _scree_plot(self, payload: Any | None, file_name: str) -> PaperFigure:
        """Render a scree plot from the best PCA payload when available."""
        figure, axis = plt.subplots(figsize=(7.0, 4.2), dpi=300)
        pca_result = getattr(payload, "pca_result", None)
        eigenvalues = getattr(pca_result, "eigenvalues", None)
        if eigenvalues is not None and len(eigenvalues) > 0:
            values = np.asarray(eigenvalues[:20], dtype=float)
            axis.bar(np.arange(1, values.size + 1), values, color="#4E8F57")
            axis.set_xlabel("Principal Component")
            axis.set_ylabel("Eigenvalue")
            axis.grid(axis="y", alpha=0.22)
        else:
            axis.text(0.5, 0.5, "No PCA eigenvalues", ha="center", va="center")
            axis.set_axis_off()
        axis.set_title("Scree Plot")
        return self._save(figure, file_name, "Figure2 Scree Plot")

    def _pca_scatter(self, payload: Any | None, file_name: str) -> PaperFigure:
        """Render a 2-D embedding scatter from the best payload."""
        figure, axis = plt.subplots(figsize=(7.0, 4.8), dpi=300)
        embedding = getattr(payload, "test_embedding", None)
        labels = getattr(payload, "test_labels", None)
        if embedding is not None and labels is not None and embedding.shape[1] >= 2:
            unique_labels = sorted(set(labels.tolist()))
            colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_labels))))
            for color, label in zip(colors, unique_labels, strict=False):
                mask = labels == label
                axis.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=18,
                    alpha=0.78,
                    label=format_style_label(str(label)),
                    color=color,
                )
            axis.legend(loc="best", fontsize=8)
            axis.set_xlabel("Component 1")
            axis.set_ylabel("Component 2")
            axis.grid(alpha=0.2)
        else:
            axis.text(0.5, 0.5, "No 2-D embedding", ha="center", va="center")
            axis.set_axis_off()
        axis.set_title("PCA / Embedding Scatter")
        return self._save(figure, file_name, "Figure3 PCA Scatter")

    def _confusion_matrix(
        self,
        data: ConfusionMatrixData | None,
        file_name: str,
    ) -> PaperFigure:
        """Render a confusion matrix heatmap."""
        figure, axis = plt.subplots(figsize=(6.0, 5.3), dpi=300)
        if data is None:
            axis.text(0.5, 0.5, "No confusion matrix", ha="center", va="center")
            axis.set_axis_off()
        else:
            matrix = np.asarray(data.matrix, dtype=float)
            image = axis.imshow(matrix, cmap="Blues")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            display_labels = [format_style_label(label) for label in data.labels]
            axis.set_xticks(np.arange(len(data.labels)))
            axis.set_yticks(np.arange(len(data.labels)))
            axis.set_xticklabels(display_labels, rotation=35, ha="right")
            axis.set_yticklabels(display_labels)
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    axis.text(
                        column,
                        row,
                        int(matrix[row, column]),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
        axis.set_title("Confusion Matrix")
        return self._save(figure, file_name, "Figure4 Confusion Matrix")

    def _metric_bar(
        self,
        rows: list[ExperimentMetricRow],
        title: str,
        attribute: str,
        file_name: str,
    ) -> PaperFigure:
        """Render one metric bar chart."""
        figure, axis = plt.subplots(figsize=(7.0, 4.2), dpi=300)
        valid = [row for row in rows if getattr(row, attribute) is not None]
        if valid:
            labels = [row.method for row in valid]
            values = [float(getattr(row, attribute) or 0.0) for row in valid]
            axis.bar(np.arange(len(labels)), values, color="#C0563D")
            axis.set_xticks(np.arange(len(labels)))
            axis.set_xticklabels(labels, rotation=25, ha="right")
            axis.set_ylim(0, max(1.0, max(values) * 1.15))
            axis.grid(axis="y", alpha=0.22)
        else:
            axis.text(0.5, 0.5, "No metric data", ha="center", va="center")
            axis.set_axis_off()
        axis.set_title(title)
        return self._save(figure, file_name, title)

    def _efficiency(
        self,
        rows: list[ExperimentMetricRow],
        file_name: str,
    ) -> PaperFigure:
        """Render time and memory comparison."""
        figure, axis = plt.subplots(figsize=(7.2, 4.4), dpi=300)
        valid = [
            row
            for row in rows
            if row.total_time_ms is not None and row.memory_mb is not None
        ]
        if valid:
            labels = [row.method for row in valid]
            x_values = np.arange(len(labels))
            axis.bar(
                x_values - 0.18,
                [float(row.total_time_ms or 0.0) for row in valid],
                width=0.36,
                label="Time (ms)",
                color="#2F6F73",
            )
            twin = axis.twinx()
            twin.bar(
                x_values + 0.18,
                [float(row.memory_mb or 0.0) for row in valid],
                width=0.36,
                label="Memory (MB)",
                color="#D08C2F",
                alpha=0.85,
            )
            axis.set_xticks(x_values)
            axis.set_xticklabels(labels, rotation=25, ha="right")
            axis.set_ylabel("Time (ms)")
            twin.set_ylabel("Memory (MB)")
            handles, labels_a = axis.get_legend_handles_labels()
            handles_b, labels_b = twin.get_legend_handles_labels()
            axis.legend(handles + handles_b, labels_a + labels_b, loc="upper right")
            axis.grid(axis="y", alpha=0.2)
        else:
            axis.text(0.5, 0.5, "No efficiency data", ha="center", va="center")
            axis.set_axis_off()
        axis.set_title("Efficiency Comparison")
        return self._save(figure, file_name, "Figure7 Efficiency Comparison")

    def _save(self, figure: object, file_name: str, title: str) -> PaperFigure:
        """Save and close one figure."""
        path = self._output_dir / file_name
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        return PaperFigure(name=Path(file_name).stem, title=title, path=path)
