"""Chart image rendering helpers for report exports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization.matplotlib_style import configure_matplotlib_fonts

configure_matplotlib_fonts()


class ReportChartRenderer:
    """Render report charts as high-DPI image files."""

    def bar_chart(
        self,
        output_path: Path,
        title: str,
        values_by_label: dict[str, int],
    ) -> Path:
        """Render a bar chart image for insertion into reports."""
        figure, axis = plt.subplots(figsize=(5.6, 3.7), dpi=180)
        axis.set_title(title)
        if values_by_label:
            labels = list(values_by_label.keys())
            values = list(values_by_label.values())
            positions = np.arange(len(labels))
            axis.bar(positions, values, color="#2F6F73")
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=25, ha="right")
            axis.grid(axis="y", alpha=0.24)
        else:
            axis.text(0.5, 0.5, "暂无数据", ha="center", va="center")
            axis.set_axis_off()
        return self._save_figure(figure, output_path)

    def variance_chart(self, output_path: Path, ratios: list[float]) -> Path:
        """Render explained variance ratio chart image."""
        figure, axis = plt.subplots(figsize=(5.6, 3.7), dpi=180)
        axis.set_title("主成分解释方差比")
        values = ratios[:12]
        if values:
            x_values = np.arange(1, len(values) + 1)
            axis.bar(x_values, values, color="#2F6F73")
            axis.set_xlabel("主成分")
            axis.set_ylabel("解释方差比")
            axis.set_ylim(0, max(1.0, max(values) * 1.2))
            axis.grid(axis="y", alpha=0.24)
        else:
            axis.text(0.5, 0.5, "暂无PCA模型", ha="center", va="center")
            axis.set_axis_off()
        return self._save_figure(figure, output_path)

    def cumulative_chart(self, output_path: Path, ratios: list[float]) -> Path:
        """Render cumulative explained variance chart image."""
        figure, axis = plt.subplots(figsize=(5.6, 3.7), dpi=180)
        axis.set_title("累计解释率")
        values = ratios[:20]
        if values:
            x_values = np.arange(1, len(values) + 1)
            axis.plot(x_values, values, marker="o", color="#C0563D")
            axis.set_xlabel("主成分数量 K")
            axis.set_ylabel("累计解释率")
            axis.set_ylim(0, 1.05)
            axis.grid(alpha=0.24)
        else:
            axis.text(0.5, 0.5, "暂无PCA模型", ha="center", va="center")
            axis.set_axis_off()
        return self._save_figure(figure, output_path)

    def _save_figure(self, figure: object, output_path: Path) -> Path:
        """Save a Matplotlib figure and close it."""
        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)
        return output_path
