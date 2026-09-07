"""Tests for PCA-specific visualization charts."""

from __future__ import annotations

from collections.abc import Callable
import os
import unittest
from pathlib import Path

MPL_CONFIG_DIR = Path("data/cache/matplotlib").resolve()
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from tests.temp_utils import ProjectTempDirectory
from visualization.pca_chart_models import PCAScatterPoint
from visualization.pca_charts import (
    draw_explained_variance_plot,
    draw_pca_scatter_2d,
    draw_pca_scatter_3d,
    draw_scree_plot,
    save_figure,
)


class PCAChartsTestCase(unittest.TestCase):
    """Validate PCA chart rendering and validation rules."""

    def test_pca_charts_save_png_files(self) -> None:
        """Scree, variance, 2-D, and 3-D charts should save as PNG files."""
        points = self._build_points()
        figures = [
            self._figure_with(draw_scree_plot, np.array([3.0, 1.5, 0.5])),
            self._figure_with(
                draw_explained_variance_plot,
                np.array([0.6, 0.3, 0.1]),
                np.array([0.6, 0.9, 1.0]),
            ),
            self._figure_with(draw_pca_scatter_2d, points),
            self._figure_with(draw_pca_scatter_3d, points),
        ]

        with ProjectTempDirectory("pca_charts") as directory:
            target_dir = Path(directory)
            for index, figure in enumerate(figures):
                target = save_figure(figure, target_dir / f"chart_{index}.png")
                self.assertTrue(target.exists())
                plt.close(figure)

    def test_invalid_variance_lengths_raise_error(self) -> None:
        """Explained and cumulative variance arrays must align by component."""
        figure = Figure(figsize=(6, 4))
        with self.assertRaises(ValueError):
            draw_explained_variance_plot(figure, [0.5, 0.5], [0.5])

    def test_tiny_cumulative_roundoff_is_clipped(self) -> None:
        """Large PCA runs may produce harmless cumulative-ratio round-off."""
        figure = Figure(figsize=(6, 4))

        draw_explained_variance_plot(
            figure,
            np.array([0.6, 0.4]),
            np.array([0.6, 1.0 + 1e-12]),
        )

        self.assertEqual(len(figure.axes), 2)

    def test_3d_scatter_requires_z_values(self) -> None:
        """3-D scatter points need a third principal component value."""
        figure = Figure(figsize=(6, 4))
        points = [
            PCAScatterPoint(
                sample_id="1",
                label="A",
                x_value=1.0,
                y_value=2.0,
            ),
        ]
        with self.assertRaises(ValueError):
            draw_pca_scatter_3d(figure, points)

    def test_many_style_scatter_uses_compact_legend_by_default(self) -> None:
        """Dense PCA plots should not cover the chart with a long legend."""
        figure = Figure(figsize=(6, 4))
        points = [
            PCAScatterPoint(str(index), f"Style_{index}", float(index), 0.0)
            for index in range(20)
        ]

        draw_pca_scatter_2d(figure, points, interactive=False)

        self.assertEqual(len(figure.axes[0].get_legend_handles_labels()[1]), 20)
        self.assertIsNone(figure.axes[0].get_legend())
        self.assertTrue(figure.axes[0].texts)

    def test_many_style_scatter_full_legend_is_available_for_zoom(self) -> None:
        """Zoomed PCA plots should expose the full style legend."""
        figure = Figure(figsize=(6, 4))
        points = [
            PCAScatterPoint(str(index), f"Style_{index}", float(index), 0.0)
            for index in range(20)
        ]

        draw_pca_scatter_2d(
            figure,
            points,
            interactive=False,
            legend_mode="full",
        )

        legend = figure.axes[0].get_legend()
        self.assertIsNone(legend)
        self.assertEqual(len(figure.axes), 2)
        self.assertIsNotNone(figure.axes[1].get_legend())
        self.assertGreaterEqual(figure.get_size_inches()[1], 9.0)
        self.assertEqual(float(figure.axes[0].get_box_aspect()), 1.0)

    def _build_points(self) -> list[PCAScatterPoint]:
        """Build deterministic PCA scatter points for chart tests."""
        return [
            PCAScatterPoint("1", "A", 0.1, 0.2, 0.3, "A-1"),
            PCAScatterPoint("2", "A", 0.4, 0.5, 0.6, "A-2"),
            PCAScatterPoint("3", "B", 1.1, 1.2, 1.3, "B-1"),
        ]

    def _figure_with(self, function: Callable[..., object], *args: object) -> Figure:
        """Create a figure and draw one chart into it."""
        figure = Figure(figsize=(6, 4), tight_layout=True)
        function(figure, *args)
        return figure


if __name__ == "__main__":
    unittest.main()
