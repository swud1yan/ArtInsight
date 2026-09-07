"""Tests for PCA visualization helpers."""

from __future__ import annotations

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

from core.pca import PCAModel, PCAConfig
from core.pca.pca_plots import (
    plot_cumulative_variance,
    plot_pca_2d,
    plot_pca_3d,
    plot_scree,
)
from tests.temp_utils import ProjectTempDirectory


class PCAPlotsTestCase(unittest.TestCase):
    """Validate that PCA plots can be generated and saved headlessly."""

    def test_plot_functions_save_files(self) -> None:
        """Scree, cumulative, 2-D, and 3-D plots should save image files."""
        rng = np.random.default_rng(17)
        data = rng.normal(size=(50, 4))
        labels = np.array(["A"] * 25 + ["B"] * 25)
        result = PCAModel(PCAConfig(n_components=3)).fit(data)

        with ProjectTempDirectory("pca_plots") as directory:
            target_dir = Path(directory)
            figures = [
                plot_scree(result.eigenvalues, target_dir / "scree.png"),
                plot_cumulative_variance(
                    result.cumulative_variance_ratio,
                    target_dir / "cumulative.png",
                ),
                plot_pca_2d(result.transformed_data, labels, target_dir / "pca2d.png"),
                plot_pca_3d(result.transformed_data, labels, target_dir / "pca3d.png"),
            ]

            for file_name in ("scree.png", "cumulative.png", "pca2d.png", "pca3d.png"):
                self.assertTrue((target_dir / file_name).exists())
            for figure in figures:
                plt.close(figure)

    def test_invalid_2d_projection_data_raises_error(self) -> None:
        """A 2-D PCA plot requires at least two components."""
        with self.assertRaises(ValueError):
            plot_pca_2d(np.ones((5, 1)))


if __name__ == "__main__":
    unittest.main()
