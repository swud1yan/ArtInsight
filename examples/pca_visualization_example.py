"""Example for generating PCA visualization images.

Run with:
    python examples/pca_visualization_example.py
"""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/cache/matplotlib").resolve()))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from core.pca import PCAConfig, PCAModel
from visualization.pca_chart_models import PCAScatterPoint
from visualization.pca_charts import (
    draw_explained_variance_plot,
    draw_pca_scatter_2d,
    draw_pca_scatter_3d,
    draw_scree_plot,
    save_figure,
)


def main() -> None:
    """Generate PCA chart examples from a deterministic synthetic dataset."""
    rng = np.random.default_rng(42)
    first_cluster = rng.normal(loc=0.0, scale=0.8, size=(36, 6))
    second_cluster = rng.normal(loc=2.8, scale=0.7, size=(36, 6))
    data = np.vstack([first_cluster, second_cluster])
    labels = ["Impressionism"] * 36 + ["Cubism"] * 36

    pca_result = PCAModel(PCAConfig(n_components=3)).fit(data)
    points = [
        PCAScatterPoint(
            sample_id=str(index + 1),
            label=labels[index],
            x_value=float(pca_result.transformed_data[index, 0]),
            y_value=float(pca_result.transformed_data[index, 1]),
            z_value=float(pca_result.transformed_data[index, 2]),
            title=f"Sample {index + 1}",
        )
        for index in range(data.shape[0])
    ]

    output_dir = Path("data/exports/pca_visualization_examples")
    charts = {
        "scree.png": _figure_with(
            draw_scree_plot,
            pca_result.eigenvalues,
        ),
        "explained_variance.png": _figure_with(
            draw_explained_variance_plot,
            pca_result.explained_variance_ratio,
            pca_result.cumulative_variance_ratio,
        ),
        "pca_2d.png": _figure_with(draw_pca_scatter_2d, points),
        "pca_3d.png": _figure_with(draw_pca_scatter_3d, points),
    }
    for file_name, figure in charts.items():
        save_figure(figure, output_dir / file_name)
        plt.close(figure)


def _figure_with(function: Callable[..., object], *args: object) -> Figure:
    """Create a figure, draw a chart into it, and return the figure."""
    figure = Figure(figsize=(8, 5), tight_layout=True)
    function(figure, *args)
    return figure


if __name__ == "__main__":
    main()
