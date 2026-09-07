"""Matplotlib visualizations for PCA results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .pca_steps import FloatMatrix, FloatVector


def plot_scree(eigenvalues: np.ndarray, save_path: Path | None = None) -> Figure:
    """Plot a Scree chart from PCA eigenvalues.

    Args:
        eigenvalues: Eigenvalues ordered by descending magnitude.
        save_path: Optional image path. Parent directories are created when
            a path is provided.

    Returns:
        The Matplotlib figure object.
    """
    values = _validate_vector(eigenvalues, "eigenvalues")
    figure, axis = plt.subplots(figsize=(8, 5))
    component_numbers = np.arange(1, values.shape[0] + 1)
    axis.bar(component_numbers, values, color="#4C78A8")
    axis.set_title("PCA Scree Plot")
    axis.set_xlabel("Principal Component")
    axis.set_ylabel("Eigenvalue")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def plot_cumulative_variance(
    cumulative_variance: np.ndarray,
    save_path: Path | None = None,
) -> Figure:
    """Plot cumulative explained variance contribution.

    Args:
        cumulative_variance: Cumulative ratio values in ``[0, 1]``.
        save_path: Optional image path.

    Returns:
        The Matplotlib figure object.
    """
    cumulative = _validate_vector(cumulative_variance, "cumulative_variance")
    figure, axis = plt.subplots(figsize=(8, 5))
    component_numbers = np.arange(1, cumulative.shape[0] + 1)
    axis.plot(component_numbers, cumulative, marker="o", color="#F58518")
    axis.set_title("Cumulative Explained Variance")
    axis.set_xlabel("Number of Components")
    axis.set_ylabel("Cumulative Ratio")
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def plot_pca_2d(
    transformed_data: np.ndarray,
    labels: Sequence[Any] | np.ndarray | None = None,
    save_path: Path | None = None,
) -> Figure:
    """Plot samples in the first two principal components.

    Args:
        transformed_data: PCA coordinates with shape
            ``(n_samples, n_components)`` where ``n_components >= 2``.
        labels: Optional class labels used for color grouping.
        save_path: Optional image path.

    Returns:
        The Matplotlib figure object.
    """
    coordinates = _validate_projection_data(transformed_data, 2)
    figure, axis = plt.subplots(figsize=(7, 6))
    _scatter_by_label(axis, coordinates[:, 0], coordinates[:, 1], labels)
    axis.set_title("PCA 2D Projection")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def plot_pca_3d(
    transformed_data: np.ndarray,
    labels: Sequence[Any] | np.ndarray | None = None,
    save_path: Path | None = None,
) -> Figure:
    """Plot samples in the first three principal components.

    Args:
        transformed_data: PCA coordinates with shape
            ``(n_samples, n_components)`` where ``n_components >= 3``.
        labels: Optional class labels used for color grouping.
        save_path: Optional image path.

    Returns:
        The Matplotlib figure object.
    """
    coordinates = _validate_projection_data(transformed_data, 3)
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    _scatter_by_label(
        axis,
        coordinates[:, 0],
        coordinates[:, 1],
        labels,
        z_values=coordinates[:, 2],
    )
    axis.set_title("PCA 3D Projection")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_zlabel("PC3")
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def _validate_vector(values: np.ndarray, name: str) -> FloatVector:
    """Validate a one-dimensional finite vector."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        msg = f"{name} must be a non-empty one-dimensional array."
        raise ValueError(msg)
    if not np.all(np.isfinite(vector)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    return vector.copy()


def _validate_projection_data(
    data: np.ndarray,
    required_components: int,
) -> FloatMatrix:
    """Validate PCA coordinates for scatter plotting."""
    coordinates = np.asarray(data, dtype=np.float64)
    if coordinates.ndim != 2:
        msg = "transformed_data must be a two-dimensional matrix."
        raise ValueError(msg)
    if coordinates.shape[0] < 1:
        msg = "transformed_data must contain at least one sample."
        raise ValueError(msg)
    if coordinates.shape[1] < required_components:
        msg = (
            "transformed_data must contain at least "
            f"{required_components} components."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(coordinates)):
        msg = "transformed_data must contain only finite values."
        raise ValueError(msg)
    return coordinates.copy()


def _scatter_by_label(
    axis: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    labels: Sequence[Any] | np.ndarray | None,
    *,
    z_values: np.ndarray | None = None,
) -> None:
    """Draw 2-D or 3-D scatter points, grouped by optional labels."""
    if labels is None:
        if z_values is None:
            axis.scatter(x_values, y_values, s=36, alpha=0.85)
        else:
            axis.scatter(x_values, y_values, z_values, s=36, alpha=0.85)
        return

    label_array = np.asarray(labels)
    if label_array.shape[0] != x_values.shape[0]:
        msg = "labels length must match the number of samples."
        raise ValueError(msg)
    unique_labels = np.unique(label_array)
    for label in unique_labels:
        mask = label_array == label
        if z_values is None:
            axis.scatter(
                x_values[mask],
                y_values[mask],
                s=36,
                alpha=0.85,
                label=str(label),
            )
        else:
            axis.scatter(
                x_values[mask],
                y_values[mask],
                z_values[mask],
                s=36,
                alpha=0.85,
                label=str(label),
            )
    axis.legend(title="Label", loc="best")


def _save_figure(figure: Figure, save_path: Path | None) -> None:
    """Save a figure when a target path is provided."""
    if save_path is None:
        return
    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160, bbox_inches="tight")