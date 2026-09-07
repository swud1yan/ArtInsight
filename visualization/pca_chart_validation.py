"""Validation helpers for PCA chart inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from visualization.pca_chart_models import PCAScatterPoint


def validate_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Validate a non-empty finite one-dimensional numeric vector."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        msg = f"{name} must be a non-empty one-dimensional vector."
        raise ValueError(msg)
    if not np.all(np.isfinite(vector)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    return vector.copy()


def validate_ratio_vector(
    values: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Validate explained variance ratios in the interval ``[0, 1]``.

    PCA cumulative ratios can exceed ``1.0`` by tiny floating-point residuals
    after thousands of components. Those numerical round-off cases are clipped
    while genuinely invalid values still raise a clear error.
    """
    vector = validate_vector(values, name)
    tolerance = 1e-8
    if np.any(vector < -tolerance) or np.any(vector > 1.0 + tolerance):
        msg = f"{name} values must be in the interval [0, 1]."
        raise ValueError(msg)
    return np.clip(vector, 0.0, 1.0)


def validate_scatter_points(
    points: Sequence[PCAScatterPoint],
    required_dimension: int,
) -> list[PCAScatterPoint]:
    """Validate PCA scatter points and required coordinate dimensions."""
    validated = list(points)
    for point in validated:
        values = [point.x_value, point.y_value]
        if required_dimension == 3:
            if point.z_value is None:
                msg = "3-D scatter points must include z_value."
                raise ValueError(msg)
            values.append(point.z_value)
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            msg = "PCA scatter coordinates must contain only finite values."
            raise ValueError(msg)
    return validated


def limit_components(values: np.ndarray, max_components: int | None) -> np.ndarray:
    """Limit a vector to the requested number of leading components."""
    if max_components is None:
        return values.copy()
    if max_components <= 0:
        msg = "max_components must be positive when provided."
        raise ValueError(msg)
    return values[:max_components].copy()
