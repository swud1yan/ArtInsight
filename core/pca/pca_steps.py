"""Step-by-step PCA numerical routines.

This module contains the textbook matrix operations used by principal
component analysis. It intentionally depends only on NumPy so the algorithm
layer remains independent from GUI, database, and service code.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]
FloatVector = NDArray[np.float64]


def validate_data_matrix(data: np.ndarray, *, min_samples: int = 2) -> FloatMatrix:
    """Validate and copy a sample-feature matrix as ``np.float64``.

    Args:
        data: Input matrix with shape ``(n_samples, n_features)``.
        min_samples: Minimum accepted number of rows.

    Returns:
        A copied ``np.float64`` matrix.

    Raises:
        TypeError: If ``data`` is not a NumPy array.
        ValueError: If the matrix shape or numeric contents are invalid.
    """
    if not isinstance(data, np.ndarray):
        msg = "data must be a NumPy ndarray."
        raise TypeError(msg)
    if min_samples < 1:
        msg = "min_samples must be at least 1."
        raise ValueError(msg)

    matrix = np.asarray(data, dtype=np.float64)
    if matrix.ndim != 2:
        msg = "data must be a 2-D matrix with shape (n_samples, n_features)."
        raise ValueError(msg)
    if matrix.shape[0] < min_samples:
        msg = f"data must contain at least {min_samples} sample(s)."
        raise ValueError(msg)
    if matrix.shape[1] < 1:
        msg = "data must contain at least one feature column."
        raise ValueError(msg)
    if not np.all(np.isfinite(matrix)):
        msg = "data must contain only finite numeric values."
        raise ValueError(msg)
    return matrix.copy()


def compute_mean(data: np.ndarray) -> FloatVector:
    """Compute the column-wise mean vector.

    Args:
        data: Input matrix with shape ``(n_samples, n_features)``.

    Returns:
        Mean vector with shape ``(n_features,)``.
    """
    matrix = validate_data_matrix(data, min_samples=1)
    return np.mean(matrix, axis=0, dtype=np.float64)


def mean_center(data: np.ndarray) -> FloatMatrix:
    """Center each feature column by subtracting its sample mean.

    Args:
        data: Input matrix with shape ``(n_samples, n_features)``.

    Returns:
        Mean-centered matrix with the same shape as ``data``.
    """
    matrix = validate_data_matrix(data, min_samples=1)
    return matrix - np.mean(matrix, axis=0, dtype=np.float64)


def standardize(data: np.ndarray) -> tuple[FloatMatrix, FloatVector, FloatVector]:
    """Standardize features to zero mean and unit sample standard deviation.

    Constant features receive a scale value of 1.0, so their standardized
    values become zeros instead of producing division-by-zero values.

    Args:
        data: Input matrix with shape ``(n_samples, n_features)``.

    Returns:
        A tuple ``(standardized, mean, scale)``. ``standardized`` has the same
        shape as ``data``; ``mean`` and ``scale`` have shape ``(n_features,)``.
    """
    matrix = validate_data_matrix(data, min_samples=2)
    mean = np.mean(matrix, axis=0, dtype=np.float64)
    scale = np.std(matrix, axis=0, ddof=1, dtype=np.float64)
    safe_scale = np.where(np.isclose(scale, 0.0), 1.0, scale)
    return (matrix - mean) / safe_scale, mean, safe_scale


def compute_covariance(data: np.ndarray) -> FloatMatrix:
    """Compute the sample covariance matrix.

    The function centers a working copy before multiplication and uses
    ``n_samples - 1`` as the denominator, matching the unbiased sample
    covariance formula commonly used in PCA.

    Args:
        data: Matrix with shape ``(n_samples, n_features)``.

    Returns:
        Covariance matrix with shape ``(n_features, n_features)``.
    """
    matrix = validate_data_matrix(data, min_samples=2)
    centered = matrix - np.mean(matrix, axis=0, dtype=np.float64)
    return (centered.T @ centered) / float(centered.shape[0] - 1)


def compute_eigen(cov: np.ndarray) -> tuple[FloatVector, FloatMatrix]:
    """Solve eigenvalues and eigenvectors for a covariance matrix.

    Covariance matrices are real and symmetric, so ``np.linalg.eigh`` is used
    for numerical stability. Returned eigenpairs are sorted by descending
    eigenvalue.

    Args:
        cov: Square covariance matrix with shape ``(n_features, n_features)``.

    Returns:
        ``(eigenvalues, eigenvectors)`` where eigenvalues have shape
        ``(n_features,)`` and eigenvectors have shape
        ``(n_features, n_features)``. Each eigenvector is stored as a column.
    """
    matrix = _validate_square_matrix(cov)
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        msg = "covariance matrix must be symmetric."
        raise ValueError(msg)

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.real_if_close(eigenvalues).astype(np.float64)
    eigenvectors = np.real_if_close(eigenvectors).astype(np.float64)
    eigenvalues[np.isclose(eigenvalues, 0.0, atol=1e-12)] = 0.0
    return sort_eigenpairs(eigenvalues, eigenvectors)


def sort_eigenpairs(
    eig_vals: np.ndarray,
    eig_vecs: np.ndarray,
) -> tuple[FloatVector, FloatMatrix]:
    """Sort eigenvalues and matching eigenvectors in descending order.

    Args:
        eig_vals: Eigenvalues with shape ``(n_features,)``.
        eig_vecs: Eigenvectors with shape ``(n_features, n_features)`` where
            each column corresponds to the eigenvalue at the same index.

    Returns:
        Sorted ``(eigenvalues, eigenvectors)``.
    """
    values = np.asarray(eig_vals, dtype=np.float64)
    vectors = np.asarray(eig_vecs, dtype=np.float64)
    if values.ndim != 1:
        msg = "eig_vals must be a one-dimensional array."
        raise ValueError(msg)
    if vectors.ndim != 2:
        msg = "eig_vecs must be a two-dimensional matrix."
        raise ValueError(msg)
    if vectors.shape[0] != vectors.shape[1] or vectors.shape[1] != values.shape[0]:
        msg = "eig_vecs shape must match the number of eigenvalues."
        raise ValueError(msg)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        msg = "eigenvalues and eigenvectors must contain finite values."
        raise ValueError(msg)

    order = np.argsort(values)[::-1]
    return values[order].copy(), vectors[:, order].copy()


def select_components(
    eig_vals: np.ndarray,
    eig_vecs: np.ndarray,
    n_components: int,
) -> FloatMatrix:
    """Select the projection matrix for the top principal components.

    Args:
        eig_vals: Eigenvalues with shape ``(n_features,)``.
        eig_vecs: Eigenvectors with shape ``(n_features, n_features)``.
        n_components: Number of leading principal components to select.

    Returns:
        Projection matrix with shape ``(n_features, n_components)``.
    """
    sorted_values, sorted_vectors = sort_eigenpairs(eig_vals, eig_vecs)
    if not 1 <= n_components <= sorted_values.shape[0]:
        msg = "n_components must be between 1 and the number of features."
        raise ValueError(msg)
    return sorted_vectors[:, :n_components].copy()


def select_components_by_variance(
    eig_vals: np.ndarray,
    eig_vecs: np.ndarray,
    variance_threshold: float,
) -> tuple[FloatMatrix, int]:
    """Select components until cumulative explained variance reaches a threshold.

    Args:
        eig_vals: Eigenvalues with shape ``(n_features,)``.
        eig_vecs: Eigenvectors with shape ``(n_features, n_features)``.
        variance_threshold: Cumulative explained variance target in ``(0, 1]``.

    Returns:
        A tuple ``(projection_matrix, n_components)``.
    """
    if not 0.0 < variance_threshold <= 1.0:
        msg = "variance_threshold must be in the interval (0, 1]."
        raise ValueError(msg)
    sorted_values, sorted_vectors = sort_eigenpairs(eig_vals, eig_vecs)
    _, cumulative = explained_variance(sorted_values)
    n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    return sorted_vectors[:, :n_components].copy(), n_components


def transform(data: np.ndarray, proj: np.ndarray) -> FloatMatrix:
    """Project data onto the selected principal component space.

    Args:
        data: Centered or standardized matrix with shape
            ``(n_samples, n_features)``.
        proj: Projection matrix with shape ``(n_features, n_components)``.

    Returns:
        Transformed matrix with shape ``(n_samples, n_components)``.
    """
    matrix = validate_data_matrix(data, min_samples=1)
    projection = _validate_projection_matrix(proj)
    if matrix.shape[1] != projection.shape[0]:
        msg = "data feature count must match projection row count."
        raise ValueError(msg)
    return matrix @ projection


def inverse_transform(
    transformed: np.ndarray,
    proj: np.ndarray,
    mean: np.ndarray,
) -> FloatMatrix:
    """Reconstruct approximate original-space data from PCA coordinates.

    Args:
        transformed: PCA coordinates with shape ``(n_samples, n_components)``.
        proj: Projection matrix with shape ``(n_features, n_components)``.
        mean: Mean vector with shape ``(n_features,)`` added after projection.

    Returns:
        Reconstructed matrix with shape ``(n_samples, n_features)``.
    """
    reduced = validate_data_matrix(transformed, min_samples=1)
    projection = _validate_projection_matrix(proj)
    mean_vector = np.asarray(mean, dtype=np.float64)
    if mean_vector.ndim != 1:
        msg = "mean must be a one-dimensional vector."
        raise ValueError(msg)
    if reduced.shape[1] != projection.shape[1]:
        msg = "transformed component count must match projection column count."
        raise ValueError(msg)
    if mean_vector.shape[0] != projection.shape[0]:
        msg = "mean length must match projection row count."
        raise ValueError(msg)
    if not np.all(np.isfinite(mean_vector)):
        msg = "mean must contain only finite values."
        raise ValueError(msg)
    return reduced @ projection.T + mean_vector


def explained_variance(eig_vals: np.ndarray) -> tuple[FloatVector, FloatVector]:
    """Compute explained variance ratio and cumulative contribution.

    Args:
        eig_vals: Eigenvalues with shape ``(n_components_or_features,)``.

    Returns:
        ``(ratio, cumulative_ratio)`` with the same length as ``eig_vals``.
    """
    values = np.asarray(eig_vals, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        msg = "eig_vals must be a non-empty one-dimensional array."
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "eig_vals must contain only finite values."
        raise ValueError(msg)
    clipped = np.where(np.isclose(values, 0.0, atol=1e-12), 0.0, values)
    if np.any(clipped < 0.0):
        msg = "eig_vals must not contain negative values."
        raise ValueError(msg)
    total = float(np.sum(clipped))
    if total <= 0.0:
        msg = "sum of eigenvalues must be positive."
        raise ValueError(msg)

    ratio = clipped / total
    cumulative = np.cumsum(ratio, dtype=np.float64)
    cumulative[-1] = 1.0
    return ratio.astype(np.float64), cumulative.astype(np.float64)


def reconstruction_error(orig: np.ndarray, recon: np.ndarray) -> float:
    """Compute mean squared reconstruction error.

    Args:
        orig: Original matrix with shape ``(n_samples, n_features)``.
        recon: Reconstructed matrix with the same shape as ``orig``.

    Returns:
        Mean squared error as a Python ``float``.
    """
    original = validate_data_matrix(orig, min_samples=1)
    reconstructed = validate_data_matrix(recon, min_samples=1)
    if original.shape != reconstructed.shape:
        msg = "orig and recon must have identical shapes."
        raise ValueError(msg)
    return float(np.mean((original - reconstructed) ** 2, dtype=np.float64))


def _validate_square_matrix(matrix: np.ndarray) -> FloatMatrix:
    """Validate a square finite matrix as ``np.float64``."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        msg = "matrix must be square."
        raise ValueError(msg)
    if array.shape[0] == 0:
        msg = "matrix must not be empty."
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "matrix must contain only finite values."
        raise ValueError(msg)
    return array.copy()


def _validate_projection_matrix(proj: np.ndarray) -> FloatMatrix:
    """Validate a projection matrix as ``np.float64``."""
    projection = np.asarray(proj, dtype=np.float64)
    if projection.ndim != 2:
        msg = "projection matrix must be two-dimensional."
        raise ValueError(msg)
    if projection.shape[0] < 1 or projection.shape[1] < 1:
        msg = "projection matrix must have positive dimensions."
        raise ValueError(msg)
    if not np.all(np.isfinite(projection)):
        msg = "projection matrix must contain only finite values."
        raise ValueError(msg)
    return projection.copy()
