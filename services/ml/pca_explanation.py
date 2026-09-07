"""PCA explanation data prepared for interactive analysis pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core.pca import pca_steps as steps
from services.ml.pca_algorithm_lab import PCAAlgorithmLabBuilder
from services.ml.model_artifact import StyleModelArtifact

if TYPE_CHECKING:
    from services.ml.pca_algorithm_lab import PCALabStep


@dataclass(frozen=True)
class PCAExplanation:
    """Computed PCA interpretation data for GUI and report consumers."""

    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    reconstruction_errors: np.ndarray
    coordinates: np.ndarray
    labels: list[str]
    titles: list[str]
    trained_components: int
    lab_steps: list[PCALabStep]

    @property
    def max_components(self) -> int:
        """Return the number of available principal components."""
        return int(self.eigenvalues.shape[0])


class PCAExplanationBuilder:
    """Build cached PCA interpretation arrays from a trained artifact."""

    def build(
        self,
        artifact: StyleModelArtifact,
        feature_matrix: np.ndarray,
        labels: list[str],
        titles: list[str],
    ) -> PCAExplanation:
        """Prepare projection coordinates and K-wise reconstruction errors.

        Args:
            artifact: Persisted style model artifact containing scaler and PCA.
            feature_matrix: Raw feature matrix with shape
                ``(n_samples, n_features)``.
            labels: Style label for each sample.
            titles: Display title for each sample.

        Returns:
            A ``PCAExplanation`` whose arrays are ready for interactive views.

        Raises:
            ValueError: If sample metadata lengths do not match the matrix.
        """
        matrix = steps.validate_data_matrix(feature_matrix, min_samples=1)
        self._validate_metadata(matrix, labels, titles)

        pca_result = artifact.pca_result
        scaled = artifact.scaler.transform(matrix)
        prepared = (scaled - pca_result.mean) / pca_result.scale
        max_components = self._available_component_count(
            pca_result.eigenvalues,
            pca_result.projection_matrix,
        )
        projection_count = min(max_components, 2)
        projection = pca_result.projection_matrix[:, :projection_count]
        coordinates = steps.transform(prepared, projection)
        reconstruction_errors = reconstruction_error_curve_from_eigenvalues(
            pca_result.eigenvalues[:max_components],
            sample_count=prepared.shape[0],
            feature_count=prepared.shape[1],
        )
        lab_steps = self._build_lab_steps(
            scaled=scaled,
            prepared=prepared,
            coordinates=coordinates,
            eigenvalues=pca_result.eigenvalues,
            eigenvectors=pca_result.eigenvectors,
            covariance_matrix=pca_result.covariance_matrix,
            titles=titles,
        )

        return PCAExplanation(
            eigenvalues=pca_result.eigenvalues[:max_components].copy(),
            explained_variance_ratio=(
                pca_result.explained_variance_ratio[:max_components].copy()
            ),
            cumulative_variance_ratio=(
                pca_result.cumulative_variance_ratio[:max_components].copy()
            ),
            reconstruction_errors=reconstruction_errors,
            coordinates=coordinates,
            labels=list(labels),
            titles=list(titles),
            trained_components=artifact.metadata.n_components,
            lab_steps=lab_steps,
        )

    def _validate_metadata(
        self,
        matrix: np.ndarray,
        labels: list[str],
        titles: list[str],
    ) -> None:
        """Validate per-sample display metadata lengths."""
        expected = matrix.shape[0]
        if len(labels) != expected or len(titles) != expected:
            msg = "labels and titles must match the number of feature rows."
            raise ValueError(msg)

    def _available_component_count(
        self,
        eigenvalues: np.ndarray,
        projection_matrix: np.ndarray,
    ) -> int:
        """Return a positive component count from the fitted PCA result."""
        values = np.asarray(eigenvalues, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            msg = "PCA eigenvalues must be a non-empty one-dimensional array."
            raise ValueError(msg)
        projection = np.asarray(projection_matrix)
        if projection.ndim != 2 or projection.shape[1] < 1:
            msg = "PCA projection matrix must contain at least one component."
            raise ValueError(msg)
        return int(min(values.size, projection.shape[1]))

    def _build_lab_steps(
        self,
        *,
        scaled: np.ndarray,
        prepared: np.ndarray,
        coordinates: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        covariance_matrix: np.ndarray,
        titles: list[str],
    ) -> list[PCALabStep]:
        """Build PCA lab steps from full arrays or compact-model previews."""
        covariance = np.asarray(covariance_matrix)
        vectors = np.asarray(eigenvectors)
        if covariance.size > 0 and vectors.size > 0:
            return PCAAlgorithmLabBuilder().build_steps(
                scaled_matrix=scaled,
                prepared_matrix=prepared,
                covariance_matrix=covariance,
                eigenvalues=eigenvalues,
                eigenvectors=vectors,
                coordinates=coordinates,
                titles=titles,
            )

        preview_rows = min(max(2, scaled.shape[0]), 6)
        preview_cols = min(scaled.shape[1], 6)
        prepared_preview = prepared[:preview_rows, :preview_cols]
        covariance_preview = steps.compute_covariance(prepared_preview)
        preview_values, preview_vectors = steps.compute_eigen(covariance_preview)
        return PCAAlgorithmLabBuilder().build_steps(
            scaled_matrix=scaled[:preview_rows, :preview_cols],
            prepared_matrix=prepared_preview,
            covariance_matrix=covariance_preview,
            eigenvalues=preview_values,
            eigenvectors=preview_vectors,
            coordinates=coordinates[:preview_rows, : min(coordinates.shape[1], 2)],
            titles=titles[:preview_rows],
        )


def reconstruction_error_curve_from_eigenvalues(
    eigenvalues: np.ndarray,
    *,
    sample_count: int,
    feature_count: int,
) -> np.ndarray:
    """Compute reconstruction MSE for K=1..max from discarded eigenvalues.

    The manual PCA implementation uses the unbiased covariance denominator
    ``n_samples - 1``. For an orthonormal PCA basis, the sum of discarded
    eigenvalues is therefore equivalent to the squared reconstruction residual
    divided by ``n_samples - 1``. The returned value is mean squared error over
    all cells of the PCA-prepared feature matrix.
    """
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        msg = "eigenvalues must be a non-empty one-dimensional array."
        raise ValueError(msg)
    if sample_count < 1 or feature_count < 1:
        msg = "sample_count and feature_count must be positive."
        raise ValueError(msg)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        msg = "eigenvalues must be finite and non-negative."
        raise ValueError(msg)

    tail_sums = np.cumsum(values[::-1], dtype=np.float64)[::-1] - values
    denominator = float(sample_count * feature_count)
    errors = tail_sums * float(max(sample_count - 1, 0)) / denominator
    return np.maximum(errors, 0.0).astype(np.float64)
