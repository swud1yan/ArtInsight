"""Stateful PCA model built from the step-by-step PCA routines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from . import pca_steps as steps
from .pca_steps import FloatMatrix, FloatVector


class PreprocessingMode(str, Enum):
    """Supported PCA preprocessing modes."""

    CENTER = "center"
    STANDARDIZE = "standardize"


@dataclass(frozen=True)
class PCAConfig:
    """Configuration for PCA fitting."""

    n_components: int | None = None
    variance_threshold: float | None = None
    preprocessing: PreprocessingMode = PreprocessingMode.CENTER


@dataclass(frozen=True)
class PCAResult:
    """Data transfer object containing a fitted PCA result."""

    mean: FloatVector
    scale: FloatVector
    covariance_matrix: FloatMatrix
    eigenvalues: FloatVector
    eigenvectors: FloatMatrix
    projection_matrix: FloatMatrix
    transformed_data: FloatMatrix
    reconstructed_data: FloatMatrix
    explained_variance_ratio: FloatVector
    cumulative_variance_ratio: FloatVector
    reconstruction_mse: float
    preprocessing: PreprocessingMode


class PCAModel:
    """Fit, transform, and reconstruct data using a manual PCA implementation."""

    def __init__(self, config: PCAConfig | None = None) -> None:
        """Create a PCA model with optional configuration."""
        self._config = config or PCAConfig()
        self._result: PCAResult | None = None

    @property
    def config(self) -> PCAConfig:
        """Return the immutable model configuration."""
        return self._config

    @property
    def is_fitted(self) -> bool:
        """Return whether ``fit`` has been called successfully."""
        return self._result is not None

    @property
    def result(self) -> PCAResult:
        """Return the fitted PCA result.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if self._result is None:
            msg = "PCA model has not been fitted."
            raise ValueError(msg)
        return self._result

    def fit(self, data: np.ndarray) -> PCAResult:
        """Fit PCA to a data matrix.

        Args:
            data: Input matrix with shape ``(n_samples, n_features)``.

        Returns:
            A ``PCAResult`` containing eigenpairs, projection matrix,
            transformed data, reconstruction, and explained variance values.
        """
        self._validate_config()
        matrix = steps.validate_data_matrix(data, min_samples=2)
        prepared, mean, scale = self._prepare_fit_matrix(matrix)

        covariance_matrix = steps.compute_covariance(prepared)
        eigenvalues, eigenvectors = steps.compute_eigen(covariance_matrix)
        variance_ratio, cumulative_ratio = steps.explained_variance(eigenvalues)
        projection_matrix = self._select_projection(eigenvalues, eigenvectors)
        transformed_data = steps.transform(prepared, projection_matrix)

        zero_mean = np.zeros(matrix.shape[1], dtype=np.float64)
        reconstructed_prepared = steps.inverse_transform(
            transformed_data,
            projection_matrix,
            zero_mean,
        )
        reconstructed_data = reconstructed_prepared * scale + mean
        mse = steps.reconstruction_error(matrix, reconstructed_data)

        self._result = PCAResult(
            mean=mean,
            scale=scale,
            covariance_matrix=covariance_matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            projection_matrix=projection_matrix,
            transformed_data=transformed_data,
            reconstructed_data=reconstructed_data,
            explained_variance_ratio=variance_ratio,
            cumulative_variance_ratio=cumulative_ratio,
            reconstruction_mse=mse,
            preprocessing=self._config.preprocessing,
        )
        return self._result

    def fit_transform(self, data: np.ndarray) -> FloatMatrix:
        """Fit PCA and return the transformed coordinates.

        Args:
            data: Input matrix with shape ``(n_samples, n_features)``.

        Returns:
            PCA coordinates with shape ``(n_samples, n_components)``.
        """
        return self.fit(data).transformed_data

    def transform(self, data: np.ndarray) -> FloatMatrix:
        """Project new samples using the fitted projection matrix.

        Args:
            data: Matrix with shape ``(n_samples, n_features)``.

        Returns:
            PCA coordinates with shape ``(n_samples, n_components)``.
        """
        result = self.result
        matrix = steps.validate_data_matrix(data, min_samples=1)
        if matrix.shape[1] != result.mean.shape[0]:
            msg = "data feature count must match the fitted PCA model."
            raise ValueError(msg)
        prepared = (matrix - result.mean) / result.scale
        return steps.transform(prepared, result.projection_matrix)

    def inverse_transform(self, transformed: np.ndarray) -> FloatMatrix:
        """Reconstruct original-space samples from PCA coordinates.

        Args:
            transformed: PCA coordinates with shape
                ``(n_samples, n_components)``.

        Returns:
            Approximate original-space matrix with shape
            ``(n_samples, n_features)``.
        """
        result = self.result
        zero_mean = np.zeros(result.mean.shape[0], dtype=np.float64)
        reconstructed = steps.inverse_transform(
            transformed,
            result.projection_matrix,
            zero_mean,
        )
        return reconstructed * result.scale + result.mean

    def compact_for_prediction(self) -> None:
        """Drop large fit-time matrices while keeping prediction usable.

        Classification and retrieval only need the PCA mean, scale, projection
        matrix, and variance summaries. Full covariance/eigenvector and
        reconstruction matrices can make persisted GUI models very large.
        """
        result = self.result
        component_count = int(result.projection_matrix.shape[1])
        feature_count = int(result.mean.shape[0])
        self._result = PCAResult(
            mean=result.mean.astype(np.float32, copy=True),
            scale=result.scale.astype(np.float32, copy=True),
            covariance_matrix=np.empty((0, 0), dtype=np.float32),
            eigenvalues=result.eigenvalues.astype(np.float32, copy=True),
            eigenvectors=np.empty((0, 0), dtype=np.float32),
            projection_matrix=result.projection_matrix.astype(np.float32, copy=True),
            transformed_data=np.empty((0, component_count), dtype=np.float32),
            reconstructed_data=np.empty((0, feature_count), dtype=np.float32),
            explained_variance_ratio=result.explained_variance_ratio.astype(
                np.float32,
                copy=True,
            ),
            cumulative_variance_ratio=result.cumulative_variance_ratio.astype(
                np.float32,
                copy=True,
            ),
            reconstruction_mse=float(result.reconstruction_mse),
            preprocessing=result.preprocessing,
        )

    def _prepare_fit_matrix(
        self,
        matrix: FloatMatrix,
    ) -> tuple[FloatMatrix, FloatVector, FloatVector]:
        """Apply configured preprocessing for fitting."""
        if self._config.preprocessing == PreprocessingMode.STANDARDIZE:
            return steps.standardize(matrix)
        mean = steps.compute_mean(matrix)
        scale = np.ones(matrix.shape[1], dtype=np.float64)
        return matrix - mean, mean, scale

    def _select_projection(
        self,
        eigenvalues: FloatVector,
        eigenvectors: FloatMatrix,
    ) -> FloatMatrix:
        """Select projection matrix by component count or variance threshold."""
        if self._config.variance_threshold is not None:
            projection_matrix, _ = steps.select_components_by_variance(
                eigenvalues,
                eigenvectors,
                self._config.variance_threshold,
            )
            return projection_matrix

        component_count = self._config.n_components or eigenvalues.shape[0]
        return steps.select_components(eigenvalues, eigenvectors, component_count)

    def _validate_config(self) -> None:
        """Validate static PCA configuration values."""
        if self._config.n_components is not None and self._config.n_components < 1:
            msg = "n_components must be positive when provided."
            raise ValueError(msg)
        if self._config.variance_threshold is not None:
            if not 0.0 < self._config.variance_threshold <= 1.0:
                msg = "variance_threshold must be in the interval (0, 1]."
                raise ValueError(msg)
        if (
            self._config.n_components is not None
            and self._config.variance_threshold is not None
        ):
            msg = "Use either n_components or variance_threshold, not both."
            raise ValueError(msg)
