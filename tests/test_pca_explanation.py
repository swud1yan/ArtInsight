"""Tests for PCA explanation helpers used by the GUI analysis page."""

from __future__ import annotations

import unittest

import numpy as np

from core.pca import PCAModel, PCAConfig, pca_steps
from services.ml.pca_algorithm_lab import PCAAlgorithmLabBuilder
from services.ml.pca_explanation import reconstruction_error_curve_from_eigenvalues


class PCAExplanationTestCase(unittest.TestCase):
    """Validate selected-K PCA reconstruction summaries."""

    def test_error_curve_matches_explicit_reconstruction(self) -> None:
        """Eigenvalue-derived MSE should match manual reconstruction errors."""
        data = np.array(
            [
                [1.0, 2.0, 1.2],
                [1.2, 1.8, 1.0],
                [4.0, 4.2, 3.9],
                [4.1, 3.9, 4.0],
                [2.0, 2.2, 1.8],
            ],
            dtype=np.float64,
        )
        model = PCAModel(PCAConfig(n_components=data.shape[1]))
        result = model.fit(data)

        errors = reconstruction_error_curve_from_eigenvalues(
            result.eigenvalues,
            sample_count=data.shape[0],
            feature_count=data.shape[1],
        )

        for component_count in range(1, data.shape[1] + 1):
            projection = pca_steps.select_components(
                result.eigenvalues,
                result.eigenvectors,
                component_count,
            )
            prepared = data - result.mean
            reduced = pca_steps.transform(prepared, projection)
            reconstructed = pca_steps.inverse_transform(
                reduced,
                projection,
                np.zeros(data.shape[1], dtype=np.float64),
            ) + result.mean
            explicit_error = pca_steps.reconstruction_error(data, reconstructed)
            self.assertAlmostEqual(
                float(errors[component_count - 1]),
                explicit_error,
                places=10,
            )

    def test_error_curve_is_monotonic(self) -> None:
        """Adding principal components must not increase reconstruction error."""
        errors = reconstruction_error_curve_from_eigenvalues(
            np.array([4.0, 2.0, 0.5, 0.1]),
            sample_count=12,
            feature_count=4,
        )

        self.assertTrue(np.all(np.diff(errors) <= 1e-12))
        self.assertEqual(float(errors[-1]), 0.0)

    def test_algorithm_lab_builder_limits_preview_tables(self) -> None:
        """Algorithm lab steps should stay compact for GUI rendering."""
        matrix = np.arange(80, dtype=np.float64).reshape(10, 8)
        covariance = np.eye(8, dtype=np.float64)
        eigenvalues = np.linspace(8.0, 1.0, 8)
        eigenvectors = np.eye(8, dtype=np.float64)
        coordinates = matrix[:, :2]
        titles = [f"Artwork {index}" for index in range(10)]

        steps = PCAAlgorithmLabBuilder().build_steps(
            scaled_matrix=matrix,
            prepared_matrix=matrix - matrix.mean(axis=0),
            covariance_matrix=covariance,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            coordinates=coordinates,
            titles=titles,
        )

        self.assertEqual(len(steps), 5)
        self.assertEqual(steps[0].title, "1. 标准化后的特征矩阵 X")
        self.assertLessEqual(len(steps[0].rows), 6)
        self.assertLessEqual(len(steps[0].column_headers), 7)
        self.assertEqual(steps[3].column_headers[0], "主成分")


if __name__ == "__main__":
    unittest.main()
