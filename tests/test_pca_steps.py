"""Tests for manual PCA matrix operations."""

from __future__ import annotations

import unittest

import numpy as np

from core.pca import (
    compute_covariance,
    compute_eigen,
    compute_mean,
    explained_variance,
    inverse_transform,
    mean_center,
    reconstruction_error,
    select_components,
    standardize,
    transform,
)


class PCAstepsTestCase(unittest.TestCase):
    """Validate textbook PCA steps independently from the model wrapper."""

    def setUp(self) -> None:
        """Create a deterministic correlated matrix."""
        rng = np.random.default_rng(7)
        base = rng.normal(size=(80, 3))
        self.data = np.column_stack(
            [
                base[:, 0],
                0.7 * base[:, 0] + 0.2 * base[:, 1],
                base[:, 1] - 0.1 * base[:, 2],
                0.5 * base[:, 2],
            ],
        )

    def test_mean_center_removes_column_mean(self) -> None:
        """Mean centering should leave every feature with near-zero mean."""
        centered = mean_center(self.data)

        np.testing.assert_allclose(centered.mean(axis=0), np.zeros(4), atol=1e-12)
        np.testing.assert_allclose(compute_mean(self.data), self.data.mean(axis=0))

    def test_standardize_handles_constant_feature(self) -> None:
        """Standardization should avoid zero division for constant columns."""
        data = np.column_stack([self.data, np.ones(self.data.shape[0])])

        standardized, mean, scale = standardize(data)

        np.testing.assert_allclose(standardized.mean(axis=0), np.zeros(5), atol=1e-12)
        self.assertEqual(float(scale[-1]), 1.0)
        self.assertEqual(float(mean[-1]), 1.0)
        np.testing.assert_allclose(standardized[:, -1], np.zeros(data.shape[0]))

    def test_compute_covariance_matches_numpy(self) -> None:
        """Covariance should match NumPy's sample covariance convention."""
        covariance = compute_covariance(self.data)
        expected = np.cov(self.data, rowvar=False, bias=False)

        np.testing.assert_allclose(covariance, expected, atol=1e-12)

    def test_compute_eigen_returns_sorted_eigenpairs(self) -> None:
        """Eigenvalues should be descending and paired with eigenvectors."""
        covariance = compute_covariance(self.data)
        eigenvalues, eigenvectors = compute_eigen(covariance)

        self.assertTrue(np.all(np.diff(eigenvalues) <= 0.0))
        residual = covariance @ eigenvectors[:, 0] - eigenvalues[0] * eigenvectors[:, 0]
        self.assertLess(float(np.linalg.norm(residual)), 1e-10)

    def test_transform_inverse_transform_reconstruct_with_all_components(self) -> None:
        """All principal components should reconstruct the input nearly exactly."""
        mean = compute_mean(self.data)
        centered = mean_center(self.data)
        covariance = compute_covariance(centered)
        eigenvalues, eigenvectors = compute_eigen(covariance)
        projection = select_components(eigenvalues, eigenvectors, n_components=4)

        transformed = transform(centered, projection)
        reconstructed = inverse_transform(transformed, projection, mean)

        self.assertEqual(transformed.shape, self.data.shape)
        self.assertLess(reconstruction_error(self.data, reconstructed), 1e-20)

    def test_explained_variance_ratios_sum_to_one(self) -> None:
        """Explained variance ratios should form a cumulative contribution curve."""
        covariance = compute_covariance(self.data)
        eigenvalues, _ = compute_eigen(covariance)

        ratio, cumulative = explained_variance(eigenvalues)

        self.assertAlmostEqual(float(ratio.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(cumulative[-1]), 1.0, places=12)
        self.assertTrue(np.all(np.diff(cumulative) >= 0.0))

    def test_invalid_component_count_raises_error(self) -> None:
        """Selecting too many components should fail clearly."""
        covariance = compute_covariance(self.data)
        eigenvalues, eigenvectors = compute_eigen(covariance)

        with self.assertRaises(ValueError):
            select_components(eigenvalues, eigenvectors, n_components=99)


if __name__ == "__main__":
    unittest.main()
