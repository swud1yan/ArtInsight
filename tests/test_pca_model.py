"""Tests for the PCA model workflow."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SklearnPCA

from core.pca import PCAModel, PCAConfig, PreprocessingMode


class PCAModelTestCase(unittest.TestCase):
    """Validate the stateful PCA model against deterministic data and sklearn."""

    def setUp(self) -> None:
        """Create a matrix with correlated feature columns."""
        rng = np.random.default_rng(11)
        latent = rng.normal(size=(120, 3))
        self.data = np.column_stack(
            [
                latent[:, 0] + 0.1 * latent[:, 1],
                latent[:, 1],
                latent[:, 2],
                0.5 * latent[:, 0] - 0.2 * latent[:, 2],
                0.3 * latent[:, 1] + 0.4 * latent[:, 2],
            ],
        )

    def test_fit_transform_and_inverse_transform_shapes(self) -> None:
        """Fitted model should project and reconstruct matrices with stable shapes."""
        model = PCAModel(
            PCAConfig(n_components=3, preprocessing=PreprocessingMode.CENTER),
        )

        result = model.fit(self.data)
        transformed = model.transform(self.data[:5])
        reconstructed = model.inverse_transform(transformed)

        self.assertTrue(model.is_fitted)
        self.assertEqual(result.transformed_data.shape, (120, 3))
        self.assertEqual(result.projection_matrix.shape, (5, 3))
        self.assertEqual(transformed.shape, (5, 3))
        self.assertEqual(reconstructed.shape, (5, 5))
        self.assertGreaterEqual(result.reconstruction_mse, 0.0)

    def test_all_components_reconstruct_original_data(self) -> None:
        """Using every component should reconstruct centered PCA input exactly."""
        model = PCAModel(
            PCAConfig(n_components=5, preprocessing=PreprocessingMode.CENTER),
        )

        result = model.fit(self.data)

        self.assertLess(result.reconstruction_mse, 1e-20)

    def test_manual_pca_matches_sklearn_explained_variance(self) -> None:
        """Manual PCA eigenvalues and ratios should match sklearn validation."""
        manual = PCAModel(
            PCAConfig(n_components=3, preprocessing=PreprocessingMode.CENTER),
        )
        manual_result = manual.fit(self.data)
        sklearn_model = SklearnPCA(n_components=3, svd_solver="full")

        sklearn_transformed = sklearn_model.fit_transform(self.data)

        np.testing.assert_allclose(
            manual_result.eigenvalues[:3],
            sklearn_model.explained_variance_,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            manual_result.explained_variance_ratio[:3],
            sklearn_model.explained_variance_ratio_,
            rtol=1e-10,
            atol=1e-10,
        )
        for index in range(3):
            correlation = np.corrcoef(
                manual_result.transformed_data[:, index],
                sklearn_transformed[:, index],
            )[0, 1]
            self.assertGreater(abs(float(correlation)), 0.999999)

    def test_variance_threshold_on_iris_dataset(self) -> None:
        """Variance threshold selection should work on the Iris dataset."""
        iris = load_iris()
        model = PCAModel(
            PCAConfig(
                variance_threshold=0.95,
                preprocessing=PreprocessingMode.STANDARDIZE,
            ),
        )

        result = model.fit(iris.data.astype(np.float64))
        selected_count = result.projection_matrix.shape[1]

        self.assertGreaterEqual(
            float(result.cumulative_variance_ratio[selected_count - 1]),
            0.95,
        )
        self.assertEqual(result.transformed_data.shape[0], iris.data.shape[0])

    def test_unfitted_model_raises_error(self) -> None:
        """Transforming before fit should fail clearly."""
        model = PCAModel(PCAConfig(n_components=2))

        with self.assertRaises(ValueError):
            model.transform(self.data)

    def test_conflicting_component_configuration_raises_error(self) -> None:
        """Component count and variance threshold should not be combined."""
        model = PCAModel(PCAConfig(n_components=2, variance_threshold=0.9))

        with self.assertRaises(ValueError):
            model.fit(self.data)


if __name__ == "__main__":
    unittest.main()