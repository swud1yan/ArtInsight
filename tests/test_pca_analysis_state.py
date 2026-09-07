"""Tests for PCA analysis page state conversion."""

from __future__ import annotations

import unittest

import numpy as np

from controllers.gui_dtos import PCAExplanationData
from views.pca_analysis_state import PCAAnalysisState


class PCAAnalysisStateTestCase(unittest.TestCase):
    """Validate selected-K table, metric, and scatter transformations."""

    def setUp(self) -> None:
        """Create deterministic PCA explanation data."""
        self.state = PCAAnalysisState()
        self.state.set_data(
            PCAExplanationData(
                eigenvalues=np.array([3.0, 1.0, 0.5]),
                explained_variance_ratio=np.array([0.6, 0.3, 0.1]),
                cumulative_variance_ratio=np.array([0.6, 0.9, 1.0]),
                reconstruction_errors=np.array([0.2, 0.05, 0.0]),
                coordinates=np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                    ],
                    dtype=np.float64,
                ),
                labels=["A", "B"],
                titles=["Artwork A", "Artwork B"],
                trained_components=2,
                lab_steps=[],
            ),
        )

    def test_metrics_use_selected_component_count(self) -> None:
        """Selected K should select the matching variance and MSE values."""
        metrics = self.state.metrics_for(2)

        self.assertEqual(metrics.component_count, 2)
        self.assertAlmostEqual(metrics.cumulative_ratio, 0.9)
        self.assertAlmostEqual(metrics.reconstruction_mse, 0.05)
        self.assertIn("90.00%", metrics.retained_text)

    def test_one_component_scatter_sets_second_axis_to_zero(self) -> None:
        """K=1 scatter keeps PC1 and uses a zero PC2 baseline."""
        points = self.state.scatter_points(1)

        self.assertEqual([point.x_value for point in points], [1.0, 4.0])
        self.assertEqual([point.y_value for point in points], [0.0, 0.0])

    def test_two_component_scatter_uses_first_two_coordinates(self) -> None:
        """K>=2 scatter should use PC1 and PC2 coordinates."""
        points = self.state.scatter_points(3)

        self.assertEqual([point.x_value for point in points], [1.0, 4.0])
        self.assertEqual([point.y_value for point in points], [2.0, 5.0])

    def test_non_monotonic_cumulative_variance_raises_error(self) -> None:
        """Invalid cumulative contribution data should fail before rendering."""
        state = PCAAnalysisState()

        with self.assertRaises(ValueError):
            state.set_data(
                PCAExplanationData(
                    eigenvalues=np.array([2.0, 1.0]),
                    explained_variance_ratio=np.array([0.7, 0.3]),
                    cumulative_variance_ratio=np.array([0.7, 0.6]),
                    reconstruction_errors=np.array([0.1, 0.0]),
                    coordinates=np.array([[1.0, 2.0]], dtype=np.float64),
                    labels=["A"],
                    titles=["Artwork A"],
                    trained_components=1,
                    lab_steps=[],
                ),
            )


if __name__ == "__main__":
    unittest.main()
