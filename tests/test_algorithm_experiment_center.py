"""Tests for the algorithm experiment center service layer."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from services.experiments import (
    AlgorithmExperimentRequest,
    AlgorithmExperimentRunner,
    ExperimentType,
)
from services.ml.training_data import TrainingDataset


class AlgorithmExperimentCenterTest(unittest.TestCase):
    """Validate experiment result, summary, retrieval, and paper figures."""

    def test_baseline_experiment_exports_core_outputs(self) -> None:
        """A small feature matrix should produce metrics and seven figures."""
        dataset = self._dataset()
        request = AlgorithmExperimentRequest(
            experiment_type=ExperimentType.BASELINE,
            algorithms=("Raw Feature + KNN", "PCA + KNN", "LDA + KNN"),
            pca_components=(2,),
            sample_limit=90,
            output_dir=Path("data/cache/test_algorithm_experiment_center"),
        )

        result = AlgorithmExperimentRunner().run_on_dataset(dataset, request)

        self.assertEqual(len(result.rows), 3)
        self.assertEqual(len(result.paper_figures), 7)
        self.assertIsNotNone(result.confusion_matrix)
        self.assertGreaterEqual(len(result.retrieval_examples), 1)
        self.assertNotEqual(result.summary.best_algorithm, "N/A")
        for figure in result.paper_figures:
            self.assertTrue(figure.path.exists())

    def test_pca_sensitivity_records_selected_dimensions(self) -> None:
        """PCA sensitivity rows should preserve the requested component counts."""
        dataset = self._dataset()
        request = AlgorithmExperimentRequest(
            experiment_type=ExperimentType.PCA_SENSITIVITY,
            pca_components=(2, 4),
            variance_thresholds=(),
            sample_limit=90,
            output_dir=Path("data/cache/test_algorithm_pca_sensitivity"),
        )

        result = AlgorithmExperimentRunner().run_on_dataset(dataset, request)

        self.assertEqual([row.pca_components for row in result.pca_rows], [2, 4])
        self.assertTrue(all(row.accuracy is not None for row in result.pca_rows))

    def _dataset(self) -> TrainingDataset:
        """Build a deterministic three-class toy dataset."""
        rng = np.random.default_rng(123)
        features: list[np.ndarray] = []
        labels: list[str] = []
        artwork_ids: list[int] = []
        for class_index, label in enumerate(("A", "B", "C")):
            center = np.zeros(18, dtype=np.float32)
            center[class_index * 3 : class_index * 3 + 3] = 2.0
            block = center + rng.normal(0, 0.35, size=(30, 18)).astype(np.float32)
            features.append(block)
            labels.extend([label] * 30)
            artwork_ids.extend(range(class_index * 30, class_index * 30 + 30))
        return TrainingDataset(
            feature_matrix=np.vstack(features),
            style_labels=labels,
            artwork_ids=artwork_ids,
            feature_type="combined",
        )


if __name__ == "__main__":
    unittest.main()
