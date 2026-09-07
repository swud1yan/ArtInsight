"""Cross-validation utilities for style classification training."""

from __future__ import annotations

from collections.abc import Callable
import logging

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from core.pca import PCAModel, PCAConfig

LOGGER = logging.getLogger(__name__)


def effective_neighbor_count(requested: int, sample_count: int) -> int:
    """Clamp a KNN neighbor count to the available sample count."""
    return max(1, min(requested, sample_count))


class CrossValidationEvaluator:
    """Evaluate PCA + KNN classification accuracy without data leakage."""

    def evaluate(
        self,
        feature_matrix: np.ndarray,
        labels: list[str],
        n_neighbors: int,
        fold_count: int,
        pca_config_factory: Callable[[int], PCAConfig],
    ) -> float | None:
        """Run stratified cross validation for the training pipeline.

        Args:
            feature_matrix: Raw feature matrix with shape
                ``(n_samples, n_features)``.
            labels: Style labels aligned with feature rows.
            n_neighbors: Requested KNN neighbor count.
            fold_count: Requested number of stratified folds.
            pca_config_factory: Factory receiving feature count and returning
                the manual PCA configuration for a fold.

        Returns:
            Mean classification accuracy, or ``None`` when the dataset is too
            small or contains only one style.
        """
        label_array = np.asarray(labels)
        if len(set(label_array)) < 2:
            LOGGER.warning("Skipping cross validation because only one style exists.")
            return None

        encoded_labels = LabelEncoder().fit_transform(label_array)
        class_counts = np.bincount(encoded_labels)
        max_folds = int(np.min(class_counts))
        actual_fold_count = min(fold_count, max_folds)
        if actual_fold_count < 2:
            LOGGER.warning(
                "Skipping cross validation because class counts are too low.",
            )
            return None

        splitter = StratifiedKFold(
            n_splits=actual_fold_count,
            shuffle=True,
            random_state=42,
        )
        scores: list[float] = []
        for train_index, test_index in splitter.split(feature_matrix, label_array):
            scores.append(
                self._score_fold(
                    feature_matrix=feature_matrix,
                    labels=label_array,
                    train_index=train_index,
                    test_index=test_index,
                    n_neighbors=n_neighbors,
                    pca_config_factory=pca_config_factory,
                ),
            )
        return float(np.mean(scores))

    def _score_fold(
        self,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        train_index: np.ndarray,
        test_index: np.ndarray,
        n_neighbors: int,
        pca_config_factory: Callable[[int], PCAConfig],
    ) -> float:
        """Fit scaler, PCA, and KNN on one fold and return accuracy."""
        scaler = StandardScaler()
        train_features = scaler.fit_transform(feature_matrix[train_index])
        test_features = scaler.transform(feature_matrix[test_index])
        pca_model = PCAModel(pca_config_factory(train_features.shape[1]))
        train_pca = pca_model.fit_transform(train_features)
        test_pca = pca_model.transform(test_features)
        neighbor_count = effective_neighbor_count(n_neighbors, len(train_index))
        classifier = KNeighborsClassifier(n_neighbors=neighbor_count)
        classifier.fit(train_pca, labels[train_index])
        return float(classifier.score(test_pca, labels[test_index]))