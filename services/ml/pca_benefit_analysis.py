"""Engineering-benefit metrics for the PCA analysis page."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from database.session import DatabaseSessionManager
from services.ml.model_artifact import ModelArtifactStore
from services.ml.training_data import TrainingDataLoader, TrainingDataset


@dataclass(frozen=True)
class PCABenefitMetrics:
    """Raw-vs-PCA comparison metrics for analysis pages."""

    raw_feature_accuracy: float | None
    pca_accuracy: float | None
    accuracy_drop: float | None
    feature_compression_ratio: float | None
    time_reduction: float | None
    memory_reduction: float | None
    raw_dimension: int
    pca_dimension: int
    raw_predict_time_ms: float | None
    pca_predict_time_ms: float | None
    raw_memory_mb: float | None
    pca_memory_mb: float | None
    sample_count: int


class PCABenefitAnalyzer:
    """Evaluate PCA's accuracy/speed/memory trade-off without retraining artifacts."""

    def __init__(
        self,
        session_manager: DatabaseSessionManager,
        artifact_store: ModelArtifactStore,
        sample_limit: int = 600,
        random_state: int = 42,
    ) -> None:
        """Create an analyzer backed by database features and model artifact."""
        self._session_manager = session_manager
        self._artifact_store = artifact_store
        self._sample_limit = sample_limit
        self._random_state = random_state

    def analyze(self) -> PCABenefitMetrics:
        """Return Raw Feature vs PCA metrics for the current trained model."""
        artifact = self._artifact_store.load()
        with self._session_manager.session_scope() as session:
            dataset = TrainingDataLoader(session).load(artifact.metadata.feature_type)
        sampled = self._sample_dataset(dataset)
        labels = np.asarray(sampled.style_labels)
        if len(set(labels.tolist())) < 2 or sampled.sample_count < 4:
            return self._empty_metrics(
                raw_dimension=sampled.feature_count,
                pca_dimension=artifact.metadata.n_components,
                sample_count=sampled.sample_count,
            )

        train_index, test_index = train_test_split(
            np.arange(sampled.sample_count),
            test_size=0.25,
            random_state=self._random_state,
            stratify=labels,
        )
        raw_train, raw_test = (
            sampled.feature_matrix[train_index],
            sampled.feature_matrix[test_index],
        )
        y_train, y_test = labels[train_index], labels[test_index]

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(raw_train)
        test_scaled = scaler.transform(raw_test)
        raw_accuracy, raw_time = self._score_knn(train_scaled, test_scaled, y_train, y_test)

        artifact_scaled = artifact.scaler.transform(sampled.feature_matrix)
        pca_coordinates = artifact.pca_model.transform(artifact_scaled)
        pca_train, pca_test = pca_coordinates[train_index], pca_coordinates[test_index]
        pca_accuracy, pca_time = self._score_knn(pca_train, pca_test, y_train, y_test)

        raw_memory = self._memory_mb(train_scaled, test_scaled)
        pca_memory = self._memory_mb(pca_train, pca_test)
        return PCABenefitMetrics(
            raw_feature_accuracy=raw_accuracy,
            pca_accuracy=pca_accuracy,
            accuracy_drop=(
                raw_accuracy - pca_accuracy
                if raw_accuracy is not None and pca_accuracy is not None
                else None
            ),
            feature_compression_ratio=(
                sampled.feature_count / artifact.metadata.n_components
                if artifact.metadata.n_components > 0
                else None
            ),
            time_reduction=self._relative_reduction(raw_time, pca_time),
            memory_reduction=self._relative_reduction(raw_memory, pca_memory),
            raw_dimension=sampled.feature_count,
            pca_dimension=artifact.metadata.n_components,
            raw_predict_time_ms=raw_time,
            pca_predict_time_ms=pca_time,
            raw_memory_mb=raw_memory,
            pca_memory_mb=pca_memory,
            sample_count=sampled.sample_count,
        )

    def _sample_dataset(self, dataset: TrainingDataset) -> TrainingDataset:
        """Return a stratified sample for fast interactive analysis."""
        if self._sample_limit <= 0 or dataset.sample_count <= self._sample_limit:
            return dataset
        labels = np.asarray(dataset.style_labels)
        if min(Counter(labels.tolist()).values()) < 2:
            indices = np.arange(self._sample_limit)
        else:
            _, indices = train_test_split(
                np.arange(dataset.sample_count),
                test_size=self._sample_limit,
                random_state=self._random_state,
                stratify=labels,
            )
            indices = np.sort(indices)
        return TrainingDataset(
            feature_matrix=dataset.feature_matrix[indices],
            style_labels=[dataset.style_labels[int(index)] for index in indices],
            artwork_ids=[dataset.artwork_ids[int(index)] for index in indices],
            feature_type=dataset.feature_type,
        )

    def _score_knn(
        self,
        train_features: np.ndarray,
        test_features: np.ndarray,
        train_labels: np.ndarray,
        test_labels: np.ndarray,
    ) -> tuple[float, float]:
        """Fit KNN and measure prediction time."""
        neighbor_count = max(1, min(5, train_features.shape[0]))
        classifier = KNeighborsClassifier(n_neighbors=neighbor_count)
        classifier.fit(train_features, train_labels)
        start = perf_counter()
        predictions = classifier.predict(test_features)
        elapsed_ms = (perf_counter() - start) * 1000
        return float(accuracy_score(test_labels, predictions)), float(elapsed_ms)

    def _memory_mb(self, *arrays: np.ndarray) -> float:
        """Return combined array memory in megabytes."""
        return float(sum(array.nbytes for array in arrays) / (1024 * 1024))

    def _relative_reduction(
        self,
        original: float | None,
        reduced: float | None,
    ) -> float | None:
        """Return fractional reduction from original to reduced."""
        if original is None or reduced is None or original <= 0:
            return None
        return float((original - reduced) / original)

    def _empty_metrics(
        self,
        *,
        raw_dimension: int,
        pca_dimension: int,
        sample_count: int,
    ) -> PCABenefitMetrics:
        """Return a structured empty result when data is insufficient."""
        return PCABenefitMetrics(
            raw_feature_accuracy=None,
            pca_accuracy=None,
            accuracy_drop=None,
            feature_compression_ratio=(
                raw_dimension / pca_dimension if pca_dimension > 0 else None
            ),
            time_reduction=None,
            memory_reduction=None,
            raw_dimension=raw_dimension,
            pca_dimension=pca_dimension,
            raw_predict_time_ms=None,
            pca_predict_time_ms=None,
            raw_memory_mb=None,
            pca_memory_mb=None,
            sample_count=sample_count,
        )
