"""Training workflow for ArtInsight style classification and retrieval."""

from __future__ import annotations

import json
import logging
from time import perf_counter

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from app.config import ConfigManager
from core.pca import PCAModel
from database.initializer import initialize_database
from database.session import DatabaseSessionManager
from models import History
from repositories import HistoryRepository
from services.ml.evaluation import CrossValidationEvaluator, effective_neighbor_count
from services.ml.model_artifact import (
    ModelArtifactStore,
    ModelMetadata,
    StyleModelArtifact,
    artifact_to_dict,
    default_model_path,
)
from services.ml.training_config import (
    DEFAULT_COMPONENT_CANDIDATES,
    DEFAULT_FEATURE_METHODS,
    DEFAULT_PREPROCESSING_OPS,
    ComponentSearchRow,
    TrainingConfig,
    TrainingResult,
    build_pca_config,
    validate_training_config,
)
from services.ml.training_data import TrainingDataLoader, TrainingDataset

LOGGER = logging.getLogger(__name__)


class ModelTrainer:
    """Train and persist PCA, KNN, and retrieval artifacts."""

    def __init__(
        self,
        session_manager: DatabaseSessionManager,
        artifact_store: ModelArtifactStore,
        evaluator: CrossValidationEvaluator | None = None,
    ) -> None:
        """Create a trainer with database and artifact persistence access."""
        self._session_manager = session_manager
        self._artifact_store = artifact_store
        self._evaluator = evaluator or CrossValidationEvaluator()

    @classmethod
    def from_config(cls, config: TrainingConfig | None = None) -> "ModelTrainer":
        """Create a trainer from application configuration."""
        app_config = ConfigManager().load()
        training_config = config or TrainingConfig()
        model_path = training_config.model_path or default_model_path(
            app_config.paths.model_dir,
        )
        session_manager = DatabaseSessionManager.from_url(app_config.database.url)
        initialize_database(session_manager.engine)
        return cls(
            session_manager=session_manager,
            artifact_store=ModelArtifactStore(model_path),
        )

    def train_model(self, config: TrainingConfig | None = None) -> TrainingResult:
        """Train the style classification and retrieval model.

        Args:
            config: Optional training configuration. When omitted, conservative
                defaults are used.

        Returns:
            A ``TrainingResult`` with persistence path and validation metrics.
        """
        training_config = config or TrainingConfig()
        validate_training_config(training_config)

        with self._session_manager.session_scope() as session:
            dataset = TrainingDataLoader(session).load(training_config.feature_type)
            selected_config, search_rows, note = self._resolve_training_config(
                dataset,
                training_config,
            )
            artifact, accuracy = self._fit_artifact(dataset, selected_config)
            self._artifact_store.save(artifact)
            self._record_history(session, artifact, accuracy, search_rows, note)

        LOGGER.info(
            "Trained style model: samples=%s, features=%s, components=%s, cv=%s.",
            dataset.sample_count,
            dataset.feature_count,
            artifact.metadata.n_components,
            accuracy,
        )
        return TrainingResult(
            model_path=self._artifact_store.model_path,
            sample_count=dataset.sample_count,
            feature_count=dataset.feature_count,
            style_count=len(set(dataset.style_labels)),
            n_components=artifact.metadata.n_components,
            cross_validation_accuracy=accuracy,
            component_search_rows=search_rows,
            component_selection_note=note,
        )

    def _resolve_training_config(
        self,
        dataset: TrainingDataset,
        config: TrainingConfig,
    ) -> tuple[TrainingConfig, list[ComponentSearchRow], str | None]:
        """Return a concrete training config, optionally after PCA dimension search."""
        if not config.auto_select_components:
            return config, [], None
        if config.variance_threshold is not None:
            msg = "Automatic component search cannot be combined with variance threshold."
            raise ValueError(msg)

        sampled_dataset = self._sample_for_component_search(dataset, config)
        candidates = self._component_candidates(sampled_dataset.feature_count, config)
        search_rows = self._run_component_search(
            sampled_dataset,
            config,
            candidates,
        )
        selected = self._select_component_count(search_rows, config)
        note = self._component_selection_note(search_rows, selected)
        resolved = TrainingConfig(
            feature_type=config.feature_type,
            feature_methods=config.feature_methods,
            preprocessing_ops=config.preprocessing_ops,
            n_components=selected,
            variance_threshold=None,
            n_neighbors=config.n_neighbors,
            cross_validation_folds=config.cross_validation_folds,
            model_path=config.model_path,
            auto_select_components=False,
            component_candidates=config.component_candidates,
            component_accuracy_tolerance=config.component_accuracy_tolerance,
            max_component_search_samples=config.max_component_search_samples,
            component_search_random_state=config.component_search_random_state,
        )
        LOGGER.info("Selected PCA components automatically: %s.", note)
        return resolved, search_rows, note

    def _sample_for_component_search(
        self,
        dataset: TrainingDataset,
        config: TrainingConfig,
    ) -> TrainingDataset:
        """Return a stratified sample for fast PCA component search."""
        max_samples = config.max_component_search_samples
        if max_samples is None or dataset.sample_count <= max_samples:
            return dataset
        labels = np.asarray(dataset.style_labels)
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        if unique_labels.shape[0] < 2 or int(np.min(label_counts)) < 2:
            return dataset

        test_size = max_samples / dataset.sample_count
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=test_size,
            random_state=config.component_search_random_state,
        )
        dummy = np.zeros((dataset.sample_count, 1), dtype=np.float32)
        sampled_index, _ = next(splitter.split(dummy, labels))
        sampled_index = np.sort(sampled_index)
        return TrainingDataset(
            feature_matrix=dataset.feature_matrix[sampled_index],
            style_labels=[dataset.style_labels[int(index)] for index in sampled_index],
            artwork_ids=[dataset.artwork_ids[int(index)] for index in sampled_index],
            feature_type=dataset.feature_type,
        )

    def _component_candidates(
        self,
        feature_count: int,
        config: TrainingConfig,
    ) -> tuple[int, ...]:
        """Return valid PCA component candidates in ascending order."""
        raw_candidates = list(config.component_candidates or DEFAULT_COMPONENT_CANDIDATES)
        upper_bound = min(feature_count, int(config.n_components or feature_count))
        raw_candidates.append(upper_bound)
        clipped = {
            max(1, min(int(candidate), upper_bound))
            for candidate in raw_candidates
            if candidate >= 1
        }
        return tuple(sorted(clipped))

    def _run_component_search(
        self,
        dataset: TrainingDataset,
        config: TrainingConfig,
        candidates: tuple[int, ...],
    ) -> list[ComponentSearchRow]:
        """Evaluate PCA candidates with one stratified validation split."""
        if dataset.sample_count < 4 or len(set(dataset.style_labels)) < 2:
            return [
                ComponentSearchRow(candidate, None, None)
                for candidate in candidates
            ]

        labels = np.asarray(dataset.style_labels)
        label_counts = np.unique(labels, return_counts=True)[1]
        if int(np.min(label_counts)) < 2:
            return [
                ComponentSearchRow(candidate, None, None)
                for candidate in candidates
            ]

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=config.component_search_random_state,
        )
        dummy = np.zeros((dataset.sample_count, 1), dtype=np.float32)
        train_index, test_index = next(splitter.split(dummy, labels))

        scaler = StandardScaler()
        train_features = scaler.fit_transform(dataset.feature_matrix[train_index])
        test_features = scaler.transform(dataset.feature_matrix[test_index])
        search_config = TrainingConfig(
            feature_type=config.feature_type,
            feature_methods=config.feature_methods,
            preprocessing_ops=config.preprocessing_ops,
            n_components=max(candidates),
            n_neighbors=config.n_neighbors,
            cross_validation_folds=config.cross_validation_folds,
            model_path=config.model_path,
        )
        pca_model = PCAModel(build_pca_config(search_config, dataset.feature_count))
        train_pca = pca_model.fit_transform(train_features).astype(np.float32)
        test_pca = pca_model.transform(test_features).astype(np.float32)

        neighbor_count = effective_neighbor_count(
            config.n_neighbors,
            len(train_index),
        )
        rows: list[ComponentSearchRow] = []
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        for candidate in candidates:
            component_count = min(candidate, train_pca.shape[1])
            classifier = KNeighborsClassifier(n_neighbors=neighbor_count)
            classifier.fit(train_pca[:, :component_count], train_labels)
            start = perf_counter()
            predictions = classifier.predict(test_pca[:, :component_count])
            elapsed_ms = (perf_counter() - start) * 1000
            rows.append(
                ComponentSearchRow(
                    n_components=component_count,
                    accuracy=float(np.mean(predictions == test_labels)),
                    predict_time_ms=float(elapsed_ms / len(test_index)),
                ),
            )
        return rows

    def _select_component_count(
        self,
        rows: list[ComponentSearchRow],
        config: TrainingConfig,
    ) -> int:
        """Choose the fastest small dimension whose accuracy is near the best."""
        valid_rows = [row for row in rows if row.accuracy is not None]
        if not valid_rows:
            return max(1, int(config.n_components or rows[-1].n_components))

        best_accuracy = max(float(row.accuracy) for row in valid_rows)
        minimum_accuracy = best_accuracy - config.component_accuracy_tolerance
        near_best = [
            row
            for row in valid_rows
            if row.accuracy is not None and row.accuracy >= minimum_accuracy
        ]
        near_best.sort(key=lambda row: row.n_components)
        return int(near_best[0].n_components)

    def _component_selection_note(
        self,
        rows: list[ComponentSearchRow],
        selected_components: int,
    ) -> str:
        """Create a short human-readable explanation for automatic selection."""
        valid_rows = [row for row in rows if row.accuracy is not None]
        selected = next(
            row for row in rows if row.n_components == selected_components
        )
        if not valid_rows:
            return (
                f"已选择 {selected_components} 个主成分；候选维度未能完成验证评估。"
            )
        best = max(valid_rows, key=lambda row: float(row.accuracy or 0.0))
        selected_accuracy = (
            f"{selected.accuracy:.2%}" if selected.accuracy is not None else "N/A"
        )
        selected_time = (
            f"{selected.predict_time_ms:.3f} ms/sample"
            if selected.predict_time_ms is not None
            else "N/A"
        )
        return (
            f"已选择 {selected_components} 个主成分，验证准确率 {selected_accuracy}，"
            f"预测耗时 {selected_time}；最高准确率为 {best.accuracy:.2%}"
            f"（{best.n_components} 个主成分）。"
        )

    def _fit_artifact(
        self,
        dataset: TrainingDataset,
        config: TrainingConfig,
    ) -> tuple[StyleModelArtifact, float | None]:
        """Fit scaler, manual PCA, classifier, and retrieval index."""
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(dataset.feature_matrix)
        pca_model = PCAModel(build_pca_config(config, dataset.feature_count))
        pca_features = pca_model.fit_transform(scaled_features).astype(np.float32)
        neighbor_count = effective_neighbor_count(
            config.n_neighbors,
            dataset.sample_count,
        )

        classifier = KNeighborsClassifier(n_neighbors=neighbor_count)
        classifier.fit(pca_features, dataset.style_labels)
        retriever = NearestNeighbors(metric="euclidean")
        retriever.fit(pca_features)
        accuracy = self._evaluate(dataset, config)

        metadata = ModelMetadata(
            feature_type=dataset.feature_type,
            feature_methods=config.feature_methods or DEFAULT_FEATURE_METHODS.copy(),
            preprocessing_ops=(
                config.preprocessing_ops or DEFAULT_PREPROCESSING_OPS.copy()
            ),
            n_components=pca_features.shape[1],
            sample_count=dataset.sample_count,
            feature_count=dataset.feature_count,
            style_labels=sorted(set(dataset.style_labels)),
        )
        pca_model.compact_for_prediction()
        artifact = StyleModelArtifact(
            scaler=scaler,
            pca_model=pca_model,
            classifier=classifier,
            retriever=retriever,
            artwork_ids=dataset.artwork_ids,
            labels=dataset.style_labels,
            metadata=metadata,
        )
        return artifact, accuracy

    def _evaluate(
        self,
        dataset: TrainingDataset,
        config: TrainingConfig,
    ) -> float | None:
        """Run cross-validation with the configured PCA and KNN pipeline."""
        return self._evaluator.evaluate(
            feature_matrix=dataset.feature_matrix,
            labels=dataset.style_labels,
            n_neighbors=config.n_neighbors,
            fold_count=config.cross_validation_folds,
            pca_config_factory=lambda feature_count: build_pca_config(
                config,
                feature_count,
            ),
        )

    def _record_history(
        self,
        session: Session,
        artifact: StyleModelArtifact,
        accuracy: float | None,
        search_rows: list[ComponentSearchRow] | None = None,
        selection_note: str | None = None,
    ) -> None:
        """Write a training audit record to the History table."""
        details = artifact_to_dict(artifact)
        details["cross_validation_accuracy"] = accuracy
        if search_rows:
            details["component_search_rows"] = [
                {
                    "n_components": row.n_components,
                    "accuracy": row.accuracy,
                    "predict_time_ms": row.predict_time_ms,
                }
                for row in search_rows
            ]
        if selection_note:
            details["component_selection_note"] = selection_note
        HistoryRepository(session).add(
            History(
                action="style_model_trained",
                details=json.dumps(details, ensure_ascii=False, sort_keys=True),
            ),
        )


def train_model(config: TrainingConfig | None = None) -> TrainingResult:
    """Convenience function for training with default app configuration."""
    return ModelTrainer.from_config(config).train_model(config)
