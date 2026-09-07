"""Experiment runner for classification, retrieval, PCA, and efficiency studies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.decomposition import KernelPCA, PCA as SklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from core.features import FeatureExtractor
from core.pca import PCAConfig, PCAModel, PreprocessingMode
from database.session import DatabaseSessionManager
from services.experiments.experiment_models import (
    BASELINE_ALGORITHMS,
    FEATURE_SET_METHODS,
    AlgorithmExperimentRequest,
    AlgorithmExperimentResult,
    ConfusionMatrixData,
    DatasetExperimentInfo,
    ExperimentMetricRow,
    ExperimentSummary,
    ExperimentType,
    PaperFigure,
    RetrievalExample,
    RetrievalMatch,
)
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from models import Artwork
from services.experiments.visualization import ExperimentFigureExporter
from services.ml.model_artifact import ModelArtifactStore
from services.ml.training_data import TrainingDataLoader, TrainingDataset


MAX_MANUAL_PCA_DIMENSION = 700


@dataclass(frozen=True)
class _EvaluationPayload:
    """Internal payload returned by one algorithm evaluation."""

    row: ExperimentMetricRow
    y_true: np.ndarray
    y_pred: np.ndarray
    labels: list[str]
    train_embedding: np.ndarray
    test_embedding: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    pca_result: object | None = None


class AlgorithmExperimentRunner:
    """Run reproducible algorithm experiments over stored feature matrices."""

    def __init__(
        self,
        session_manager: DatabaseSessionManager | None = None,
        export_dir: Path | None = None,
        artifact_store: ModelArtifactStore | None = None,
    ) -> None:
        """Create a runner with optional database and export targets."""
        self._session_manager = session_manager
        self._export_dir = export_dir or Path("data/exports")
        self._artifact_store = artifact_store

    def run(self, request: AlgorithmExperimentRequest) -> AlgorithmExperimentResult:
        """Load configured feature data from the database and run experiments."""
        if self._session_manager is None:
            msg = "A database session manager is required for database experiments."
            raise ValueError(msg)
        with self._session_manager.session_scope() as session:
            dataset = TrainingDataLoader(session).load(request.feature_type)
            artwork_metadata = self._load_artwork_metadata(session, dataset.artwork_ids)
        return self.run_on_dataset(dataset, request, artwork_metadata)

    def run_on_dataset(
        self,
        dataset: TrainingDataset,
        request: AlgorithmExperimentRequest,
        artwork_metadata: dict[int, dict[str, str]] | None = None,
    ) -> AlgorithmExperimentResult:
        """Run experiments against an already-loaded feature matrix."""
        logs = [
            "Loading Dataset...",
            f"Loaded {dataset.sample_count} samples with {dataset.feature_count} features.",
        ]
        matrix, labels, artwork_ids, indices = self._sample_dataset(dataset, request)
        labels_array = np.asarray(labels)
        logs.append(
            f"Experiment sample: {matrix.shape[0]} samples, {matrix.shape[1]} dimensions.",
        )

        baseline_rows: list[ExperimentMetricRow] = []
        feature_rows: list[ExperimentMetricRow] = []
        pca_rows: list[ExperimentMetricRow] = []
        payloads: list[_EvaluationPayload] = []
        confusion_data: ConfusionMatrixData | None = None
        retrieval_examples: list[RetrievalExample] = []

        base_matrix = matrix
        logs.append("Baseline uses the full stored feature matrix.")

        if request.experiment_type in {
            ExperimentType.BASELINE,
            ExperimentType.CLASSIFICATION_EVALUATION,
            ExperimentType.RETRIEVAL_EVALUATION,
            ExperimentType.EFFICIENCY_BENCHMARK,
        }:
            logs.append("Running baseline algorithms...")
            for algorithm in request.algorithms:
                payload = self._evaluate_algorithm(
                    algorithm=algorithm,
                    matrix=base_matrix,
                    labels=labels_array,
                    request=request,
                    group="Baseline Comparison",
                    method_label=algorithm,
                    sample_indices=indices,
                )
                baseline_rows.append(payload.row)
                payloads.append(payload)

        if request.experiment_type == ExperimentType.FEATURE_ABLATION:
            logs.append("Running feature ablation...")
            for feature_set in request.feature_sets:
                feature_matrix, note = self._matrix_for_feature_set(
                    matrix,
                    request,
                    feature_set,
                )
                payload = self._evaluate_algorithm(
                    algorithm="Raw Feature + KNN",
                    matrix=feature_matrix,
                    labels=labels_array,
                    request=request,
                    group="Feature Ablation",
                    method_label=feature_set,
                    sample_indices=indices,
                    extra_note=note,
                )
                feature_rows.append(payload.row)
                payloads.append(payload)

        if request.experiment_type == ExperimentType.PCA_SENSITIVITY:
            logs.append("Running PCA sensitivity...")
            for component_count in request.pca_components:
                payload = self._evaluate_algorithm(
                    algorithm="PCA + KNN",
                    matrix=base_matrix,
                    labels=labels_array,
                    request=request,
                    group="PCA Sensitivity",
                    method_label=f"PCA k={component_count}",
                    sample_indices=indices,
                    pca_components=component_count,
                )
                pca_rows.append(payload.row)
                payloads.append(payload)
            for threshold in request.variance_thresholds:
                payload = self._evaluate_algorithm(
                    algorithm="PCA + KNN",
                    matrix=base_matrix,
                    labels=labels_array,
                    request=request,
                    group="PCA Sensitivity",
                    method_label=f"PCA variance={threshold:.0%}",
                    sample_indices=indices,
                    variance_threshold=threshold,
                )
                pca_rows.append(payload.row)
                payloads.append(payload)

        all_rows = baseline_rows + feature_rows + pca_rows
        if not all_rows:
            payload = self._evaluate_algorithm(
                algorithm="PCA + KNN",
                matrix=base_matrix,
                labels=labels_array,
                request=request,
                group=request.experiment_type.value,
                method_label="PCA + KNN",
                sample_indices=indices,
            )
            all_rows.append(payload.row)
            payloads.append(payload)

        best_payload = self._best_payload(payloads)
        if best_payload is not None:
            confusion_data = self._confusion_payload(best_payload)
            retrieval_examples = self._build_retrieval_examples(
                best_payload,
                dataset,
                artwork_ids,
                request,
                artwork_metadata or {},
            )

        dataset_info = DatasetExperimentInfo(
            sample_count=int(matrix.shape[0]),
            feature_count=int(matrix.shape[1]),
            style_count=len(set(labels)),
            feature_type=dataset.feature_type,
            class_distribution=dict(sorted(Counter(labels).items())),
        )
        summary = self._build_summary(all_rows)
        discussion = self._build_discussion(summary, all_rows, dataset_info)
        logs.append("Generating Report...")
        paper_figures = self._export_figures(
            dataset_info,
            all_rows,
            baseline_rows,
            feature_rows,
            pca_rows,
            confusion_data,
            best_payload,
            request,
        )
        logs.append(f"Generated {len(paper_figures)} paper figures.")

        return AlgorithmExperimentResult(
            dataset_info=dataset_info,
            rows=all_rows,
            baseline_rows=baseline_rows,
            feature_rows=feature_rows,
            pca_rows=pca_rows,
            confusion_matrix=confusion_data,
            retrieval_examples=retrieval_examples,
            paper_figures=paper_figures,
            summary=summary,
            discussion=discussion,
            logs=logs,
        )

    def _sample_dataset(
        self,
        dataset: TrainingDataset,
        request: AlgorithmExperimentRequest,
    ) -> tuple[np.ndarray, list[str], list[int], np.ndarray]:
        """Return a stratified sample suitable for interactive experiments."""
        matrix = np.asarray(dataset.feature_matrix, dtype=np.float32)
        labels = np.asarray(dataset.style_labels)
        artwork_ids = np.asarray(dataset.artwork_ids)
        if request.sample_limit <= 0 or matrix.shape[0] <= request.sample_limit:
            indices = np.arange(matrix.shape[0])
            return matrix, labels.tolist(), artwork_ids.tolist(), indices

        encoded_counts = Counter(labels.tolist())
        if min(encoded_counts.values()) < 2:
            indices = np.arange(min(request.sample_limit, matrix.shape[0]))
        else:
            _, sample_indices = train_test_split(
                np.arange(matrix.shape[0]),
                test_size=request.sample_limit,
                random_state=request.random_state,
                stratify=labels,
            )
            indices = np.sort(sample_indices)
        return (
            matrix[indices],
            labels[indices].tolist(),
            artwork_ids[indices].tolist(),
            indices,
        )

    def _matrix_for_feature_set(
        self,
        matrix: np.ndarray,
        request: AlgorithmExperimentRequest,
        feature_set: str,
    ) -> tuple[np.ndarray, str]:
        """Return a sliced feature matrix for an ablation feature set."""
        methods = FEATURE_SET_METHODS.get(feature_set)
        if methods is None:
            return matrix, f"Unknown feature set {feature_set}; using full matrix."
        if tuple(methods) == tuple(request.feature_methods):
            return matrix, ""

        slices, note = self._feature_slices(request, matrix.shape[1])
        selected: list[np.ndarray] = []
        missing: list[str] = []
        for method in methods:
            feature_slice = slices.get(method)
            if feature_slice is None:
                missing.append(method)
            else:
                selected.append(matrix[:, feature_slice])
        if not selected:
            return matrix, (
                f"Feature set {feature_set} is unavailable in stored vectors; "
                "using full matrix."
            )
        result = np.concatenate(selected, axis=1).astype(np.float32, copy=False)
        suffix = f" Missing methods: {', '.join(missing)}." if missing else ""
        return result, f"{note}{suffix}".strip()

    def _feature_slices(
        self,
        request: AlgorithmExperimentRequest,
        feature_count: int,
    ) -> tuple[dict[str, slice], str]:
        """Infer sub-feature slices by reusing the configured extractor."""
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        cursor = 0
        slices: dict[str, slice] = {}
        extractor = FeatureExtractor()
        for method in request.feature_methods:
            length = int(extractor.extract(dummy, [method]).shape[0])
            slices[method] = slice(cursor, cursor + length)
            cursor += length
        if cursor != feature_count:
            return {}, (
                "Stored feature dimension does not match configured feature layout; "
                "ablation falls back where needed."
            )
        return slices, ""

    def _evaluate_algorithm(
        self,
        *,
        algorithm: str,
        matrix: np.ndarray,
        labels: np.ndarray,
        request: AlgorithmExperimentRequest,
        group: str,
        method_label: str,
        sample_indices: np.ndarray,
        pca_components: int | None = None,
        variance_threshold: float | None = None,
        extra_note: str = "",
    ) -> _EvaluationPayload:
        """Evaluate one feature/algorithm combination."""
        if algorithm == "CLIP Embedding + KNN":
            row = ExperimentMetricRow(
                group=group,
                method=method_label,
                notes="CLIP embeddings are not configured locally.",
            )
            empty = np.empty((0, 0), dtype=np.float32)
            return _EvaluationPayload(row, labels[:0], labels[:0], [], empty, empty, labels[:0], labels[:0], sample_indices[:0], sample_indices[:0])

        if len(set(labels.tolist())) < 2 or matrix.shape[0] < 4:
            row = ExperimentMetricRow(
                group=group,
                method=method_label,
                notes="Not enough labeled samples for evaluation.",
            )
            empty = np.empty((0, 0), dtype=np.float32)
            return _EvaluationPayload(row, labels[:0], labels[:0], [], empty, empty, labels[:0], labels[:0], sample_indices[:0], sample_indices[:0])

        train_index, test_index = train_test_split(
            np.arange(matrix.shape[0]),
            test_size=0.25,
            random_state=request.random_state,
            stratify=labels,
        )
        train_raw = matrix[train_index]
        test_raw = matrix[test_index]
        train_labels = labels[train_index]
        test_labels = labels[test_index]

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_raw)

        reduction_start = perf_counter()
        note = extra_note
        pca_result = None
        if algorithm == "Raw Feature + KNN":
            train_embedding = train_scaled
            test_embedding = test_scaled
        elif algorithm == "PCA + KNN":
            artifact_payload = (
                self._artifact_pca_embedding(matrix, train_index, test_index)
                if group == "Baseline Comparison" and pca_components is None
                else None
            )
            if artifact_payload is not None:
                train_embedding, test_embedding, pca_result, artifact_note = artifact_payload
                note = self._combine_notes(note, artifact_note)
            elif train_scaled.shape[1] > MAX_MANUAL_PCA_DIMENSION:
                component_count = min(
                    pca_components or self._default_component_count(request),
                    train_scaled.shape[0] - 1,
                    train_scaled.shape[1],
                )
                reducer = SklearnPCA(
                    n_components=component_count,
                    svd_solver="randomized",
                    random_state=request.random_state,
                )
                train_embedding = reducer.fit_transform(train_scaled)
                test_embedding = reducer.transform(test_scaled)
                note = self._combine_notes(
                    note,
                    (
                        "Sensitivity scan uses sklearn randomized PCA for high-dimensional "
                        "interactive benchmarking; the project self-developed PCA remains "
                        "used for the trained system artifact."
                    ),
                )
            else:
                pca_config = self._pca_config(
                    request,
                    train_scaled.shape[1],
                    pca_components,
                    variance_threshold,
                )
                pca_model = PCAModel(pca_config)
                train_embedding = pca_model.fit_transform(train_scaled)
                test_embedding = pca_model.transform(test_scaled)
                pca_result = pca_model.result
        elif algorithm == "LDA + KNN":
            component_count = max(1, min(len(set(train_labels.tolist())) - 1, train_scaled.shape[1]))
            reducer = LinearDiscriminantAnalysis(n_components=component_count)
            train_embedding = reducer.fit_transform(train_scaled, train_labels)
            test_embedding = reducer.transform(test_scaled)
        elif algorithm == "Kernel PCA + KNN":
            component_count = min(
                pca_components or self._default_component_count(request),
                train_scaled.shape[0] - 1,
                60,
            )
            reducer = KernelPCA(
                n_components=component_count,
                kernel="rbf",
                gamma=None,
                random_state=request.random_state,
            )
            train_embedding = reducer.fit_transform(train_scaled)
            test_embedding = reducer.transform(test_scaled)
        else:
            row = ExperimentMetricRow(
                group=group,
                method=method_label,
                notes=f"Unsupported algorithm: {algorithm}",
            )
            empty = np.empty((0, 0), dtype=np.float32)
            return _EvaluationPayload(row, test_labels, test_labels[:0], [], empty, empty, train_labels, test_labels, sample_indices[train_index], sample_indices[test_index])
        reduction_time_ms = (perf_counter() - reduction_start) * 1000

        neighbor_count = max(1, min(request.n_neighbors, train_embedding.shape[0]))
        classifier = KNeighborsClassifier(n_neighbors=neighbor_count)
        train_start = perf_counter()
        classifier.fit(train_embedding, train_labels)
        train_time_ms = (perf_counter() - train_start) * 1000

        predict_start = perf_counter()
        predictions = classifier.predict(test_embedding)
        predict_time_ms = (perf_counter() - predict_start) * 1000

        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            test_labels,
            predictions,
            average="macro",
            zero_division=0,
        )
        retrieval_metrics = self._retrieval_metrics(
            train_embedding,
            train_labels,
            test_embedding,
            test_labels,
            request.top_k_values,
        )
        row = ExperimentMetricRow(
            group=group,
            method=method_label,
            accuracy=float(accuracy_score(test_labels, predictions)),
            macro_f1=float(macro_f1),
            precision=float(precision),
            recall=float(recall),
            map_score=retrieval_metrics["map"],
            precision_at_k=retrieval_metrics["precision"],
            recall_at_k=retrieval_metrics["recall"],
            ndcg=retrieval_metrics["ndcg"],
            train_time_ms=float(train_time_ms),
            predict_time_ms=float(predict_time_ms),
            reduction_time_ms=float(reduction_time_ms),
            memory_mb=self._embedding_memory_mb(train_embedding, test_embedding),
            dimension=int(train_embedding.shape[1]),
            model_size_mb=float(train_embedding.nbytes / (1024 * 1024)),
            pca_components=int(train_embedding.shape[1])
            if algorithm == "PCA + KNN"
            else None,
            cumulative_variance=self._cumulative_variance(pca_result),
            notes=note,
        )
        return _EvaluationPayload(
            row=row,
            y_true=test_labels,
            y_pred=predictions,
            labels=sorted(set(labels.tolist())),
            train_embedding=np.asarray(train_embedding, dtype=np.float32),
            test_embedding=np.asarray(test_embedding, dtype=np.float32),
            train_labels=train_labels,
            test_labels=test_labels,
            train_indices=sample_indices[train_index],
            test_indices=sample_indices[test_index],
            pca_result=pca_result,
        )

    def _artifact_pca_embedding(
        self,
        matrix: np.ndarray,
        train_index: np.ndarray,
        test_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, object, str] | None:
        """Project samples with the current system PCA artifact when compatible."""
        if self._artifact_store is None:
            return None
        try:
            artifact = self._artifact_store.load()
        except (FileNotFoundError, TypeError, ValueError, AttributeError):
            return None
        if artifact.metadata.feature_count != matrix.shape[1]:
            return None
        scaled = artifact.scaler.transform(matrix)
        coordinates = artifact.pca_model.transform(scaled)
        return (
            coordinates[train_index],
            coordinates[test_index],
            artifact.pca_result,
            "Projected by the current trained PCA artifact.",
        )

    def _combine_notes(self, first: str, second: str) -> str:
        """Join non-empty note strings."""
        notes = [note for note in (first, second) if note]
        return " ".join(notes)

    def _pca_config(
        self,
        request: AlgorithmExperimentRequest,
        feature_count: int,
        pca_components: int | None,
        variance_threshold: float | None,
    ) -> PCAConfig:
        """Build a manual PCA config for one experiment."""
        if variance_threshold is not None:
            return PCAConfig(
                variance_threshold=variance_threshold,
                preprocessing=PreprocessingMode.CENTER,
            )
        component_count = min(pca_components or self._default_component_count(request), feature_count)
        return PCAConfig(
            n_components=max(1, component_count),
            preprocessing=PreprocessingMode.CENTER,
        )

    def _default_component_count(self, request: AlgorithmExperimentRequest) -> int:
        """Return the first selected PCA component count, or a robust default."""
        if request.pca_components:
            return int(request.pca_components[0])
        return 12

    def _retrieval_metrics(
        self,
        train_embedding: np.ndarray,
        train_labels: np.ndarray,
        test_embedding: np.ndarray,
        test_labels: np.ndarray,
        top_k_values: tuple[int, ...],
    ) -> dict[str, float | None]:
        """Compute Precision@K, Recall@K, mAP, and NDCG for nearest retrieval."""
        if train_embedding.size == 0 or test_embedding.size == 0:
            return {"precision": None, "recall": None, "map": None, "ndcg": None}
        max_k = max(top_k_values or (5,))
        max_k = max(1, min(max_k, train_embedding.shape[0]))
        neighbors = NearestNeighbors(n_neighbors=max_k, metric="euclidean")
        neighbors.fit(train_embedding)
        _, indices = neighbors.kneighbors(test_embedding)
        precision_values: list[float] = []
        recall_values: list[float] = []
        ap_values: list[float] = []
        ndcg_values: list[float] = []
        label_counts = Counter(train_labels.tolist())
        for query_index, neighbor_indices in enumerate(indices):
            query_label = str(test_labels[query_index])
            relevance = np.asarray(
                [str(train_labels[index]) == query_label for index in neighbor_indices],
                dtype=float,
            )
            relevant_total = max(1, label_counts.get(query_label, 0))
            precision_values.append(float(np.mean(relevance)))
            recall_values.append(float(np.sum(relevance) / relevant_total))
            hit_count = 0.0
            precision_sum = 0.0
            for rank, is_relevant in enumerate(relevance, start=1):
                if is_relevant:
                    hit_count += 1.0
                    precision_sum += hit_count / rank
            ap_values.append(float(precision_sum / max(1.0, min(relevant_total, max_k))))
            discounts = 1.0 / np.log2(np.arange(2, relevance.size + 2))
            dcg = float(np.sum(relevance * discounts))
            ideal_relevance = np.ones(min(relevant_total, max_k), dtype=float)
            ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_relevance.size + 2))
            idcg = float(np.sum(ideal_relevance * ideal_discounts))
            ndcg_values.append(dcg / idcg if idcg else 0.0)
        return {
            "precision": float(np.mean(precision_values)),
            "recall": float(np.mean(recall_values)),
            "map": float(np.mean(ap_values)),
            "ndcg": float(np.mean(ndcg_values)),
        }

    def _embedding_memory_mb(
        self,
        train_embedding: np.ndarray,
        test_embedding: np.ndarray,
    ) -> float:
        """Estimate embedding memory footprint in megabytes."""
        return float((train_embedding.nbytes + test_embedding.nbytes) / (1024 * 1024))

    def _cumulative_variance(self, pca_result: object | None) -> float | None:
        """Return retained cumulative variance for a PCA payload."""
        if pca_result is None:
            return None
        ratios = getattr(pca_result, "cumulative_variance_ratio", None)
        projection = getattr(pca_result, "projection_matrix", None)
        if ratios is None or projection is None:
            return None
        component_count = int(projection.shape[1])
        if component_count < 1:
            return None
        return float(ratios[component_count - 1])

    def _best_payload(
        self,
        payloads: list[_EvaluationPayload],
    ) -> _EvaluationPayload | None:
        """Return the payload with the best available accuracy."""
        valid = [payload for payload in payloads if payload.row.accuracy is not None]
        if not valid:
            return None
        return max(valid, key=lambda payload: payload.row.accuracy or 0.0)

    def _confusion_payload(self, payload: _EvaluationPayload) -> ConfusionMatrixData:
        """Build a confusion matrix from the best classification payload."""
        matrix = confusion_matrix(payload.y_true, payload.y_pred, labels=payload.labels)
        return ConfusionMatrixData(
            labels=payload.labels,
            matrix=matrix.astype(int).tolist(),
            method=payload.row.method,
        )

    def _build_retrieval_examples(
        self,
        payload: _EvaluationPayload,
        dataset: TrainingDataset,
        sampled_artwork_ids: list[int],
        request: AlgorithmExperimentRequest,
        artwork_metadata: dict[int, dict[str, str]],
    ) -> list[RetrievalExample]:
        """Build deterministic query/top-k retrieval examples for the UI."""
        if payload.train_embedding.size == 0 or payload.test_embedding.size == 0:
            return []
        top_k = min(5, payload.train_embedding.shape[0])
        neighbors = NearestNeighbors(n_neighbors=top_k, metric="euclidean")
        neighbors.fit(payload.train_embedding)
        distances, indices = neighbors.kneighbors(payload.test_embedding)
        rng = np.random.default_rng(request.random_state)
        query_count = min(5, payload.test_embedding.shape[0])
        chosen = np.sort(
            rng.choice(payload.test_embedding.shape[0], size=query_count, replace=False),
        )
        examples: list[RetrievalExample] = []
        sampled_ids = np.asarray(sampled_artwork_ids)
        for query_index in chosen:
            global_index = int(payload.test_indices[query_index])
            source_position = int(global_index)
            if source_position >= len(dataset.artwork_ids):
                continue
            query_id = int(dataset.artwork_ids[source_position])
            query_meta = artwork_metadata.get(query_id, {})
            query_title = query_meta.get("title", f"Artwork {query_id}")
            matches: list[RetrievalMatch] = []
            max_distance = max(float(np.max(distances[query_index])), 1e-9)
            for rank, neighbor_position in enumerate(indices[query_index]):
                source_neighbor_position = int(payload.train_indices[neighbor_position])
                if source_neighbor_position >= len(dataset.artwork_ids):
                    continue
                sampled_artwork_id = int(dataset.artwork_ids[source_neighbor_position])
                match_meta = artwork_metadata.get(sampled_artwork_id, {})
                distance = float(distances[query_index][rank])
                matches.append(
                    RetrievalMatch(
                        title=match_meta.get("title", f"Artwork {sampled_artwork_id}"),
                        label=str(payload.train_labels[neighbor_position]),
                        image_path=match_meta.get("image_path", ""),
                        similarity_score=max(0.0, 1.0 - distance / max_distance),
                    ),
                )
            examples.append(
                RetrievalExample(
                    query_title=query_title,
                    query_label=str(dataset.style_labels[source_position]),
                    query_image_path=query_meta.get("image_path", ""),
                    matches=matches,
                ),
            )
        return examples

    def _load_artwork_metadata(
        self,
        session: object,
        artwork_ids: list[int],
    ) -> dict[int, dict[str, str]]:
        """Load titles and image paths for retrieval examples."""
        if not artwork_ids:
            return {}
        statement = (
            select(Artwork)
            .options(joinedload(Artwork.style), joinedload(Artwork.artist))
            .where(Artwork.id.in_(artwork_ids))
        )
        return {
            artwork.id: {
                "title": artwork.title,
                "image_path": artwork.image_path,
                "style": artwork.style.name if artwork.style else "",
                "artist": artwork.artist.name if artwork.artist else "",
            }
            for artwork in session.scalars(statement)
        }

    def _build_summary(self, rows: list[ExperimentMetricRow]) -> ExperimentSummary:
        """Create a compact best-result summary."""
        best_accuracy = self._best_row(rows, "accuracy")
        best_map = self._best_row(rows, "map_score")
        fastest = self._fastest_row(rows)
        lowest_memory = self._lowest_row(rows, "memory_mb")
        best_pca = self._best_pca_row(rows)
        return ExperimentSummary(
            best_algorithm=best_accuracy.method if best_accuracy else "N/A",
            best_accuracy=best_accuracy.accuracy if best_accuracy else None,
            best_map_method=best_map.method if best_map else "N/A",
            best_map=best_map.map_score if best_map else None,
            fastest_method=fastest.method if fastest else "N/A",
            fastest_time_ms=fastest.total_time_ms if fastest else None,
            lowest_memory_method=lowest_memory.method if lowest_memory else "N/A",
            lowest_memory_mb=lowest_memory.memory_mb if lowest_memory else None,
            best_pca_dimension=best_pca.pca_components if best_pca else None,
        )

    def _build_discussion(
        self,
        summary: ExperimentSummary,
        rows: list[ExperimentMetricRow],
        dataset_info: DatasetExperimentInfo,
    ) -> str:
        """Generate a paper-style discussion from measured metrics."""
        valid_rows = [row for row in rows if row.accuracy is not None]
        if not valid_rows:
            return (
                "No valid quantitative result was produced. Check whether the selected "
                "feature vectors exist and whether every class has enough samples."
            )
        raw = next((row for row in valid_rows if row.method == "Raw Feature + KNN"), None)
        pca = next((row for row in valid_rows if row.method.startswith("PCA")), None)
        lines = [
            (
                f"The best observed method is {summary.best_algorithm} on "
                f"{dataset_info.sample_count} samples and {dataset_info.style_count} "
                "style classes."
            ),
        ]
        if raw is not None and pca is not None:
            compression = (
                raw.dimension / pca.dimension
                if raw.dimension and pca.dimension
                else None
            )
            accuracy_drop = (
                raw.accuracy - pca.accuracy
                if raw.accuracy is not None and pca.accuracy is not None
                else None
            )
            if compression is not None and accuracy_drop is not None:
                lines.append(
                    f"PCA compresses the feature space by {compression:.1f}x with "
                    f"an accuracy change of {-accuracy_drop:.2%}; this highlights "
                    "the PCA trade-off between compact representation, speed, memory, "
                    "and interpretability."
                )
        lines.append(
            "Differences among feature sets reflect the complementary roles of color, "
            "texture, and shape descriptors. Color and texture are usually more stable "
            "for art-style grouping, while high-dimensional shape descriptors can improve "
            "detail sensitivity at a higher computational cost."
        )
        return "\n\n".join(lines)

    def _export_figures(
        self,
        dataset_info: DatasetExperimentInfo,
        rows: list[ExperimentMetricRow],
        baseline_rows: list[ExperimentMetricRow],
        feature_rows: list[ExperimentMetricRow],
        pca_rows: list[ExperimentMetricRow],
        confusion_data: ConfusionMatrixData | None,
        best_payload: _EvaluationPayload | None,
        request: AlgorithmExperimentRequest,
    ) -> list[PaperFigure]:
        """Render paper figures as 300-dpi PNG files."""
        output_root = request.output_dir or self._export_dir / "algorithm_experiments"
        exporter = ExperimentFigureExporter(output_root)
        return exporter.export(
            dataset_info=dataset_info,
            rows=rows,
            baseline_rows=baseline_rows,
            feature_rows=feature_rows,
            pca_rows=pca_rows,
            confusion_data=confusion_data,
            best_payload=best_payload,
        )

    def _best_row(
        self,
        rows: list[ExperimentMetricRow],
        attribute: str,
    ) -> ExperimentMetricRow | None:
        """Return row with the largest numeric attribute."""
        valid = [row for row in rows if getattr(row, attribute) is not None]
        if not valid:
            return None
        return max(valid, key=lambda row: float(getattr(row, attribute) or 0.0))

    def _lowest_row(
        self,
        rows: list[ExperimentMetricRow],
        attribute: str,
    ) -> ExperimentMetricRow | None:
        """Return row with the smallest numeric attribute."""
        valid = [row for row in rows if getattr(row, attribute) is not None]
        if not valid:
            return None
        return min(valid, key=lambda row: float(getattr(row, attribute) or 0.0))

    def _fastest_row(self, rows: list[ExperimentMetricRow]) -> ExperimentMetricRow | None:
        """Return row with the lowest measured total time."""
        valid = [row for row in rows if row.total_time_ms is not None]
        if not valid:
            return None
        return min(valid, key=lambda row: float(row.total_time_ms or 0.0))

    def _best_pca_row(
        self,
        rows: list[ExperimentMetricRow],
    ) -> ExperimentMetricRow | None:
        """Return the best PCA row by accuracy."""
        valid = [
            row
            for row in rows
            if row.pca_components is not None and row.accuracy is not None
        ]
        if not valid:
            return None
        return max(valid, key=lambda row: float(row.accuracy or 0.0))
