"""Data transfer objects used by GUI controllers and pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AppSummary:
    """Application dashboard summary."""

    dataset_count: int
    artwork_count: int
    style_count: int
    feature_count: int
    model_available: bool
    latest_action: str
    pca_completed: bool = False
    pca_component_count: int = 0
    cumulative_variance_ratio: float | None = None
    latest_experiment_time: str = "尚未开始"
    latest_dataset_name: str = "请先导入WikiArt数据集"
    latest_accuracy: float | None = None
    latest_precision: float | None = None
    latest_recall: float | None = None


@dataclass(frozen=True)
class DatasetImportViewResult:
    """Dataset import summary returned to GUI pages."""

    dataset_id: int
    dataset_name: str
    imported_count: int
    skipped_count: int
    style_count: int
    artist_count: int


@dataclass(frozen=True)
class DatasetRow:
    """Dataset table row for the GUI."""

    dataset_id: int
    is_enabled: bool
    name: str
    path: str
    image_count: int
    style_count: int
    artist_count: int
    imported_at: str


@dataclass(frozen=True)
class DatasetOperationResult:
    """Dataset rows and optional completion message after a background task."""

    rows: list[DatasetRow]
    message: str | None = None


@dataclass(frozen=True)
class ArtworkCard:
    """Small artwork preview DTO."""

    artwork_id: int
    title: str
    style_label: str
    artist_name: str
    image_path: str


@dataclass(frozen=True)
class StyleScoreView:
    """Display score for one possible style."""

    style_label: str
    score: float
    distance: float


@dataclass(frozen=True)
class ClassificationViewResult:
    """Classification result plus same-style examples."""

    style_label: str
    confidence: float
    examples: list[ArtworkCard]
    style_scores: list[StyleScoreView]


@dataclass(frozen=True)
class SimilarArtworkViewResult:
    """Retriever result enriched with artwork metadata."""

    artwork: ArtworkCard
    distance: float


@dataclass(frozen=True)
class PCAViewData:
    """PCA eigenvalue and contribution data for charts."""

    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    n_components: int


@dataclass(frozen=True)
class PCALabStepData:
    """One teaching step shown in the PCA algorithm laboratory."""

    title: str
    description: str
    column_headers: list[str]
    rows: list[list[str]]
    note: str


@dataclass(frozen=True)
class PCAAlgorithmDemoStep:
    """One visual step for the standalone PCA algorithm laboratory page."""

    title: str
    explanation: str
    table_headers: list[str]
    table_rows: list[list[str]]
    points: np.ndarray
    vector_start: tuple[float, float] | None = None
    vector_end: tuple[float, float] | None = None
    highlight_mean: tuple[float, float] | None = None
    annotation: str = ""


@dataclass(frozen=True)
class PCAExplanationData:
    """PCA interpretation data used by the interactive analysis page."""

    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    reconstruction_errors: np.ndarray
    coordinates: np.ndarray
    labels: list[str]
    titles: list[str]
    trained_components: int
    lab_steps: list[PCALabStepData]

    @property
    def max_components(self) -> int:
        """Return the number of available principal components."""
        return int(self.eigenvalues.shape[0])


@dataclass(frozen=True)
class PCABenefitViewData:
    """Raw-feature versus PCA engineering benefit metrics."""

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


@dataclass(frozen=True)
class PCAMapPoint:
    """One artwork point in the first two PCA components."""

    artwork_id: int
    x_value: float
    y_value: float
    z_value: float | None
    style_label: str
    image_path: str
    title: str
    artist_name: str


@dataclass(frozen=True)
class UploadedMapPoint:
    """Highlighted point representing a freshly uploaded image."""

    x_value: float
    y_value: float
    z_value: float | None
    style_label: str
    artist_name: str
    confidence: float
    image_path: str
    title: str


@dataclass(frozen=True)
class FeaturePreview:
    """Single-image feature extraction preview."""

    length: int
    first_values: list[float]
    dtype_name: str


@dataclass(frozen=True)
class FeatureExtractionRunResult:
    """Batch feature extraction summary for GUI pages."""

    processed_count: int
    skipped_count: int
    failed_count: int
    feature_type: str
    feature_methods: list[str]


@dataclass(frozen=True)
class TrainingViewResult:
    """Model training summary for GUI pages."""

    model_path: Path
    sample_count: int
    feature_count: int
    style_count: int
    n_components: int
    cross_validation_accuracy: float | None
    component_selection_note: str | None = None
    component_search_rows: list[dict[str, float | int | None]] | None = None


@dataclass(frozen=True)
class ExportResult:
    """Report export operation result."""

    output_path: Path
    message: str


@dataclass(frozen=True)
class PaperExperimentExportResult:
    """Paper experiment export result."""

    output_dir: Path
    metrics_csv: Path
    summary_json: Path
    report_md: Path
    figures_dir: Path
    message: str


@dataclass(frozen=True)
class PaperExperimentSummaryResult:
    """Aggregated paper experiment summary result."""

    output_dir: Path
    aggregate_csv: Path
    significance_csv: Path
    report_md: Path
    metrics_file_count: int
    message: str
