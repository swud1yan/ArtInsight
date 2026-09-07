"""DTOs for the algorithm experiment center."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ExperimentType(str, Enum):
    """Supported experiment workflows."""

    BASELINE = "Baseline Comparison"
    FEATURE_ABLATION = "Feature Ablation"
    PCA_SENSITIVITY = "PCA Sensitivity"
    RETRIEVAL_EVALUATION = "Retrieval Evaluation"
    CLASSIFICATION_EVALUATION = "Classification Evaluation"
    EFFICIENCY_BENCHMARK = "Efficiency Benchmark"


BASELINE_ALGORITHMS: tuple[str, ...] = (
    "Raw Feature + KNN",
    "PCA + KNN",
    "LDA + KNN",
    "Kernel PCA + KNN",
    "CLIP Embedding + KNN",
)

FEATURE_SET_METHODS: dict[str, tuple[str, ...]] = {
    "RGB": ("rgb_hist",),
    "HSV": ("hsv_hist",),
    "HOG": ("hog",),
    "LBP": ("lbp",),
    "GLCM": ("glcm",),
    "RGB+HOG": ("rgb_hist", "hog"),
    "RGB+LBP": ("rgb_hist", "lbp"),
    "Color+Texture": ("rgb_hist", "hsv_hist", "lbp", "glcm"),
    "All Features": ("rgb_hist", "hsv_hist", "hog", "lbp", "glcm"),
}

DEFAULT_FEATURE_METHODS: tuple[str, ...] = FEATURE_SET_METHODS["All Features"]
DEFAULT_BASE_FEATURE_SET = "Color+Texture"
DEFAULT_PCA_COMPONENTS: tuple[int, ...] = (2, 5, 10, 20, 30, 50, 100)
DEFAULT_VARIANCE_THRESHOLDS: tuple[float, ...] = (0.95, 0.98, 0.99)
DEFAULT_TOP_K: tuple[int, ...] = (1, 3, 5, 10)


@dataclass(frozen=True)
class AlgorithmExperimentRequest:
    """Configuration selected in the algorithm experiment center."""

    experiment_type: ExperimentType = ExperimentType.BASELINE
    algorithms: tuple[str, ...] = BASELINE_ALGORITHMS[:4]
    feature_sets: tuple[str, ...] = tuple(FEATURE_SET_METHODS)
    pca_components: tuple[int, ...] = DEFAULT_PCA_COMPONENTS
    variance_thresholds: tuple[float, ...] = DEFAULT_VARIANCE_THRESHOLDS
    top_k_values: tuple[int, ...] = DEFAULT_TOP_K
    feature_type: str = "combined"
    base_feature_set: str = DEFAULT_BASE_FEATURE_SET
    feature_methods: tuple[str, ...] = DEFAULT_FEATURE_METHODS
    n_neighbors: int = 5
    sample_limit: int = 600
    random_state: int = 42
    output_dir: Path | None = None


@dataclass(frozen=True)
class ExperimentMetricRow:
    """One row of method-level experimental metrics."""

    group: str
    method: str
    accuracy: float | None = None
    macro_f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    map_score: float | None = None
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    ndcg: float | None = None
    train_time_ms: float | None = None
    predict_time_ms: float | None = None
    reduction_time_ms: float | None = None
    memory_mb: float | None = None
    dimension: int | None = None
    model_size_mb: float | None = None
    pca_components: int | None = None
    cumulative_variance: float | None = None
    notes: str = ""

    @property
    def total_time_ms(self) -> float | None:
        """Return total measured time used by result tables."""
        values = [
            value
            for value in (
                self.train_time_ms,
                self.predict_time_ms,
                self.reduction_time_ms,
            )
            if value is not None
        ]
        if not values:
            return None
        return float(sum(values))


@dataclass(frozen=True)
class ConfusionMatrixData:
    """Confusion matrix payload for charts."""

    labels: list[str]
    matrix: list[list[int]]
    method: str


@dataclass(frozen=True)
class RetrievalMatch:
    """One retrieved neighbor shown in the experiment UI."""

    title: str
    label: str
    image_path: str
    similarity_score: float


@dataclass(frozen=True)
class RetrievalExample:
    """Query image and its top retrieved neighbors."""

    query_title: str
    query_label: str
    query_image_path: str
    matches: list[RetrievalMatch]


@dataclass(frozen=True)
class PaperFigure:
    """Generated 300-dpi figure for paper experiments."""

    name: str
    title: str
    path: Path


@dataclass(frozen=True)
class ExperimentSummary:
    """Auto-generated result summary for paper discussion."""

    best_algorithm: str
    best_accuracy: float | None
    best_map_method: str
    best_map: float | None
    fastest_method: str
    fastest_time_ms: float | None
    lowest_memory_method: str
    lowest_memory_mb: float | None
    best_pca_dimension: int | None


@dataclass(frozen=True)
class DatasetExperimentInfo:
    """Dataset metadata used by the experiment run."""

    sample_count: int
    feature_count: int
    style_count: int
    feature_type: str
    class_distribution: dict[str, int]


@dataclass(frozen=True)
class AlgorithmExperimentResult:
    """Complete output consumed by the PySide6 experiment center."""

    dataset_info: DatasetExperimentInfo
    rows: list[ExperimentMetricRow]
    baseline_rows: list[ExperimentMetricRow]
    feature_rows: list[ExperimentMetricRow]
    pca_rows: list[ExperimentMetricRow]
    confusion_matrix: ConfusionMatrixData | None
    retrieval_examples: list[RetrievalExample]
    paper_figures: list[PaperFigure]
    summary: ExperimentSummary
    discussion: str
    logs: list[str] = field(default_factory=list)
