"""Configuration DTOs and helpers for style model training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.pca import PCAConfig, PreprocessingMode

DEFAULT_FEATURE_METHODS = ["rgb_hist", "hsv_hist", "lbp", "glcm", "hog"]
DEFAULT_PREPROCESSING_OPS = ["resize:224x224"]
DEFAULT_COMPONENT_CANDIDATES = (5, 10, 20, 30, 50, 80, 100, 150, 200)


@dataclass(frozen=True)
class ComponentSearchRow:
    """One PCA dimension candidate evaluated during automatic selection."""

    n_components: int
    accuracy: float | None
    predict_time_ms: float | None


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training the style model."""

    feature_type: str = "combined"
    feature_methods: list[str] | None = None
    preprocessing_ops: list[str] | None = None
    n_components: int | None = 2
    variance_threshold: float | None = None
    n_neighbors: int = 5
    cross_validation_folds: int = 5
    model_path: Path | None = None
    auto_select_components: bool = False
    component_candidates: tuple[int, ...] | None = None
    component_accuracy_tolerance: float = 0.01
    max_component_search_samples: int | None = 6000
    component_search_random_state: int = 42


@dataclass(frozen=True)
class TrainingResult:
    """Summary returned after a completed training run."""

    model_path: Path
    sample_count: int
    feature_count: int
    style_count: int
    n_components: int
    cross_validation_accuracy: float | None
    component_search_rows: list[ComponentSearchRow] = field(default_factory=list)
    component_selection_note: str | None = None


def validate_training_config(config: TrainingConfig) -> None:
    """Validate training configuration before database access."""
    if config.n_neighbors < 1:
        msg = "n_neighbors must be positive."
        raise ValueError(msg)
    if config.cross_validation_folds < 2:
        msg = "cross_validation_folds must be at least 2."
        raise ValueError(msg)
    if config.n_components is not None and config.n_components < 1:
        msg = "n_components must be positive when provided."
        raise ValueError(msg)
    if config.variance_threshold is not None:
        if not 0.0 < config.variance_threshold <= 1.0:
            msg = "variance_threshold must be in the interval (0, 1]."
            raise ValueError(msg)
    if config.n_components is not None and config.variance_threshold is not None:
        msg = "Use either n_components or variance_threshold, not both."
        raise ValueError(msg)
    if config.component_accuracy_tolerance < 0.0:
        msg = "component_accuracy_tolerance must not be negative."
        raise ValueError(msg)
    if (
        config.max_component_search_samples is not None
        and config.max_component_search_samples < 2
    ):
        msg = "max_component_search_samples must be at least 2 when provided."
        raise ValueError(msg)
    if config.component_candidates is not None:
        if not config.component_candidates:
            msg = "component_candidates must not be empty when provided."
            raise ValueError(msg)
        if any(candidate < 1 for candidate in config.component_candidates):
            msg = "component_candidates must contain only positive integers."
            raise ValueError(msg)


def build_pca_config(config: TrainingConfig, feature_count: int) -> PCAConfig:
    """Build manual PCA configuration for a feature matrix."""
    if feature_count < 1:
        msg = "feature_count must be positive."
        raise ValueError(msg)
    if config.variance_threshold is not None:
        return PCAConfig(
            variance_threshold=config.variance_threshold,
            preprocessing=PreprocessingMode.CENTER,
        )
    component_count = config.n_components or min(feature_count, 2)
    component_count = min(component_count, feature_count)
    return PCAConfig(
        n_components=component_count,
        preprocessing=PreprocessingMode.CENTER,
    )
