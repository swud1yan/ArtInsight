"""DTOs used by the ArtInsight report generation service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class DatasetReportRow:
    """Dataset summary row included in generated reports."""

    name: str
    path: str
    artwork_count: int
    imported_at: datetime


@dataclass(frozen=True)
class PCAReportSummary:
    """PCA model summary included in generated reports."""

    model_available: bool
    feature_type: str
    n_components: int
    sample_count: int
    feature_count: int
    explained_variance_ratio: list[float]
    cumulative_variance_ratio: list[float]

    @property
    def cumulative_at_k(self) -> float:
        """Return cumulative explained variance at the retained component count."""
        if not self.cumulative_variance_ratio:
            return 0.0
        index = max(0, min(self.n_components, len(self.cumulative_variance_ratio)) - 1)
        return float(self.cumulative_variance_ratio[index])


@dataclass(frozen=True)
class TrainingReportSummary:
    """Training metadata extracted from audit history."""

    cross_validation_accuracy: float | None
    feature_methods: list[str]
    preprocessing_ops: list[str]
    latest_action: str


@dataclass(frozen=True)
class ReportSnapshot:
    """Complete report input assembled from the application state."""

    project_name: str
    subtitle: str
    report_name: str
    author: str
    created_at: datetime
    dataset_count: int
    artwork_count: int
    style_count: int
    feature_count: int
    datasets: list[DatasetReportRow]
    style_distribution: dict[str, int]
    artist_distribution: dict[str, int]
    pca_summary: PCAReportSummary
    training_summary: TrainingReportSummary
    example_image_paths: list[Path]
