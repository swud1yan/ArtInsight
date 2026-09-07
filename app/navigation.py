"""Navigation definitions for the desktop shell."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NavigationRoute(str, Enum):
    """Stable route identifiers for top-level application pages."""

    HOME = "home"
    DATASET = "dataset"
    UPLOAD = "upload"
    FEATURE_EXTRACTION = "feature_extraction"
    MODEL_TRAINING = "model_training"
    PCA_ANALYSIS = "pca_analysis"
    ALGORITHM_LAB = "algorithm_lab"
    ALGORITHM_EXPERIMENTS = "algorithm_experiments"
    STYLE_CLASSIFICATION = "style_classification"
    SIMILARITY_RETRIEVAL = "similarity_retrieval"
    STYLE_MAP = "style_map"
    STATISTICS = "statistics"
    REPORT_EXPORT = "report_export"
    SETTINGS = "settings"


@dataclass(frozen=True)
class PageDefinition:
    """Metadata used to register a page in the main window."""

    route: NavigationRoute
    title: str
