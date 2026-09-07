"""Registry for top-level ArtInsight pages."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from app.navigation import NavigationRoute, PageDefinition
from controllers.page_data_controller import PageDataController
from views.pages.base_page import BasePage
from views.pages.algorithm_experiment_page import AlgorithmExperimentPage
from views.pages.algorithm_lab_page import AlgorithmLabPage
from views.pages.dataset_page import DatasetPage
from views.pages.feature_extraction_page import FeatureExtractionPage
from views.pages.home_page import HomePage
from views.pages.model_training_page import ModelTrainingPage
from views.pages.pca_analysis_page import PCAAnalysisPage
from views.pages.report_export_page import ReportExportPage
from views.pages.settings_page import SettingsPage
from views.pages.similarity_retrieval_page import SimilarityRetrievalPage
from views.pages.statistics_page import StatisticsPage
from views.pages.style_classification_page import StyleClassificationPage
from views.pages.style_map_page import StyleMapPage
from views.pages.upload_page import UploadPage

PageRegistration = tuple[NavigationRoute, type[BasePage]]

PAGE_REGISTRATIONS: tuple[PageRegistration, ...] = (
    (NavigationRoute.HOME, HomePage),
    (NavigationRoute.DATASET, DatasetPage),
    (NavigationRoute.FEATURE_EXTRACTION, FeatureExtractionPage),
    (NavigationRoute.MODEL_TRAINING, ModelTrainingPage),
    (NavigationRoute.PCA_ANALYSIS, PCAAnalysisPage),
    (NavigationRoute.ALGORITHM_LAB, AlgorithmLabPage),
    (NavigationRoute.UPLOAD, UploadPage),
    (NavigationRoute.STYLE_CLASSIFICATION, StyleClassificationPage),
    (NavigationRoute.SIMILARITY_RETRIEVAL, SimilarityRetrievalPage),
    (NavigationRoute.STYLE_MAP, StyleMapPage),
    (NavigationRoute.STATISTICS, StatisticsPage),
    (NavigationRoute.REPORT_EXPORT, ReportExportPage),
    (NavigationRoute.ALGORITHM_EXPERIMENTS, AlgorithmExperimentPage),
    (NavigationRoute.SETTINGS, SettingsPage),
)


def build_page_registry(
    controller: PageDataController,
) -> list[tuple[PageDefinition, QWidget]]:
    """Build page definitions and widgets for navigation."""
    return [
        (PageDefinition(route, page_class.page_title), page_class(controller))
        for route, page_class in PAGE_REGISTRATIONS
    ]
