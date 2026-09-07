"""Controller facade used by GUI pages to access backend services."""

from __future__ import annotations

from pathlib import Path

from app.config import Config
from controllers.artwork_card_query import ArtworkCardQuery
from controllers.dashboard_query import DashboardQuery
from controllers.dataset_page_adapter import DatasetPageAdapter
from controllers.gui_dtos import (
    AppSummary,
    ArtworkCard,
    ClassificationViewResult,
    DatasetImportViewResult,
    DatasetRow,
    ExportResult,
    FeatureExtractionRunResult,
    FeaturePreview,
    PaperExperimentExportResult,
    PaperExperimentSummaryResult,
    PCALabStepData,
    PCAAlgorithmDemoStep,
    PCABenefitViewData,
    PCAExplanationData,
    PCAMapPoint,
    PCAViewData,
    SimilarArtworkViewResult,
    TrainingViewResult,
    UploadedMapPoint,
)
from controllers.image_analysis_adapter import ImageAnalysisAdapter
from controllers.pca_view_adapter import PCAViewAdapter
from controllers.report_export_adapter import ReportExportAdapter
from controllers.settings_adapter import SettingsAdapter
from database.session import DatabaseSessionManager
from services.ml.model_artifact import ModelArtifactStore, default_model_path
from services.ml.pca_algorithm_demo import PCAAlgorithmDemoBuilder
from services.ml.pca_benefit_analysis import PCABenefitAnalyzer
from services.ml.prediction_pipeline import PredictionPipeline
from services.ml.trainer import ModelTrainer
from services.ml.training_config import TrainingConfig
from services.feature_batch_extractor import (
    FeatureBatchExtractionConfig,
    FeatureBatchExtractionService,
)
from services.experiments import (
    AlgorithmExperimentRequest,
    AlgorithmExperimentResult,
    AlgorithmExperimentRunner,
    ExperimentResultExporter,
    ExperimentSummaryBuilder,
)


class PageDataController:
    """Thin facade between GUI pages and backend-oriented adapters."""

    def __init__(self, config: Config, session_manager: DatabaseSessionManager) -> None:
        """Create a controller with app configuration and database sessions."""
        self._config = config
        model_path = default_model_path(config.paths.model_dir)
        artifact_store = ModelArtifactStore(model_path)
        prediction_pipeline = PredictionPipeline(artifact_store)
        artwork_query = ArtworkCardQuery(session_manager)
        dashboard_query = DashboardQuery(session_manager, artifact_store)
        self._dashboard_query = dashboard_query
        self._dataset_adapter = DatasetPageAdapter(session_manager)
        self._artwork_query = artwork_query
        self._image_adapter = ImageAnalysisAdapter(
            prediction_pipeline,
            artwork_query,
        )
        self._pca_adapter = PCAViewAdapter(
            session_manager,
            artifact_store,
            artwork_query,
        )
        self._pca_benefit_analyzer = PCABenefitAnalyzer(
            session_manager,
            artifact_store,
        )
        self._report_adapter = ReportExportAdapter(config, session_manager)
        self._settings_adapter = SettingsAdapter(config)
        self._algorithm_demo_builder = PCAAlgorithmDemoBuilder()
        self._feature_batch_service = FeatureBatchExtractionService(session_manager)
        self._trainer = ModelTrainer(session_manager, artifact_store)
        self._experiment_runner = AlgorithmExperimentRunner(
            session_manager=session_manager,
            export_dir=config.paths.export_dir,
            artifact_store=artifact_store,
        )
        self._experiment_exporter = ExperimentResultExporter()
        self._experiment_summary_builder = ExperimentSummaryBuilder()

    @property
    def config(self) -> Config:
        """Return immutable application configuration."""
        return self._config

    def get_summary(self) -> AppSummary:
        """Load dashboard counters and latest history action."""
        return self._dashboard_query.get_summary()

    def style_distribution(self) -> dict[str, int]:
        """Return artwork counts grouped by style."""
        return self._dashboard_query.style_distribution()

    def artist_distribution(self, limit: int = 12) -> dict[str, int]:
        """Return top artwork counts grouped by artist."""
        return self._dashboard_query.artist_distribution(limit)

    def list_datasets(self) -> list[DatasetRow]:
        """Return imported datasets with aggregate statistics."""
        return self._dataset_adapter.list_datasets()

    def import_dataset(self, dataset_path: str) -> DatasetImportViewResult:
        """Import a dataset folder through the dataset service."""
        return self._dataset_adapter.import_dataset(dataset_path)

    def delete_dataset(self, dataset_id: int) -> None:
        """Delete a dataset after confirmation in the GUI layer."""
        self._dataset_adapter.delete_dataset(dataset_id)

    def set_dataset_enabled(self, dataset_id: int, enabled: bool) -> None:
        """Enable or disable a dataset for experiment-scoped analysis."""
        self._dataset_adapter.set_dataset_enabled(dataset_id, enabled)

    def set_dataset_enabled_many(self, dataset_ids: list[int], enabled: bool) -> None:
        """Enable or disable several datasets for experiment-scoped analysis."""
        self._dataset_adapter.set_dataset_enabled_many(dataset_ids, enabled)

    def set_exclusive_enabled_datasets(self, dataset_ids: list[int]) -> None:
        """Use exactly the provided datasets for the current experiment scope."""
        self._dataset_adapter.set_exclusive_enabled_datasets(dataset_ids)

    def classify_image(self, image_path: Path) -> ClassificationViewResult:
        """Classify an image file and return same-style examples."""
        return self._image_adapter.classify_image(image_path)

    def retrieve_similar_image(
        self,
        image_path: Path,
        top_n: int = 10,
    ) -> list[SimilarArtworkViewResult]:
        """Retrieve similar artworks for an uploaded image."""
        return self._image_adapter.retrieve_similar_image(image_path, top_n)

    def extract_feature_preview(
        self,
        image_path: Path,
        preprocessing_ops: list[str],
        feature_methods: list[str],
    ) -> FeaturePreview:
        """Extract a feature vector preview for a single image file."""
        return self._image_adapter.extract_feature_preview(
            image_path,
            preprocessing_ops,
            feature_methods,
        )

    def extract_dataset_features(
        self,
        feature_methods: list[str],
        feature_type: str = "combined",
        replace_existing: bool = False,
    ) -> FeatureExtractionRunResult:
        """Extract configured features for all imported artworks."""
        result = self._feature_batch_service.extract_features(
            FeatureBatchExtractionConfig(
                feature_type=feature_type,
                feature_methods=feature_methods,
                replace_existing=replace_existing,
            ),
        )
        return FeatureExtractionRunResult(
            processed_count=result.processed_count,
            skipped_count=result.skipped_count,
            failed_count=result.failed_count,
            feature_type=result.feature_type,
            feature_methods=result.feature_methods,
        )

    def train_style_model(
        self,
        n_components: int,
        n_neighbors: int,
        cross_validation_folds: int,
        feature_type: str = "combined",
        feature_methods: list[str] | None = None,
        auto_select_components: bool = False,
    ) -> TrainingViewResult:
        """Train the style classifier and similar-artwork retrieval model."""
        result = self._trainer.train_model(
            TrainingConfig(
                feature_type=feature_type,
                feature_methods=feature_methods,
                n_components=n_components,
                n_neighbors=n_neighbors,
                cross_validation_folds=cross_validation_folds,
                auto_select_components=auto_select_components,
            ),
        )
        return TrainingViewResult(
            model_path=result.model_path,
            sample_count=result.sample_count,
            feature_count=result.feature_count,
            style_count=result.style_count,
            n_components=result.n_components,
            cross_validation_accuracy=result.cross_validation_accuracy,
            component_selection_note=result.component_selection_note,
            component_search_rows=[
                {
                    "n_components": row.n_components,
                    "accuracy": row.accuracy,
                    "predict_time_ms": row.predict_time_ms,
                }
                for row in result.component_search_rows
            ],
        )

    def run_algorithm_experiment(
        self,
        request: AlgorithmExperimentRequest,
    ) -> AlgorithmExperimentResult:
        """Run the MATLAB-style algorithm experiment center workflow."""
        return self._experiment_runner.run(request)

    def export_paper_experiment_result(
        self,
        result: AlgorithmExperimentResult,
        request: AlgorithmExperimentRequest,
        output_dir: Path | None = None,
    ) -> PaperExperimentExportResult:
        """Export one experiment run as paper-ready CSV, JSON, Markdown, and figures."""
        experiment_name = request.experiment_type.name.lower()
        seed = request.random_state
        if output_dir is not None:
            resolved_dir = output_dir
        elif request.output_dir is not None:
            resolved_dir = request.output_dir.parent
        else:
            resolved_dir = (
                self._config.paths.export_dir
                / "paper_experiments"
                / f"{experiment_name}_seed_{seed}"
            )
        exported = self._experiment_exporter.export(
            result,
            resolved_dir,
            experiment_name=experiment_name,
            seed=seed,
        )
        figures_dir = (
            result.paper_figures[0].path.parent
            if result.paper_figures
            else resolved_dir / "figures"
        )
        figures_dir.mkdir(parents=True, exist_ok=True)
        message = f"论文结果已导出至 {resolved_dir.resolve()}"
        return PaperExperimentExportResult(
            output_dir=resolved_dir.expanduser().resolve(),
            metrics_csv=exported.metrics_csv,
            summary_json=exported.summary_json,
            report_md=exported.report_md,
            figures_dir=figures_dir,
            message=message,
        )

    def summarize_paper_experiments(
        self,
        metrics_root: Path | None = None,
        output_dir: Path | None = None,
        reference_method: str = "PCA + KNN",
        compared_methods: tuple[str, ...] = (
            "Raw Feature + KNN",
            "LDA + KNN",
            "Kernel PCA + KNN",
            "CLIP Embedding + KNN",
        ),
        significance_metrics: tuple[str, ...] = ("accuracy", "map_score", "ndcg"),
    ) -> PaperExperimentSummaryResult:
        """Summarize exported paper experiments into aggregate tables."""
        resolved_metrics_root = (
            metrics_root
            if metrics_root is not None
            else self._config.paths.export_dir / "paper_experiments"
        )
        metric_paths = sorted(
            path
            for path in resolved_metrics_root.expanduser().resolve().rglob("metrics.csv")
            if path.is_file()
        )
        if not metric_paths:
            raise FileNotFoundError(
                f"No metrics.csv files were found under {resolved_metrics_root}.",
            )
        bundle = self._experiment_summary_builder.build(
            metric_paths,
            reference_method=reference_method,
            compared_methods=compared_methods,
            significance_metrics=significance_metrics,
        )
        resolved_output_dir = (
            output_dir
            if output_dir is not None
            else self._config.paths.export_dir / "paper_experiments_summary"
        )
        aggregate_csv, significance_csv, report_md = self._experiment_summary_builder.export(
            bundle,
            resolved_output_dir,
        )
        message = f"论文汇总已生成，共 {len(metric_paths)} 个 metrics.csv 文件。"
        return PaperExperimentSummaryResult(
            output_dir=resolved_output_dir.expanduser().resolve(),
            aggregate_csv=aggregate_csv,
            significance_csv=significance_csv,
            report_md=report_md,
            metrics_file_count=len(metric_paths),
            message=message,
        )

    def project_uploaded_image(self, image_path: Path) -> UploadedMapPoint:
        """Project an uploaded image for the PCA style map highlight."""
        return self._image_adapter.project_uploaded_image(image_path)

    def get_pca_view_data(self) -> PCAViewData:
        """Load PCA eigenvalue and variance contribution data."""
        return self._pca_adapter.get_pca_view_data()

    def get_pca_explanation_data(self) -> PCAExplanationData:
        """Load PCA analysis data for interactive K comparison."""
        return self._pca_adapter.get_pca_explanation_data()

    def get_pca_benefit_data(self) -> PCABenefitViewData:
        """Load raw-vs-PCA accuracy, speed, memory, and compression metrics."""
        metrics = self._pca_benefit_analyzer.analyze()
        return PCABenefitViewData(
            raw_feature_accuracy=metrics.raw_feature_accuracy,
            pca_accuracy=metrics.pca_accuracy,
            accuracy_drop=metrics.accuracy_drop,
            feature_compression_ratio=metrics.feature_compression_ratio,
            time_reduction=metrics.time_reduction,
            memory_reduction=metrics.memory_reduction,
            raw_dimension=metrics.raw_dimension,
            pca_dimension=metrics.pca_dimension,
            raw_predict_time_ms=metrics.raw_predict_time_ms,
            pca_predict_time_ms=metrics.pca_predict_time_ms,
            raw_memory_mb=metrics.raw_memory_mb,
            pca_memory_mb=metrics.pca_memory_mb,
            sample_count=metrics.sample_count,
        )

    def get_pca_algorithm_demo_steps(self) -> list[PCAAlgorithmDemoStep]:
        """Return standalone two-dimensional PCA teaching steps."""
        return self._algorithm_demo_builder.build_demo_steps()

    def get_pca_algorithm_lab_demo_steps(self) -> list[PCALabStepData]:
        """Return compact PCA teaching steps for the embedded laboratory tab."""
        return [
            PCALabStepData(
                title=step.title,
                description=step.explanation,
                column_headers=list(step.table_headers),
                rows=[list(row) for row in step.table_rows],
                note=step.annotation,
            )
            for step in self._algorithm_demo_builder.build_demo_steps()
        ]

    def get_pca_map_points(self) -> list[PCAMapPoint]:
        """Return all training artworks projected to two PCA dimensions."""
        return self._pca_adapter.get_pca_map_points()

    def list_artwork_examples(
        self,
        style_label: str | None = None,
        limit: int = 12,
    ) -> list[ArtworkCard]:
        """Return artwork preview cards, optionally filtered by style."""
        return self._artwork_query.list_examples(style_label, limit)

    def get_artwork_cards(self, artwork_ids: list[int]) -> list[ArtworkCard]:
        """Return artwork preview cards for selected artwork IDs."""
        return self._artwork_query.get_cards(artwork_ids)

    def export_report(self, report_name: str, export_format: str) -> ExportResult:
        """Export a lightweight project report."""
        return self._report_adapter.export_report(report_name, export_format)

    def save_config_values(self, values: dict[str, str]) -> None:
        """Save editable configuration values to the default JSON config file."""
        self._settings_adapter.save_config_values(values)
