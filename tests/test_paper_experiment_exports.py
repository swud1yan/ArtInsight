"""Tests for paper experiment export and aggregation helpers."""

from __future__ import annotations

import csv
from pathlib import Path
import unittest

from app.config import (
    Config,
    DatabaseSettings,
    LoggingSettings,
    PathSettings,
    UISettings,
)
from controllers.page_data_controller import PageDataController
from database.initializer import initialize_database
from database.session import DatabaseSessionManager
from services.experiments import (
    AlgorithmExperimentRequest,
    AlgorithmExperimentResult,
    ExperimentMetricRow,
    ExperimentResultExporter,
    ExperimentSummary,
    ExperimentSummaryBuilder,
    ExperimentType,
)
from services.experiments.experiment_models import DatasetExperimentInfo
from tests.temp_utils import ProjectTempDirectory


class PaperExperimentExportTestCase(unittest.TestCase):
    """Validate paper-oriented experiment artifacts."""

    def setUp(self) -> None:
        """Create an isolated project-local output directory."""
        self.temp_dir = ProjectTempDirectory("paper_experiment_exports")
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean generated files."""
        self.temp_dir.cleanup()

    def test_exporter_writes_metrics_summary_and_markdown(self) -> None:
        """One experiment result should be exported as CSV, JSON, and Markdown."""
        result = self._result(seed_shift=0.0)

        paths = ExperimentResultExporter().export(
            result,
            self.root / "seed_42",
            experiment_name="baseline",
            seed=42,
        )

        self.assertTrue(paths.metrics_csv.exists())
        self.assertTrue(paths.summary_json.exists())
        self.assertTrue(paths.report_md.exists())
        with paths.metrics_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["method"], "PCA + KNN")
        self.assertEqual(rows[0]["seed"], "42")
        self.assertIn("ArtInsight Paper Experiment", paths.report_md.read_text(encoding="utf-8"))

    def test_summary_builder_aggregates_and_tests_paired_methods(self) -> None:
        """Multiple exported metric files should produce aggregate tables."""
        exporter = ExperimentResultExporter()
        metrics_paths = []
        for seed, shift in ((42, 0.00), (2024, 0.02), (3407, -0.01)):
            paths = exporter.export(
                self._result(seed_shift=shift),
                self.root / f"seed_{seed}",
                experiment_name="baseline",
                seed=seed,
            )
            metrics_paths.append(paths.metrics_csv)

        builder = ExperimentSummaryBuilder()
        bundle = builder.build(
            metrics_paths,
            reference_method="PCA + KNN",
            compared_methods=("Raw Feature + KNN",),
            significance_metrics=("accuracy",),
        )
        aggregate_csv, significance_csv, markdown_path = builder.export(
            bundle,
            self.root / "summary",
        )

        self.assertEqual(len(bundle.rows), 2)
        pca_row = next(row for row in bundle.rows if row.method == "PCA + KNN")
        self.assertEqual(pca_row.run_count, 3)
        self.assertAlmostEqual(
            pca_row.metric_means["accuracy"] or 0.0,
            (0.82 + 0.84 + 0.81) / 3,
            places=6,
        )
        self.assertEqual(len(bundle.significance_rows), 1)
        self.assertEqual(bundle.significance_rows[0].paired_count, 3)
        self.assertTrue(aggregate_csv.exists())
        self.assertTrue(significance_csv.exists())
        self.assertIn("Aggregate Metrics", markdown_path.read_text(encoding="utf-8"))

    def test_controller_exports_and_summarizes_paper_results(self) -> None:
        """GUI controller should expose paper export and summary artifacts."""
        config = self._config()
        session_manager = DatabaseSessionManager.from_url(config.database.url)
        initialize_database(session_manager.engine)
        try:
            controller = PageDataController(config, session_manager)
            request = AlgorithmExperimentRequest(
                experiment_type=ExperimentType.BASELINE,
                random_state=2024,
                output_dir=(
                    config.paths.export_dir
                    / "paper_experiments"
                    / "baseline_seed_2024"
                    / "figures"
                ),
            )

            exported = controller.export_paper_experiment_result(
                self._result(seed_shift=0.0),
                request,
            )
            summary = controller.summarize_paper_experiments(
                metrics_root=config.paths.export_dir / "paper_experiments",
                output_dir=self.root / "paper_summary",
                compared_methods=("Raw Feature + KNN",),
                significance_metrics=("accuracy",),
            )
        finally:
            session_manager.engine.dispose()

        self.assertTrue(exported.metrics_csv.exists())
        self.assertTrue(exported.summary_json.exists())
        self.assertTrue(exported.report_md.exists())
        self.assertTrue(exported.figures_dir.exists())
        self.assertIn("baseline", exported.metrics_csv.read_text(encoding="utf-8-sig"))
        self.assertEqual(summary.metrics_file_count, 1)
        self.assertTrue(summary.aggregate_csv.exists())
        self.assertTrue(summary.significance_csv.exists())
        self.assertTrue(summary.report_md.exists())

    def _config(self) -> Config:
        """Build a project-local test configuration."""
        return Config(
            paths=PathSettings(
                dataset_root=self.root / "datasets",
                cache_dir=self.root / "cache",
                model_dir=self.root / "models",
                export_dir=self.root / "exports",
            ),
            database=DatabaseSettings(
                url=f"sqlite:///{(self.root / 'artinsight.sqlite3').as_posix()}",
            ),
            logging=LoggingSettings(
                log_dir=self.root / "logs",
                level="INFO",
                file_name="test.log",
            ),
            ui=UISettings(
                font_family="Arial",
                font_size=10,
                theme_path=self.root / "theme.qss",
            ),
        )

    def _result(self, seed_shift: float) -> AlgorithmExperimentResult:
        """Build a compact deterministic experiment result."""
        dataset_info = DatasetExperimentInfo(
            sample_count=120,
            feature_count=18,
            style_count=3,
            feature_type="combined",
            class_distribution={"A": 40, "B": 40, "C": 40},
        )
        rows = [
            ExperimentMetricRow(
                group="Baseline Comparison",
                method="PCA + KNN",
                accuracy=0.82 + seed_shift,
                macro_f1=0.80 + seed_shift,
                precision=0.81 + seed_shift,
                recall=0.80 + seed_shift,
                map_score=0.76 + seed_shift,
                ndcg=0.78 + seed_shift,
                train_time_ms=1.0,
                predict_time_ms=2.0,
                reduction_time_ms=3.0,
                memory_mb=0.02,
                dimension=5,
                pca_components=5,
            ),
            ExperimentMetricRow(
                group="Baseline Comparison",
                method="Raw Feature + KNN",
                accuracy=0.75 + seed_shift,
                macro_f1=0.73 + seed_shift,
                precision=0.74 + seed_shift,
                recall=0.73 + seed_shift,
                map_score=0.68 + seed_shift,
                ndcg=0.69 + seed_shift,
                train_time_ms=0.5,
                predict_time_ms=2.5,
                reduction_time_ms=0.0,
                memory_mb=0.09,
                dimension=18,
            ),
        ]
        return AlgorithmExperimentResult(
            dataset_info=dataset_info,
            rows=rows,
            baseline_rows=rows,
            feature_rows=[],
            pca_rows=[],
            confusion_matrix=None,
            retrieval_examples=[],
            paper_figures=[],
            summary=ExperimentSummary(
                best_algorithm="PCA + KNN",
                best_accuracy=0.82 + seed_shift,
                best_map_method="PCA + KNN",
                best_map=0.76 + seed_shift,
                fastest_method="Raw Feature + KNN",
                fastest_time_ms=3.0,
                lowest_memory_method="PCA + KNN",
                lowest_memory_mb=0.02,
                best_pca_dimension=5,
            ),
            discussion="PCA improves retrieval while reducing dimension.",
            logs=["ok"],
        )


if __name__ == "__main__":
    unittest.main()
