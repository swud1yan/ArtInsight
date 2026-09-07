"""Report export adapter for GUI pages."""

from __future__ import annotations

from app.config import Config
from controllers.gui_dtos import ExportResult
from database.session import DatabaseSessionManager
from services.ml.model_artifact import ModelArtifactStore, default_model_path
from services.report_data_provider import ReportDataProvider
from services.report_generator import ReportExportOptions, ReportGenerator


class ReportExportAdapter:
    """Export project analysis reports for the GUI."""

    def __init__(
        self,
        config: Config,
        session_manager: DatabaseSessionManager,
    ) -> None:
        """Create the report adapter."""
        self._config = config
        artifact_store = ModelArtifactStore(default_model_path(config.paths.model_dir))
        data_provider = ReportDataProvider(session_manager, artifact_store)
        self._report_generator = ReportGenerator(
            data_provider=data_provider,
            export_dir=config.paths.export_dir,
        )

    def export_report(self, report_name: str, export_format: str) -> ExportResult:
        """Export an analysis report for the current application state."""
        generated = self._report_generator.export_report(
            ReportExportOptions(
                report_name=report_name,
                output_format=export_format,
                author="ArtInsight用户",
            ),
        )
        return ExportResult(
            output_path=generated.output_path,
            message=f"{generated.output_format.upper()}报告已导出。",
        )
