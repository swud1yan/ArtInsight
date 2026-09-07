"""Coordinate ArtInsight report export operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from services.report_data_provider import ReportDataProvider
from services.report_html_renderer import ReportHtmlRenderer
from services.report_pdf_renderer import ReportPdfRenderer

SUPPORTED_REPORT_FORMATS = {"pdf", "html"}


@dataclass(frozen=True)
class ReportExportOptions:
    """Options supplied to one report export operation."""

    report_name: str
    output_format: str
    output_path: Path | None = None
    author: str = "ArtInsight用户"


@dataclass(frozen=True)
class GeneratedReport:
    """Result of a completed report export."""

    output_path: Path
    output_format: str


class ReportGenerator:
    """Generate template-based ArtInsight reports from application state."""

    def __init__(
        self,
        data_provider: ReportDataProvider,
        export_dir: Path,
        pdf_renderer: ReportPdfRenderer | None = None,
        html_renderer: ReportHtmlRenderer | None = None,
    ) -> None:
        """Create a report generator."""
        self._data_provider = data_provider
        self._export_dir = export_dir
        self._pdf_renderer = pdf_renderer or ReportPdfRenderer()
        self._html_renderer = html_renderer or ReportHtmlRenderer()

    def export_report(self, options: ReportExportOptions) -> GeneratedReport:
        """Generate a report file in the requested format."""
        report_format = self._normalize_format(options.output_format)
        output_path = self._resolve_output_path(options, report_format)
        self._ensure_writable(output_path)
        snapshot = self._data_provider.build_snapshot(
            report_name=options.report_name,
            author=options.author,
        )
        if report_format == "pdf":
            self._pdf_renderer.write(snapshot, output_path)
        elif report_format == "html":
            self._html_renderer.write(snapshot, output_path)
        else:
            msg = f"Unsupported report format: {report_format}"
            raise ValueError(msg)
        return GeneratedReport(output_path=output_path, output_format=report_format)

    def _normalize_format(self, output_format: str) -> str:
        """Normalize and validate the requested report format."""
        normalized = output_format.lower().strip().lstrip(".")
        if normalized not in SUPPORTED_REPORT_FORMATS:
            msg = f"报告格式仅支持：{', '.join(sorted(SUPPORTED_REPORT_FORMATS))}"
            raise ValueError(msg)
        return normalized

    def _resolve_output_path(
        self,
        options: ReportExportOptions,
        output_format: str,
    ) -> Path:
        """Resolve a safe output path for the report."""
        if options.output_path is not None:
            output_path = Path(options.output_path)
            if output_path.suffix.lower() != f".{output_format}":
                output_path = output_path.with_suffix(f".{output_format}")
            return output_path
        safe_name = self._safe_file_stem(options.report_name or "ArtInsight_Report")
        return self._export_dir / f"{safe_name}.{output_format}"

    def _ensure_writable(self, output_path: Path) -> None:
        """Ensure the report output directory is writable."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        probe_path = output_path.parent / ".write_probe"
        try:
            probe_path.write_text("ok", encoding="utf-8")
        finally:
            if probe_path.exists():
                probe_path.unlink()

    def _safe_file_stem(self, raw_name: str) -> str:
        """Convert user input to a filesystem-safe file stem."""
        stripped = raw_name.strip() or "ArtInsight_Report"
        return re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", stripped)
