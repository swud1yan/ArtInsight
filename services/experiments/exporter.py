"""Export algorithm experiment results into paper-ready artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from services.experiments.experiment_models import (
    AlgorithmExperimentResult,
    ExperimentMetricRow,
)
from services.style_translations import format_style_label


METRIC_COLUMNS: tuple[str, ...] = (
    "experiment_name",
    "seed",
    "group",
    "method",
    "sample_count",
    "feature_count",
    "style_count",
    "feature_type",
    "accuracy",
    "macro_f1",
    "precision",
    "recall",
    "map_score",
    "precision_at_k",
    "recall_at_k",
    "ndcg",
    "train_time_ms",
    "predict_time_ms",
    "reduction_time_ms",
    "total_time_ms",
    "memory_mb",
    "dimension",
    "model_size_mb",
    "pca_components",
    "cumulative_variance",
    "notes",
)


@dataclass(frozen=True)
class ExportedExperimentPaths:
    """Filesystem paths written for one exported experiment result."""

    metrics_csv: Path
    summary_json: Path
    report_md: Path


class ExperimentResultExporter:
    """Write one experiment result as CSV, JSON, and Markdown."""

    def export(
        self,
        result: AlgorithmExperimentResult,
        output_dir: Path,
        *,
        experiment_name: str,
        seed: int,
    ) -> ExportedExperimentPaths:
        """Export a result object to paper-oriented files."""
        resolved_dir = output_dir.expanduser().resolve()
        resolved_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = resolved_dir / "metrics.csv"
        summary_json = resolved_dir / "summary.json"
        report_md = resolved_dir / "report.md"

        metric_rows = [
            self._metric_dict(result, row, experiment_name=experiment_name, seed=seed)
            for row in result.rows
        ]
        self._write_metrics_csv(metrics_csv, metric_rows)
        summary_json.write_text(
            json.dumps(
                self._summary_dict(result, experiment_name=experiment_name, seed=seed),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report_md.write_text(
            self._markdown_report(
                result,
                metric_rows,
                experiment_name=experiment_name,
                seed=seed,
            ),
            encoding="utf-8",
        )
        return ExportedExperimentPaths(
            metrics_csv=metrics_csv,
            summary_json=summary_json,
            report_md=report_md,
        )

    def _write_metrics_csv(
        self,
        output_path: Path,
        rows: list[dict[str, object]],
    ) -> None:
        """Write metric rows with a stable column order."""
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _metric_dict(
        self,
        result: AlgorithmExperimentResult,
        row: ExperimentMetricRow,
        *,
        experiment_name: str,
        seed: int,
    ) -> dict[str, object]:
        """Convert one metric row to a CSV-friendly dictionary."""
        dataset = result.dataset_info
        return {
            "experiment_name": experiment_name,
            "seed": seed,
            "group": row.group,
            "method": row.method,
            "sample_count": dataset.sample_count,
            "feature_count": dataset.feature_count,
            "style_count": dataset.style_count,
            "feature_type": dataset.feature_type,
            "accuracy": _nullable(row.accuracy),
            "macro_f1": _nullable(row.macro_f1),
            "precision": _nullable(row.precision),
            "recall": _nullable(row.recall),
            "map_score": _nullable(row.map_score),
            "precision_at_k": _nullable(row.precision_at_k),
            "recall_at_k": _nullable(row.recall_at_k),
            "ndcg": _nullable(row.ndcg),
            "train_time_ms": _nullable(row.train_time_ms),
            "predict_time_ms": _nullable(row.predict_time_ms),
            "reduction_time_ms": _nullable(row.reduction_time_ms),
            "total_time_ms": _nullable(row.total_time_ms),
            "memory_mb": _nullable(row.memory_mb),
            "dimension": _nullable(row.dimension),
            "model_size_mb": _nullable(row.model_size_mb),
            "pca_components": _nullable(row.pca_components),
            "cumulative_variance": _nullable(row.cumulative_variance),
            "notes": row.notes,
        }

    def _summary_dict(
        self,
        result: AlgorithmExperimentResult,
        *,
        experiment_name: str,
        seed: int,
    ) -> dict[str, Any]:
        """Build a JSON summary for one exported experiment."""
        return {
            "experiment_name": experiment_name,
            "seed": seed,
            "dataset": asdict(result.dataset_info),
            "summary": asdict(result.summary),
            "discussion": result.discussion,
            "rows": [self._row_as_json(row) for row in result.rows],
            "paper_figures": [
                {
                    "name": figure.name,
                    "title": figure.title,
                    "path": str(figure.path),
                }
                for figure in result.paper_figures
            ],
            "logs": list(result.logs),
        }

    def _row_as_json(self, row: ExperimentMetricRow) -> dict[str, object]:
        """Convert an experiment row to a JSON-friendly dictionary."""
        payload = asdict(row)
        payload["total_time_ms"] = row.total_time_ms
        return payload

    def _markdown_report(
        self,
        result: AlgorithmExperimentResult,
        rows: list[dict[str, object]],
        *,
        experiment_name: str,
        seed: int,
    ) -> str:
        """Build a concise Markdown report for one experiment run."""
        dataset = result.dataset_info
        lines = [
            f"# ArtInsight Paper Experiment: {experiment_name}",
            "",
            f"- Seed: `{seed}`",
            f"- Samples: `{dataset.sample_count}`",
            f"- Feature dimension: `{dataset.feature_count}`",
            f"- Styles: `{dataset.style_count}`",
            f"- Feature type: `{dataset.feature_type}`",
            "",
            "## Dataset Distribution",
            "",
            "| Style | Samples |",
            "| --- | ---: |",
        ]
        for label, count in dataset.class_distribution.items():
            lines.append(f"| {format_style_label(label)} | {count} |")

        lines.extend(
            [
                "",
                "## Metrics",
                "",
                (
                    "| Group | Method | Accuracy | Macro-F1 | mAP | NDCG | "
                    "Time (ms) | Memory (MB) | Dim |"
                ),
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ],
        )
        for row in rows:
            lines.append(
                "| {group} | {method} | {accuracy} | {macro_f1} | "
                "{map_score} | {ndcg} | {total_time_ms} | {memory_mb} | "
                "{dimension} |".format(
                    group=row["group"],
                    method=row["method"],
                    accuracy=_format_percent(row["accuracy"]),
                    macro_f1=_format_percent(row["macro_f1"]),
                    map_score=_format_percent(row["map_score"]),
                    ndcg=_format_percent(row["ndcg"]),
                    total_time_ms=_format_float(row["total_time_ms"]),
                    memory_mb=_format_float(row["memory_mb"]),
                    dimension=_format_float(row["dimension"], digits=0),
                ),
            )

        lines.extend(["", "## Auto Discussion", "", result.discussion, ""])
        if result.paper_figures:
            lines.extend(["## Figures", ""])
            for figure in result.paper_figures:
                lines.append(f"- {figure.title}: `{figure.path}`")
            lines.append("")
        return "\n".join(lines)


def _nullable(value: object | None) -> object:
    """Return an empty string for missing values so CSV stays compact."""
    return "" if value is None else value


def _format_float(value: object, digits: int = 3) -> str:
    """Format a possibly empty numeric value."""
    if value == "" or value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}f}"


def _format_percent(value: object) -> str:
    """Format a ratio value as a percentage."""
    if value == "" or value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)
