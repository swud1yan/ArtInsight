"""Aggregate exported experiment metrics across seeds."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isnan
from pathlib import Path
from statistics import mean, stdev
import warnings

import numpy as np
from scipy import stats


AGGREGATE_METRICS: tuple[str, ...] = (
    "accuracy",
    "macro_f1",
    "map_score",
    "ndcg",
    "precision_at_k",
    "recall_at_k",
    "total_time_ms",
    "memory_mb",
    "dimension",
)


@dataclass(frozen=True)
class AggregatedMetricRow:
    """One method-level mean/std row across repeated experiment runs."""

    group: str
    method: str
    run_count: int
    metric_means: dict[str, float | None]
    metric_stds: dict[str, float | None]


@dataclass(frozen=True)
class SignificanceRow:
    """Paired comparison between two methods for one metric."""

    metric: str
    method_a: str
    method_b: str
    paired_count: int
    mean_difference: float | None
    ttest_pvalue: float | None
    wilcoxon_pvalue: float | None


@dataclass(frozen=True)
class ExperimentSummaryBundle:
    """Aggregated rows and pairwise significance tests."""

    rows: list[AggregatedMetricRow]
    significance_rows: list[SignificanceRow]


class ExperimentSummaryBuilder:
    """Read exported metrics and build paper-ready aggregate tables."""

    def build(
        self,
        metrics_paths: list[Path],
        *,
        reference_method: str = "PCA + KNN",
        compared_methods: tuple[str, ...] = (
            "Raw Feature + KNN",
            "LDA + KNN",
            "Kernel PCA + KNN",
            "CLIP Embedding + KNN",
        ),
        significance_metrics: tuple[str, ...] = ("accuracy", "map_score", "ndcg"),
    ) -> ExperimentSummaryBundle:
        """Aggregate metric CSV files and compute paired significance tests."""
        raw_rows = self._read_metric_rows(metrics_paths)
        aggregate_rows = self._aggregate_rows(raw_rows)
        significance_rows = self._significance_rows(
            raw_rows,
            reference_method=reference_method,
            compared_methods=compared_methods,
            metrics=significance_metrics,
        )
        return ExperimentSummaryBundle(
            rows=aggregate_rows,
            significance_rows=significance_rows,
        )

    def export(
        self,
        bundle: ExperimentSummaryBundle,
        output_dir: Path,
    ) -> tuple[Path, Path, Path]:
        """Write aggregate CSV, significance CSV, and Markdown summary."""
        resolved_dir = output_dir.expanduser().resolve()
        resolved_dir.mkdir(parents=True, exist_ok=True)
        aggregate_csv = resolved_dir / "aggregate_metrics.csv"
        significance_csv = resolved_dir / "significance_tests.csv"
        markdown_path = resolved_dir / "paper_experiment_summary.md"
        self._write_aggregate_csv(bundle.rows, aggregate_csv)
        self._write_significance_csv(bundle.significance_rows, significance_csv)
        markdown_path.write_text(self._markdown(bundle), encoding="utf-8")
        return aggregate_csv, significance_csv, markdown_path

    def _read_metric_rows(self, metrics_paths: list[Path]) -> list[dict[str, str]]:
        """Read all metric CSV rows."""
        rows: list[dict[str, str]] = []
        for path in metrics_paths:
            resolved = path.expanduser().resolve()
            with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
                rows.extend(dict(row) for row in csv.DictReader(handle))
        return rows

    def _aggregate_rows(
        self,
        raw_rows: list[dict[str, str]],
    ) -> list[AggregatedMetricRow]:
        """Aggregate rows by group and method."""
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in raw_rows:
            grouped.setdefault((row.get("group", ""), row.get("method", "")), []).append(row)

        aggregate_rows: list[AggregatedMetricRow] = []
        for (group, method), rows in sorted(grouped.items()):
            metric_means: dict[str, float | None] = {}
            metric_stds: dict[str, float | None] = {}
            for metric in AGGREGATE_METRICS:
                values = [_parse_float(row.get(metric, "")) for row in rows]
                finite_values = [value for value in values if value is not None]
                metric_means[metric] = mean(finite_values) if finite_values else None
                metric_stds[metric] = (
                    stdev(finite_values) if len(finite_values) >= 2 else 0.0
                    if finite_values
                    else None
                )
            aggregate_rows.append(
                AggregatedMetricRow(
                    group=group,
                    method=method,
                    run_count=len(rows),
                    metric_means=metric_means,
                    metric_stds=metric_stds,
                ),
            )
        return aggregate_rows

    def _significance_rows(
        self,
        raw_rows: list[dict[str, str]],
        *,
        reference_method: str,
        compared_methods: tuple[str, ...],
        metrics: tuple[str, ...],
    ) -> list[SignificanceRow]:
        """Build paired method comparisons by experiment name and seed."""
        by_key_method: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in raw_rows:
            key = (
                row.get("experiment_name", ""),
                row.get("seed", ""),
                row.get("method", ""),
            )
            by_key_method[key] = row

        experiment_seed_pairs = sorted(
            {
                (row.get("experiment_name", ""), row.get("seed", ""))
                for row in raw_rows
            },
        )
        output: list[SignificanceRow] = []
        for metric in metrics:
            for compared_method in compared_methods:
                reference_values: list[float] = []
                compared_values: list[float] = []
                for experiment_name, seed in experiment_seed_pairs:
                    reference_row = by_key_method.get(
                        (experiment_name, seed, reference_method),
                    )
                    compared_row = by_key_method.get(
                        (experiment_name, seed, compared_method),
                    )
                    if reference_row is None or compared_row is None:
                        continue
                    reference_value = _parse_float(reference_row.get(metric, ""))
                    compared_value = _parse_float(compared_row.get(metric, ""))
                    if reference_value is None or compared_value is None:
                        continue
                    reference_values.append(reference_value)
                    compared_values.append(compared_value)
                output.append(
                    self._paired_test(
                        metric=metric,
                        method_a=reference_method,
                        method_b=compared_method,
                        values_a=reference_values,
                        values_b=compared_values,
                    ),
                )
        return output

    def _paired_test(
        self,
        *,
        metric: str,
        method_a: str,
        method_b: str,
        values_a: list[float],
        values_b: list[float],
    ) -> SignificanceRow:
        """Run paired t-test and Wilcoxon when enough pairs exist."""
        pair_count = min(len(values_a), len(values_b))
        if pair_count < 2:
            return SignificanceRow(
                metric=metric,
                method_a=method_a,
                method_b=method_b,
                paired_count=pair_count,
                mean_difference=None,
                ttest_pvalue=None,
                wilcoxon_pvalue=None,
            )
        array_a = np.asarray(values_a[:pair_count], dtype=float)
        array_b = np.asarray(values_b[:pair_count], dtype=float)
        differences = array_a - array_b
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ttest_result = stats.ttest_rel(array_a, array_b)
        ttest_pvalue = float(ttest_result.pvalue)
        if isnan(ttest_pvalue):
            ttest_pvalue = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                wilcoxon_result = stats.wilcoxon(array_a, array_b)
            wilcoxon_pvalue = float(wilcoxon_result.pvalue)
        except ValueError:
            wilcoxon_pvalue = None
        if wilcoxon_pvalue is not None and isnan(wilcoxon_pvalue):
            wilcoxon_pvalue = None
        return SignificanceRow(
            metric=metric,
            method_a=method_a,
            method_b=method_b,
            paired_count=pair_count,
            mean_difference=float(np.mean(differences)),
            ttest_pvalue=ttest_pvalue,
            wilcoxon_pvalue=wilcoxon_pvalue,
        )

    def _write_aggregate_csv(
        self,
        rows: list[AggregatedMetricRow],
        output_path: Path,
    ) -> None:
        """Write aggregate metric rows."""
        fieldnames = ["group", "method", "run_count"]
        for metric in AGGREGATE_METRICS:
            fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                payload: dict[str, object] = {
                    "group": row.group,
                    "method": row.method,
                    "run_count": row.run_count,
                }
                for metric in AGGREGATE_METRICS:
                    payload[f"{metric}_mean"] = _empty_if_none(row.metric_means[metric])
                    payload[f"{metric}_std"] = _empty_if_none(row.metric_stds[metric])
                writer.writerow(payload)

    def _write_significance_csv(
        self,
        rows: list[SignificanceRow],
        output_path: Path,
    ) -> None:
        """Write significance-test rows."""
        fieldnames = [
            "metric",
            "method_a",
            "method_b",
            "paired_count",
            "mean_difference",
            "ttest_pvalue",
            "wilcoxon_pvalue",
        ]
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "metric": row.metric,
                        "method_a": row.method_a,
                        "method_b": row.method_b,
                        "paired_count": row.paired_count,
                        "mean_difference": _empty_if_none(row.mean_difference),
                        "ttest_pvalue": _empty_if_none(row.ttest_pvalue),
                        "wilcoxon_pvalue": _empty_if_none(row.wilcoxon_pvalue),
                    },
                )

    def _markdown(self, bundle: ExperimentSummaryBundle) -> str:
        """Build a compact Markdown summary."""
        lines = [
            "# ArtInsight Paper Experiment Summary",
            "",
            "## Aggregate Metrics",
            "",
            (
                "| Group | Method | Runs | Accuracy | Macro-F1 | mAP | NDCG | "
                "Time (ms) | Memory (MB) | Dim |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in bundle.rows:
            lines.append(
                "| {group} | {method} | {runs} | {accuracy} | {macro_f1} | "
                "{map_score} | {ndcg} | {time} | {memory} | {dimension} |".format(
                    group=row.group,
                    method=row.method,
                    runs=row.run_count,
                    accuracy=_mean_std_percent(row, "accuracy"),
                    macro_f1=_mean_std_percent(row, "macro_f1"),
                    map_score=_mean_std_percent(row, "map_score"),
                    ndcg=_mean_std_percent(row, "ndcg"),
                    time=_mean_std_float(row, "total_time_ms"),
                    memory=_mean_std_float(row, "memory_mb"),
                    dimension=_mean_std_float(row, "dimension", digits=0),
                ),
            )

        lines.extend(
            [
                "",
                "## Significance Tests",
                "",
                (
                    "| Metric | Method A | Method B | Pairs | Mean Diff | "
                    "Paired t-test p | Wilcoxon p |"
                ),
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ],
        )
        for row in bundle.significance_rows:
            lines.append(
                "| {metric} | {method_a} | {method_b} | {pairs} | {diff} | "
                "{ttest} | {wilcoxon} |".format(
                    metric=row.metric,
                    method_a=row.method_a,
                    method_b=row.method_b,
                    pairs=row.paired_count,
                    diff=_format_optional(row.mean_difference),
                    ttest=_format_optional(row.ttest_pvalue),
                    wilcoxon=_format_optional(row.wilcoxon_pvalue),
                ),
            )
        lines.append("")
        return "\n".join(lines)


def _parse_float(value: str | None) -> float | None:
    """Parse a metric cell into a float when possible."""
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except ValueError:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _empty_if_none(value: float | None) -> float | str:
    """Return an empty CSV cell for missing values."""
    return "" if value is None else value


def _mean_std_percent(row: AggregatedMetricRow, metric: str) -> str:
    """Format an aggregate ratio as percent mean plus std."""
    mean_value = row.metric_means.get(metric)
    std_value = row.metric_stds.get(metric)
    if mean_value is None or std_value is None:
        return "N/A"
    return f"{mean_value * 100:.2f} +/- {std_value * 100:.2f}"


def _mean_std_float(
    row: AggregatedMetricRow,
    metric: str,
    *,
    digits: int = 3,
) -> str:
    """Format an aggregate numeric metric as mean plus std."""
    mean_value = row.metric_means.get(metric)
    std_value = row.metric_stds.get(metric)
    if mean_value is None or std_value is None:
        return "N/A"
    return f"{mean_value:.{digits}f} +/- {std_value:.{digits}f}"


def _format_optional(value: float | None) -> str:
    """Format a nullable p-value or difference."""
    if value is None:
        return "N/A"
    return f"{value:.6g}"
