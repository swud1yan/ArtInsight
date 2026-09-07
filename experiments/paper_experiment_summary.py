"""Aggregate exported paper experiment metrics into manuscript tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.experiments import ExperimentSummaryBuilder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize ArtInsight paper experiment metrics across seeds.",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("data/exports/paper_experiments"),
        help="Root folder searched recursively for metrics.csv files.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        type=Path,
        default=(),
        help="Optional explicit metrics.csv paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/paper_experiments"),
    )
    parser.add_argument("--reference-method", default="PCA + KNN")
    parser.add_argument(
        "--compared-methods",
        default="Raw Feature + KNN,LDA + KNN,Kernel PCA + KNN,CLIP Embedding + KNN",
    )
    parser.add_argument(
        "--significance-metrics",
        default="accuracy,map_score,ndcg",
    )
    return parser.parse_args()


def main() -> None:
    """Build aggregate CSV/Markdown files."""
    args = parse_args()
    metric_paths = _discover_metrics(args.metrics_root, list(args.metrics))
    if not metric_paths:
        raise SystemExit(
            "No metrics.csv files were found. Run experiments.paper_experiment_runner first.",
        )

    builder = ExperimentSummaryBuilder()
    bundle = builder.build(
        metric_paths,
        reference_method=args.reference_method,
        compared_methods=tuple(_split_csv(args.compared_methods)),
        significance_metrics=tuple(_split_csv(args.significance_metrics)),
    )
    aggregate_csv, significance_csv, markdown_path = builder.export(
        bundle,
        args.output_dir,
    )
    print(f"Read metrics files: {len(metric_paths)}", flush=True)
    print(f"Aggregate CSV: {aggregate_csv}", flush=True)
    print(f"Significance CSV: {significance_csv}", flush=True)
    print(f"Markdown summary: {markdown_path}", flush=True)


def _discover_metrics(metrics_root: Path, explicit_paths: list[Path]) -> list[Path]:
    """Return explicit and recursively discovered metrics files."""
    discovered = [path.expanduser().resolve() for path in explicit_paths]
    root = metrics_root.expanduser().resolve()
    if root.exists():
        discovered.extend(path.resolve() for path in root.rglob("metrics.csv"))
    unique: dict[str, Path] = {}
    for path in discovered:
        if path.exists() and path.is_file():
            unique[str(path)] = path
    return sorted(unique.values(), key=lambda path: str(path))


def _split_csv(raw_value: str) -> list[str]:
    """Split comma-separated values."""
    return [value.strip() for value in raw_value.split(",") if value.strip()]


if __name__ == "__main__":
    main()
