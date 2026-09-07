"""Run paper-oriented algorithm experiments and export tabular artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ConfigManager
from database.session import DatabaseSessionManager
from services.experiments import (
    AlgorithmExperimentRequest,
    AlgorithmExperimentRunner,
    ExperimentResultExporter,
    ExperimentType,
)
from services.experiments.experiment_models import (
    DEFAULT_BASE_FEATURE_SET,
    DEFAULT_FEATURE_METHODS,
)
from services.ml.model_artifact import ModelArtifactStore, default_model_path


EXPERIMENT_ALIASES: dict[str, ExperimentType] = {
    "baseline": ExperimentType.BASELINE,
    "feature_ablation": ExperimentType.FEATURE_ABLATION,
    "pca_sensitivity": ExperimentType.PCA_SENSITIVITY,
    "retrieval": ExperimentType.RETRIEVAL_EVALUATION,
    "classification": ExperimentType.CLASSIFICATION_EVALUATION,
    "efficiency": ExperimentType.EFFICIENCY_BENCHMARK,
}

DEFAULT_PAPER_EXPERIMENTS: tuple[str, ...] = (
    "baseline",
    "feature_ablation",
    "pca_sensitivity",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ArtInsight paper experiments from stored database features.",
    )
    parser.add_argument(
        "--experiments",
        default=",".join(DEFAULT_PAPER_EXPERIMENTS),
        help=(
            "Comma-separated experiment names: baseline, feature_ablation, "
            "pca_sensitivity, retrieval, classification, efficiency, or all."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-type", default="combined")
    parser.add_argument(
        "--feature-methods",
        default=",".join(DEFAULT_FEATURE_METHODS),
        help="Comma-separated feature layout used by stored vectors.",
    )
    parser.add_argument("--base-feature-set", default=DEFAULT_BASE_FEATURE_SET)
    parser.add_argument(
        "--algorithms",
        default="Raw Feature + KNN,PCA + KNN,LDA + KNN,Kernel PCA + KNN",
        help="Comma-separated algorithms to evaluate.",
    )
    parser.add_argument(
        "--feature-sets",
        default=(
            "RGB,HSV,HOG,LBP,GLCM,RGB+HOG,RGB+LBP,Color+Texture,All Features"
        ),
        help="Comma-separated feature ablation sets.",
    )
    parser.add_argument("--pca-components", default="2,5,10,20,30,50,100")
    parser.add_argument("--variance-thresholds", default="0.95,0.98,0.99")
    parser.add_argument("--top-k", default="1,3,5,10")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="0 means use all database samples.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/exports/paper_experiments"),
    )
    return parser.parse_args()


def main() -> None:
    """Run selected experiments and export their results."""
    args = parse_args()
    config = ConfigManager().load()
    session_manager = DatabaseSessionManager.from_url(config.database.url)
    artifact_store = ModelArtifactStore(default_model_path(config.paths.model_dir))
    runner = AlgorithmExperimentRunner(
        session_manager=session_manager,
        export_dir=config.paths.export_dir,
        artifact_store=artifact_store,
    )
    exporter = ExperimentResultExporter()

    experiment_types = _parse_experiments(args.experiments)
    output_root = args.output_root.expanduser().resolve()
    written_metrics: list[Path] = []
    for experiment_type in experiment_types:
        experiment_name = _experiment_name(experiment_type)
        run_dir = output_root / f"{experiment_name}_seed_{args.seed}"
        request = AlgorithmExperimentRequest(
            experiment_type=experiment_type,
            algorithms=tuple(_split_csv(args.algorithms)),
            feature_sets=tuple(_split_csv(args.feature_sets)),
            pca_components=tuple(_parse_ints(args.pca_components)),
            variance_thresholds=tuple(_parse_floats(args.variance_thresholds)),
            top_k_values=tuple(_parse_ints(args.top_k)),
            feature_type=args.feature_type,
            base_feature_set=args.base_feature_set,
            feature_methods=tuple(_split_csv(args.feature_methods)),
            n_neighbors=args.neighbors,
            sample_limit=args.sample_limit,
            random_state=args.seed,
            output_dir=run_dir / "figures",
        )
        print(f"Running {experiment_name} seed={args.seed} ...", flush=True)
        result = runner.run(request)
        paths = exporter.export(
            result,
            run_dir,
            experiment_name=experiment_name,
            seed=args.seed,
        )
        written_metrics.append(paths.metrics_csv)
        print(f"  metrics: {paths.metrics_csv}", flush=True)
        print(f"  report:  {paths.report_md}", flush=True)
    session_manager.engine.dispose()

    print("Completed paper experiments.", flush=True)
    for path in written_metrics:
        print(path, flush=True)


def _parse_experiments(raw_value: str) -> list[ExperimentType]:
    """Parse selected experiment names."""
    names = [name.strip().lower() for name in raw_value.split(",") if name.strip()]
    if not names or names == ["all"]:
        names = list(DEFAULT_PAPER_EXPERIMENTS)
    experiment_types: list[ExperimentType] = []
    for name in names:
        try:
            experiment_types.append(EXPERIMENT_ALIASES[name])
        except KeyError as exc:
            valid = ", ".join(sorted([*EXPERIMENT_ALIASES, "all"]))
            raise SystemExit(f"Unknown experiment {name!r}. Valid values: {valid}") from exc
    return experiment_types


def _experiment_name(experiment_type: ExperimentType) -> str:
    """Return a filesystem-safe experiment name."""
    for name, value in EXPERIMENT_ALIASES.items():
        if value == experiment_type:
            return name
    return experiment_type.value.lower().replace(" ", "_")


def _split_csv(raw_value: str) -> list[str]:
    """Split comma-separated values while preserving method names."""
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def _parse_ints(raw_value: str) -> list[int]:
    """Parse comma-separated positive integers."""
    values: list[int] = []
    for value in _split_csv(raw_value):
        parsed = int(value)
        if parsed > 0:
            values.append(parsed)
    return values


def _parse_floats(raw_value: str) -> list[float]:
    """Parse comma-separated floats."""
    return [float(value) for value in _split_csv(raw_value)]


if __name__ == "__main__":
    main()
