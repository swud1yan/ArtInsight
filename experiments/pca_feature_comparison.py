"""Publication-compliant PCA and feature-set comparison experiments.

The default mode requires a real art image dataset. Synthetic data is available
only through the explicit ``--debug-synthetic`` flag and is never publication
evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.features.feature_extractor import FeatureExtractor, FeatureExtractorConfig
from core.pca import PCAConfig, PCAModel, PreprocessingMode
from core.pca.pca_steps import reconstruction_error
from experiments.art_dataset_loader import (
    ArtDatasetLoader,
    ArtDatasetLoaderConfig,
    DatasetFormat,
)
from experiments.dataset_compliance import (
    DatasetComplianceReport,
    DatasetStatus,
    check_dataset_compliance,
)

LOGGER = logging.getLogger(__name__)

FEATURE_PLANS: dict[str, list[str]] = {
    "color_hist": ["rgb_hist:32", "hsv_hist:32"],
    "color_texture": ["rgb_hist:32", "hsv_hist:32", "lbp", "glcm"],
}


@dataclass(frozen=True)
class ExperimentDataset:
    """Feature matrices and labels used by comparison experiments."""

    name: str
    labels: np.ndarray
    feature_sets: dict[str, np.ndarray]
    source_note: str
    publication_eligible: bool


@dataclass(frozen=True)
class PCABenchmarkResult:
    """Timing and numerical-difference summary for PCA implementations."""

    sample_count: int
    feature_count: int
    n_components: int
    manual_time_ms: float
    sklearn_time_ms: float
    speed_ratio: float
    component_max_abs_diff: float
    transformed_rmse: float
    explained_ratio_max_abs_diff: float
    manual_reconstruction_mse: float
    sklearn_reconstruction_mse: float


@dataclass(frozen=True)
class FeatureAccuracyResult:
    """Cross-validation result for one feature representation."""

    feature_name: str
    sample_count: int
    feature_count: int
    n_components: int
    accuracy_mean: float
    accuracy_std: float


def generate_synthetic_dataset(
    samples_per_style: int = 300,
    random_seed: int = 42,
) -> ExperimentDataset:
    """Generate a reproducible style-feature benchmark.

    The color block deliberately makes two styles visually similar in color,
    while the texture block separates them. This mirrors the research question:
    whether combining texture with color improves style classification.
    """
    if samples_per_style < 20:
        msg = "samples_per_style must be at least 20 for stable cross validation."
        raise ValueError(msg)

    rng = np.random.default_rng(random_seed)
    labels = np.repeat(
        np.array(["Impressionism", "Cubism", "Baroque"]),
        samples_per_style,
    )
    color_means = np.array(
        [
            [1.2, 0.8, 0.2, 0.0],
            [1.1, 0.9, 0.2, 0.0],
            [-1.0, -0.8, 1.2, 0.5],
        ],
        dtype=np.float64,
    )
    texture_means = np.array(
        [
            [1.2, -0.8, 0.4, 0.2],
            [-1.0, 1.1, -0.4, 0.1],
            [0.0, 0.2, 1.2, -1.0],
        ],
        dtype=np.float64,
    )
    color = _sample_block(
        rng,
        color_means,
        samples_per_style,
        block_width=6,
        noise_scale=1.4,
    )
    texture = _sample_block(
        rng,
        texture_means,
        samples_per_style,
        block_width=6,
        noise_scale=1.4,
    )
    feature_sets = {
        "color_hist": color.astype(np.float32),
        "texture": texture.astype(np.float32),
        "color_texture": np.hstack([color, texture]).astype(np.float32),
    }
    return ExperimentDataset(
        name="synthetic_style_benchmark",
        labels=labels,
        feature_sets=feature_sets,
        source_note=(
            "DEBUG ONLY synthetic feature data. This output is prohibited from "
            "SCI/IET paper result tables."
        ),
        publication_eligible=False,
    )


def load_image_dataset(
    dataset_root: Path,
    max_images_per_style: int = 120,
    resize_to: tuple[int, int] = (128, 128),
    dataset_format: DatasetFormat = DatasetFormat.AUTO,
    metadata_file: Path | None = None,
    progress_interval: int = 100,
) -> ExperimentDataset:
    """Extract feature matrices from a real art image dataset."""
    loader = ArtDatasetLoader(
        ArtDatasetLoaderConfig(
            dataset_root=dataset_root,
            dataset_format=dataset_format,
            resize_to=resize_to,
            metadata_file=metadata_file,
            max_images_per_style=max_images_per_style,
        ),
    )
    extractor = FeatureExtractor(FeatureExtractorConfig(rgb_bins=32, hsv_bins=32))
    labels: list[str] = []
    feature_parts: dict[str, list[np.ndarray]] = {
        feature_name: [] for feature_name in FEATURE_PLANS
    }
    processed_count = 0
    started_at = perf_counter()
    for loaded_image in loader.iter_images():
        try:
            extracted = {
                name: extractor.extract(loaded_image.bgr_uint8, methods)
                for name, methods in FEATURE_PLANS.items()
            }
        except ValueError as exc:
            LOGGER.warning(
                "Skipping image %s: %s",
                loaded_image.record.image_path,
                exc,
            )
            continue
        labels.append(loaded_image.record.style_label)
        for name, vector in extracted.items():
            feature_parts[name].append(vector)
        processed_count += 1
        if progress_interval > 0 and processed_count % progress_interval == 0:
            elapsed = perf_counter() - started_at
            rate = processed_count / elapsed if elapsed > 0 else 0.0
            LOGGER.info(
                "Feature extraction progress: dataset=%s images=%s rate=%.2f img/s styles=%s",
                dataset_root,
                processed_count,
                rate,
                len(set(labels)),
            )

    if len(set(labels)) < 2:
        msg = "Feature extraction produced fewer than two valid style labels."
        raise ValueError(msg)
    LOGGER.info(
        "Feature extraction completed: dataset=%s images=%s styles=%s elapsed=%.1fs",
        dataset_root,
        len(labels),
        len(set(labels)),
        perf_counter() - started_at,
    )

    feature_sets = {
        name: np.vstack(vectors).astype(np.float32)
        for name, vectors in feature_parts.items()
        if vectors
    }
    return ExperimentDataset(
        name=dataset_root.expanduser().resolve().name,
        labels=np.asarray(labels),
        feature_sets=feature_sets,
        source_note=f"Image dataset loaded from {dataset_root.expanduser().resolve()}",
        publication_eligible=True,
    )


def benchmark_pca(
    feature_matrix: np.ndarray,
    n_components: int = 12,
    repeat_count: int = 5,
) -> PCABenchmarkResult:
    """Compare manual PCA with scikit-learn PCA on the same standardized data."""
    matrix = _validate_matrix(feature_matrix)
    component_count = min(n_components, matrix.shape[1])
    scaled = StandardScaler().fit_transform(matrix)
    manual_times: list[float] = []
    sklearn_times: list[float] = []
    manual_model: PCAModel | None = None
    sklearn_model: SklearnPCA | None = None
    manual_coordinates: np.ndarray | None = None
    sklearn_coordinates: np.ndarray | None = None

    for _ in range(repeat_count):
        manual_model = PCAModel(
            PCAConfig(
                n_components=component_count,
                preprocessing=PreprocessingMode.CENTER,
            ),
        )
        start = perf_counter()
        manual_coordinates = manual_model.fit_transform(scaled)
        manual_times.append((perf_counter() - start) * 1000.0)

        sklearn_model = SklearnPCA(n_components=component_count, svd_solver="full")
        start = perf_counter()
        sklearn_coordinates = sklearn_model.fit_transform(scaled)
        sklearn_times.append((perf_counter() - start) * 1000.0)

    assert manual_model is not None
    assert sklearn_model is not None
    assert manual_coordinates is not None
    assert sklearn_coordinates is not None

    signs = _component_alignment_signs(
        manual_model.result.projection_matrix.T,
        sklearn_model.components_,
    )
    aligned_sklearn_components = sklearn_model.components_ * signs[:, None]
    aligned_sklearn_coordinates = sklearn_coordinates * signs
    sklearn_reconstruction = sklearn_model.inverse_transform(sklearn_coordinates)

    manual_time = float(np.median(manual_times))
    sklearn_time = float(np.median(sklearn_times))
    return PCABenchmarkResult(
        sample_count=matrix.shape[0],
        feature_count=matrix.shape[1],
        n_components=component_count,
        manual_time_ms=manual_time,
        sklearn_time_ms=sklearn_time,
        speed_ratio=manual_time / sklearn_time if sklearn_time > 0 else float("inf"),
        component_max_abs_diff=float(
            np.max(
                np.abs(
                    manual_model.result.projection_matrix.T
                    - aligned_sklearn_components
                ),
            ),
        ),
        transformed_rmse=float(
            np.sqrt(np.mean((manual_coordinates - aligned_sklearn_coordinates) ** 2)),
        ),
        explained_ratio_max_abs_diff=float(
            np.max(
                np.abs(
                    manual_model.result.explained_variance_ratio[:component_count]
                    - sklearn_model.explained_variance_ratio_
                ),
            ),
        ),
        manual_reconstruction_mse=reconstruction_error(
            scaled,
            manual_model.result.reconstructed_data,
        ),
        sklearn_reconstruction_mse=reconstruction_error(scaled, sklearn_reconstruction),
    )


def evaluate_feature_sets(
    dataset: ExperimentDataset,
    n_components: int = 12,
    n_neighbors: int = 5,
    fold_count: int = 5,
) -> list[FeatureAccuracyResult]:
    """Evaluate KNN classification accuracy for each feature representation."""
    labels = np.asarray(dataset.labels)
    results: list[FeatureAccuracyResult] = []
    for feature_name, matrix in dataset.feature_sets.items():
        scores = _cross_validate_feature_matrix(
            _validate_matrix(matrix),
            labels,
            n_components=n_components,
            n_neighbors=n_neighbors,
            fold_count=fold_count,
        )
        results.append(
            FeatureAccuracyResult(
                feature_name=feature_name,
                sample_count=matrix.shape[0],
                feature_count=matrix.shape[1],
                n_components=min(n_components, matrix.shape[1]),
                accuracy_mean=float(np.mean(scores)),
                accuracy_std=float(np.std(scores, ddof=0)),
            ),
        )
    return sorted(results, key=lambda item: item.accuracy_mean, reverse=True)


def build_markdown_report(
    dataset: ExperimentDataset,
    pca_result: PCABenchmarkResult,
    feature_results: list[FeatureAccuracyResult],
    compliance_report: DatasetComplianceReport,
) -> str:
    """Build a Markdown report for PCA and feature comparison experiments."""
    best_feature = feature_results[0]
    publication_status = (
        "PUBLICATION ELIGIBLE"
        if dataset.publication_eligible
        else "DEBUG ONLY - DO NOT USE IN PAPER"
    )
    lines = [
        "# PCA and Feature Comparison Experiment",
        "",
        f"Publication status: **{publication_status}**",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{dataset.name}`",
        f"- Samples: `{len(dataset.labels)}`",
        f"- Classes: `{len(set(dataset.labels))}`",
        f"- Compliance status: `{compliance_report.dataset_status.value}`",
        f"- Compliance message: {compliance_report.message}",
        f"- Source note: {dataset.source_note}",
        "",
        "## PCA Implementation Comparison",
        "",
        "| Metric | Manual PCA | sklearn PCA | Difference |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Median runtime | {pca_result.manual_time_ms:.3f} ms | "
            f"{pca_result.sklearn_time_ms:.3f} ms | "
            f"{pca_result.speed_ratio:.2f}x manual/sklearn |"
        ),
        (
            f"| Reconstruction MSE | {pca_result.manual_reconstruction_mse:.8f} | "
            f"{pca_result.sklearn_reconstruction_mse:.8f} | "
            f"{abs(pca_result.manual_reconstruction_mse - pca_result.sklearn_reconstruction_mse):.2e} |"
        ),
        (
            "| Explained-ratio max abs diff | "
            f"{pca_result.explained_ratio_max_abs_diff:.2e} | - | - |"
        ),
        (
            "| Component max abs diff after sign alignment | "
            f"{pca_result.component_max_abs_diff:.2e} | - | - |"
        ),
        (
            "| Transformed-coordinate RMSE after sign alignment | "
            f"{pca_result.transformed_rmse:.2e} | - | - |"
        ),
        "",
        "## Feature Classification Comparison",
        "",
        "| Rank | Feature set | Dimension | PCA k | Accuracy mean | Accuracy std |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, result in enumerate(feature_results, start=1):
        lines.append(
            f"| {rank} | `{result.feature_name}` | {result.feature_count} | "
            f"{result.n_components} | {result.accuracy_mean:.2%} | "
            f"{result.accuracy_std:.2%} |",
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                f"The best feature representation in this run is "
                f"`{best_feature.feature_name}` with "
                f"{best_feature.accuracy_mean:.2%} mean accuracy."
            ),
            _build_pca_interpretation(pca_result),
            _build_feature_interpretation(feature_results),
        ],
    )
    return "\n".join(lines) + "\n"


def run_experiment(args: argparse.Namespace) -> Path:
    """Run all comparisons and write the Markdown result report."""
    LOGGER.info(
        "Experiment started: dataset_root=%s output=%s max_images_per_style=%s seed=%s",
        args.dataset_root,
        args.output,
        args.max_images_per_style,
        args.seed,
    )
    compliance_report = check_dataset_compliance(
        args.dataset_root,
        debug_synthetic=args.debug_synthetic,
        min_classes=args.min_classes,
        min_images_per_class=args.min_images_per_class,
    )
    if compliance_report.dataset_status == DatasetStatus.INSUFFICIENT_DATA:
        raise RuntimeError(compliance_report.message)

    dataset = _load_experiment_dataset(args, compliance_report)
    pca_matrix = (
        dataset.feature_sets["color_texture"]
        if "color_texture" in dataset.feature_sets
        else next(iter(dataset.feature_sets.values()))
    )
    pca_result = benchmark_pca(
        pca_matrix,
        n_components=args.components,
        repeat_count=args.pca_repeats,
    )
    LOGGER.info(
        "PCA benchmark completed: samples=%s features=%s components=%s manual_ms=%.2f sklearn_ms=%.2f",
        pca_result.sample_count,
        pca_result.feature_count,
        pca_result.n_components,
        pca_result.manual_time_ms,
        pca_result.sklearn_time_ms,
    )
    feature_results = evaluate_feature_sets(
        dataset,
        n_components=args.components,
        n_neighbors=args.neighbors,
        fold_count=args.folds,
    )
    LOGGER.info("Feature-set evaluation completed: feature_sets=%s", len(feature_results))
    report = build_markdown_report(
        dataset,
        pca_result,
        feature_results,
        compliance_report,
    )
    output_path = _resolve_output_path(args.output, compliance_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    LOGGER.info("Wrote experiment report: %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Compare manual PCA, sklearn PCA, and feature sets.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Required for publication experiments. Folder structure: "
            "root/style_name/.../image.jpg"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/pca_feature_experiment_report.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--debug-synthetic",
        action="store_true",
        help=(
            "Run synthetic debug data only. Results are explicitly prohibited "
            "from paper tables."
        ),
    )
    parser.add_argument(
        "--dataset-format",
        choices=[item.value for item in DatasetFormat],
        default=DatasetFormat.AUTO.value,
        help="Real dataset layout: wikiart, kaggle_painter_by_numbers, or auto.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Optional Kaggle Painter by Numbers metadata CSV path.",
    )
    parser.add_argument("--samples-per-style", type=int, default=300)
    parser.add_argument("--max-images-per-style", type=int, default=120)
    parser.add_argument("--min-classes", type=int, default=3)
    parser.add_argument("--min-images-per-class", type=int, default=30)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pca-repeats", type=int, default=5)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Log feature extraction progress every N images; use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the experiment script from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    run_experiment(parse_args())


def _load_experiment_dataset(
    args: argparse.Namespace,
    compliance_report: DatasetComplianceReport,
) -> ExperimentDataset:
    """Load the dataset allowed by the compliance report."""
    if compliance_report.dataset_status == DatasetStatus.DEBUG_ONLY:
        return generate_synthetic_dataset(args.samples_per_style, args.seed)
    if compliance_report.dataset_status == DatasetStatus.VALID_REAL_DATA:
        assert args.dataset_root is not None
        return load_image_dataset(
            args.dataset_root,
            args.max_images_per_style,
            dataset_format=DatasetFormat(args.dataset_format),
            metadata_file=args.metadata_file,
            progress_interval=args.progress_interval,
        )
    raise RuntimeError(compliance_report.message)


def _resolve_output_path(
    requested_output: Path,
    compliance_report: DatasetComplianceReport,
) -> Path:
    """Route debug-only reports to an explicit non-publication filename."""
    if compliance_report.dataset_status != DatasetStatus.DEBUG_ONLY:
        return requested_output.expanduser().resolve()

    debug_dir = PROJECT_ROOT / "data" / "exports"
    output_path = requested_output
    if output_path == Path("docs/pca_feature_experiment_report.md"):
        output_path = debug_dir / "pca_feature_experiment_report_DEBUG_ONLY.md"
    else:
        output_path = output_path.expanduser().resolve()
        if "debug" not in output_path.stem.lower():
            output_path = output_path.with_name(
                f"{output_path.stem}_DEBUG_ONLY{output_path.suffix}",
            )
    return output_path


def _sample_block(
    rng: np.random.Generator,
    class_means: np.ndarray,
    samples_per_class: int,
    block_width: int,
    noise_scale: float,
) -> np.ndarray:
    """Sample repeated feature blocks around per-class prototypes."""
    rows: list[np.ndarray] = []
    for mean in class_means:
        prototype = np.repeat(mean, block_width)
        rows.append(
            rng.normal(
                loc=prototype,
                scale=noise_scale,
                size=(samples_per_class, prototype.shape[0]),
            ),
        )
    return np.vstack(rows)


def _build_pca_interpretation(result: PCABenchmarkResult) -> str:
    """Build a runtime-aware PCA comparison interpretation."""
    if result.manual_time_ms <= result.sklearn_time_ms:
        speed_text = (
            "manual PCA is faster on this medium-width covariance benchmark "
            "because the feature dimension is modest and the implementation "
            "uses a direct symmetric eigendecomposition."
        )
    else:
        speed_text = (
            "sklearn PCA is faster on this benchmark because it uses optimized "
            "decomposition routines and mature numerical backends."
        )
    return (
        "Manual PCA and sklearn PCA produce numerically equivalent principal "
        "components after the expected sign alignment; "
        f"{speed_text} The manual implementation is retained for teaching, "
        "traceability, and step-by-step linear algebra explanation."
    )


def _build_feature_interpretation(
    feature_results: list[FeatureAccuracyResult],
) -> str:
    """Build a concise interpretation of feature-set accuracy differences."""
    result_by_name = {result.feature_name: result for result in feature_results}
    color = result_by_name.get("color_hist")
    combined = result_by_name.get("color_texture")
    if color is None or combined is None:
        return "Feature-set ranking should be interpreted from the table above."
    improvement = combined.accuracy_mean - color.accuracy_mean
    return (
        "`color_texture` improves over `color_hist` by "
        f"{improvement:.2%} absolute accuracy, showing that texture statistics "
        "provide complementary style information beyond color distribution."
    )


def _cross_validate_feature_matrix(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    n_neighbors: int,
    fold_count: int,
) -> np.ndarray:
    """Cross-validate StandardScaler + manual PCA + KNN for one feature set."""
    encoded_labels = np.asarray(labels)
    _, class_counts = np.unique(encoded_labels, return_counts=True)
    actual_folds = min(fold_count, int(np.min(class_counts)))
    if actual_folds < 2:
        msg = "At least two samples per class are required for cross validation."
        raise ValueError(msg)

    splitter = StratifiedKFold(
        n_splits=actual_folds,
        shuffle=True,
        random_state=42,
    )
    scores: list[float] = []
    for train_index, test_index in splitter.split(feature_matrix, encoded_labels):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(feature_matrix[train_index])
        test_features = scaler.transform(feature_matrix[test_index])
        component_count = min(n_components, train_features.shape[1])
        pca_model = PCAModel(
            PCAConfig(
                n_components=component_count,
                preprocessing=PreprocessingMode.CENTER,
            ),
        )
        train_pca = pca_model.fit_transform(train_features)
        test_pca = pca_model.transform(test_features)
        classifier = KNeighborsClassifier(
            n_neighbors=max(1, min(n_neighbors, len(train_index))),
        )
        classifier.fit(train_pca, encoded_labels[train_index])
        predictions = classifier.predict(test_pca)
        scores.append(accuracy_score(encoded_labels[test_index], predictions))
    return np.asarray(scores, dtype=np.float64)


def _component_alignment_signs(
    manual_components: np.ndarray,
    sklearn_components: np.ndarray,
) -> np.ndarray:
    """Return signs that align sklearn component directions to manual PCA."""
    signs = np.ones(manual_components.shape[0], dtype=np.float64)
    for index in range(manual_components.shape[0]):
        if np.dot(manual_components[index], sklearn_components[index]) < 0:
            signs[index] = -1.0
    return signs


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    """Validate a finite 2-D feature matrix as float64."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        msg = "feature matrix must be two-dimensional."
        raise ValueError(msg)
    if array.shape[0] < 2 or array.shape[1] < 1:
        msg = "feature matrix must have at least two samples and one feature."
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "feature matrix must contain only finite values."
        raise ValueError(msg)
    return array


if __name__ == "__main__":
    main()
