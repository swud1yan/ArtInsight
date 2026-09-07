"""Create reproducible WikiArt subsets for course demos and paper experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
import random
import shutil
from pathlib import Path

import cv2

from experiments.dataset_compliance import SUPPORTED_IMAGE_SUFFIXES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "raw" / "wikiart"
DEFAULT_COURSE_ROOT = PROJECT_ROOT / "data" / "course_dataset"
DEFAULT_PAPER_ROOT = PROJECT_ROOT / "data" / "paper_dataset"
DEFAULT_OVERVIEW_CSV = PROJECT_ROOT / "data" / "dataset_overview.csv"
DEFAULT_RANDOM_SEED = 42
COURSE_SAMPLES_PER_STYLE = 500
PAPER_SAMPLES_PER_STYLE = 800
MIN_COURSE_IMAGES = 3000
DATA_SOURCE = "WikiArt"
SELECTED_STYLES = [
    "Baroque",
    "Realism",
    "Expressionism",
    "Japanese_Art",
    "Romanticism",
    "Symbolism",
]

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
try:
    cv2.setLogLevel(2)
except AttributeError:
    pass


@dataclass(frozen=True)
class SampledDatasetSpec:
    """Configuration for one reproducible sampled dataset."""

    name: str
    output_root: Path
    samples_per_style: int
    minimum_total_images: int


@dataclass(frozen=True)
class VerificationResult:
    """Validation summary for a sampled dataset."""

    status: str
    dataset_root: Path
    expected_per_style: int
    total_images: int
    class_counts: dict[str, int]
    empty_styles: list[str]
    corrupted_images: list[Path]
    message: str


def scan_dataset(source_root: Path = DEFAULT_SOURCE_ROOT) -> dict[str, int]:
    """Count supported image files for every first-level WikiArt style."""
    root = source_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Source dataset root does not exist: {root}")

    counts: dict[str, int] = {}
    for style_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        counts[style_dir.name] = len(_image_paths(style_dir))
    return counts


def write_dataset_overview(
    counts: dict[str, int],
    output_csv: Path = DEFAULT_OVERVIEW_CSV,
) -> Path:
    """Write per-style image counts to a CSV overview file."""
    output_path = output_csv.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["style", "image_count"])
        for style, count in sorted(counts.items()):
            writer.writerow([style, count])
    return output_path


def generate_sampled_dataset(
    *,
    source_root: Path,
    spec: SampledDatasetSpec,
    selected_styles: list[str] = SELECTED_STYLES,
    seed: int = DEFAULT_RANDOM_SEED,
    progress_interval: int = 100,
) -> VerificationResult:
    """Sample and copy a fixed number of images per style."""
    source = source_root.expanduser().resolve()
    output = spec.output_root.expanduser().resolve()
    overview_counts = scan_dataset(source)
    sampled: dict[str, list[Path]] = {}

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    for style in selected_styles:
        source_style_dir = source / style
        candidates = _image_paths(source_style_dir)
        if len(candidates) < spec.samples_per_style:
            raise ValueError(
                f"Style {style} has only {len(candidates)} images; "
                f"{spec.samples_per_style} required."
            )
        rng = random.Random(f"{seed}:{style}:{spec.name}")
        selected = sorted(rng.sample(candidates, spec.samples_per_style))
        sampled[style] = selected
        print(
            f"[{spec.name}] sampling {style}: original={len(candidates)} "
            f"selected={len(selected)}",
            flush=True,
        )
        _copy_style_images(
            selected,
            output / style,
            dataset_name=spec.name,
            style=style,
            progress_interval=progress_interval,
        )

    manifest_path = _write_manifest(
        output,
        sampled,
        source_root=source,
        seed=seed,
        dataset_name=spec.name,
    )
    result = verify_sampled_dataset(
        output,
        selected_styles=selected_styles,
        expected_per_style=spec.samples_per_style,
        minimum_total_images=spec.minimum_total_images,
        progress_interval=progress_interval,
    )
    _write_sample_report(
        output / "sample_report.txt",
        dataset_name=spec.name,
        source_root=source,
        seed=seed,
        selected_styles=selected_styles,
        original_counts=overview_counts,
        sampled_counts={style: len(paths) for style, paths in sampled.items()},
        verification=result,
        manifest_path=manifest_path,
    )
    return result


def verify_sampled_dataset(
    dataset_root: Path | str,
    *,
    selected_styles: list[str] = SELECTED_STYLES,
    expected_per_style: int,
    minimum_total_images: int,
    progress_interval: int = 0,
) -> VerificationResult:
    """Validate class counts, empty folders, corrupted images, and totals."""
    root = Path(dataset_root).expanduser().resolve()
    class_counts: dict[str, int] = {}
    empty_styles: list[str] = []
    corrupted_images: list[Path] = []

    checked_count = 0
    for style in selected_styles:
        style_dir = root / style
        paths = _image_paths(style_dir) if style_dir.exists() else []
        class_counts[style] = len(paths)
        if not paths:
            empty_styles.append(style)
        for image_path in paths:
            if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
                corrupted_images.append(image_path)
            checked_count += 1
            if progress_interval > 0 and checked_count % progress_interval == 0:
                print(
                    f"[verify] {root.name}: checked={checked_count} "
                    f"corrupted={len(corrupted_images)}",
                    flush=True,
                )

    total_images = sum(class_counts.values())
    count_mismatches = {
        style: count
        for style, count in class_counts.items()
        if count != expected_per_style
    }
    errors: list[str] = []
    if count_mismatches:
        errors.append(f"count mismatches: {count_mismatches}")
    if empty_styles:
        errors.append(f"empty styles: {empty_styles}")
    if corrupted_images:
        errors.append(f"corrupted images: {len(corrupted_images)}")
    if total_images < minimum_total_images:
        errors.append(
            f"total images below requirement: {total_images} < {minimum_total_images}"
        )

    status = "PASS" if not errors else "FAIL"
    message = "Sampled dataset validation passed." if not errors else "; ".join(errors)
    return VerificationResult(
        status=status,
        dataset_root=root,
        expected_per_style=expected_per_style,
        total_images=total_images,
        class_counts=class_counts,
        empty_styles=empty_styles,
        corrupted_images=corrupted_images,
        message=message,
    )


def build_default_datasets(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    course_root: Path = DEFAULT_COURSE_ROOT,
    paper_root: Path = DEFAULT_PAPER_ROOT,
    overview_csv: Path = DEFAULT_OVERVIEW_CSV,
    seed: int = DEFAULT_RANDOM_SEED,
    progress_interval: int = 100,
) -> tuple[VerificationResult, VerificationResult]:
    """Create both course and paper datasets from the downloaded WikiArt data."""
    counts = scan_dataset(source_root)
    write_dataset_overview(counts, overview_csv)

    course_result = generate_sampled_dataset(
        source_root=source_root,
        spec=SampledDatasetSpec(
            name="course_dataset",
            output_root=course_root,
            samples_per_style=COURSE_SAMPLES_PER_STYLE,
            minimum_total_images=MIN_COURSE_IMAGES,
        ),
        seed=seed,
        progress_interval=progress_interval,
    )
    paper_result = generate_sampled_dataset(
        source_root=source_root,
        spec=SampledDatasetSpec(
            name="paper_dataset",
            output_root=paper_root,
            samples_per_style=PAPER_SAMPLES_PER_STYLE,
            minimum_total_images=PAPER_SAMPLES_PER_STYLE * len(SELECTED_STYLES),
        ),
        seed=seed,
        progress_interval=progress_interval,
    )
    return course_result, paper_result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for reproducible dataset sampling."""
    parser = argparse.ArgumentParser(
        description="Generate course and paper WikiArt subsets.",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--course-root", type=Path, default=DEFAULT_COURSE_ROOT)
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_PAPER_ROOT)
    parser.add_argument("--overview-csv", type=Path, default=DEFAULT_OVERVIEW_CSV)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print copy/verification progress every N images; use 0 to disable.",
    )
    args = parser.parse_args(argv)

    course_result, paper_result = build_default_datasets(
        source_root=args.source_root,
        course_root=args.course_root,
        paper_root=args.paper_root,
        overview_csv=args.overview_csv,
        seed=args.seed,
        progress_interval=args.progress_interval,
    )
    _print_final_summary(course_result, paper_result, args.seed)
    return 0 if course_result.status == "PASS" and paper_result.status == "PASS" else 2


def _image_paths(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def _copy_style_images(
    image_paths: list[Path],
    output_style_dir: Path,
    *,
    dataset_name: str,
    style: str,
    progress_interval: int,
) -> None:
    output_style_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    for index, image_path in enumerate(image_paths, start=1):
        target_name = _unique_target_name(image_path, used_names)
        shutil.copy2(image_path, output_style_dir / target_name)
        if progress_interval > 0 and index % progress_interval == 0:
            print(
                f"[{dataset_name}] copied {style}: {index}/{len(image_paths)}",
                flush=True,
            )


def _unique_target_name(image_path: Path, used_names: set[str]) -> str:
    candidate = image_path.name
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()
    candidate = f"{image_path.stem}_{digest[:10]}{image_path.suffix.lower()}"
    used_names.add(candidate)
    return candidate


def _write_manifest(
    output_root: Path,
    sampled: dict[str, list[Path]],
    *,
    source_root: Path,
    seed: int,
    dataset_name: str,
) -> Path:
    manifest_path = output_root / "sample_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "seed", "style", "source_relative_path"])
        for style, paths in sorted(sampled.items()):
            for path in paths:
                writer.writerow([dataset_name, seed, style, path.relative_to(source_root)])
    return manifest_path


def _write_sample_report(
    report_path: Path,
    *,
    dataset_name: str,
    source_root: Path,
    seed: int,
    selected_styles: list[str],
    original_counts: dict[str, int],
    sampled_counts: dict[str, int],
    verification: VerificationResult,
    manifest_path: Path,
) -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        f"Dataset: {dataset_name}",
        f"Sampling time: {now}",
        f"Random seed: {seed}",
        f"Data source: {DATA_SOURCE}",
        f"Source root: {source_root}",
        f"Output root: {verification.dataset_root}",
        f"Manifest: {manifest_path}",
        "",
        "Selected styles:",
    ]
    for style in selected_styles:
        lines.append(
            f"- {style}: original={original_counts.get(style, 0)}, "
            f"sampled={sampled_counts.get(style, 0)}"
        )
    lines.extend(
        [
            "",
            f"Total images: {verification.total_images}",
            f"Verification status: {verification.status}",
            f"Verification message: {verification.message}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_final_summary(
    course_result: VerificationResult,
    paper_result: VerificationResult,
    seed: int,
) -> None:
    print("Course dataset:")
    print(f"  path: {course_result.dataset_root}")
    print(f"  images: {course_result.total_images}")
    print(f"  status: {course_result.status}")
    print("Paper dataset:")
    print(f"  path: {paper_result.dataset_root}")
    print(f"  images: {paper_result.total_images}")
    print(f"  status: {paper_result.status}")
    print(f"Random seed: {seed}")
    print(f"Styles: {len(SELECTED_STYLES)}")
    print("Selected styles: " + ", ".join(SELECTED_STYLES))
    overall = "PASS" if course_result.status == "PASS" and paper_result.status == "PASS" else "FAIL"
    print(f"Validation status: {overall}")


if __name__ == "__main__":
    raise SystemExit(main())
