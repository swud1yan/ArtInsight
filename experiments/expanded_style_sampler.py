"""Build an expanded multi-style art dataset from one or more folder sources."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import random
import shutil
from pathlib import Path

from experiments.dataset_compliance import SUPPORTED_IMAGE_SUFFIXES
from experiments.openverse_style_backfill import backfill_public_styles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOTS = [
    PROJECT_ROOT / "data" / "raw" / "wikiart",
    PROJECT_ROOT / "data" / "raw" / "wikiart_full",
]
DEFAULT_PUBLIC_BACKFILL_ROOT = PROJECT_ROOT / "data" / "raw" / "public_style_backfill"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "expanded_50_style_dataset"
DEFAULT_COVERAGE_CSV = PROJECT_ROOT / "data" / "expanded_style_coverage.csv"
DEFAULT_REPORT = DEFAULT_OUTPUT_ROOT / "sample_report.txt"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "sample_manifest.csv"
DEFAULT_SEED = 42
DEFAULT_TARGET_PER_STYLE = 800

EXPANDED_STYLES = [
    "Early Renaissance",
    "High Renaissance",
    "Northern Renaissance",
    "Renaissance",
    "Mannerism",
    "Proto Renaissance",
    "Baroque",
    "Rococo",
    "Neoclassicism",
    "Academic Art",
    "Romanticism",
    "Realism",
    "Symbolism",
    "Orientalism",
    "Impressionism",
    "Post-Impressionism",
    "Pointillism",
    "Divisionism",
    "Tonalism",
    "Naturalism",
    "Social Realism",
    "Pre-Raphaelite",
    "Expressionism",
    "Fauvism",
    "Cubism",
    "Analytical Cubism",
    "Synthetic Cubism",
    "Futurism",
    "Constructivism",
    "Suprematism",
    "Dada",
    "Surrealism",
    "Pop Art",
    "Minimalism",
    "Op Art",
    "Art Nouveau",
    "Art Deco",
    "Abstract Art",
    "Abstract Expressionism",
    "Color Field Painting",
    "Lyrical Abstraction",
    "Action Painting",
    "Hard Edge Painting",
    "Tachisme",
    "Art Informel",
    "Ukiyo-e",
    "Japanese Art",
    "Chinese Ink Painting",
    "Photorealism",
    "Contemporary Realism",
]

STYLE_ALIASES = {
    "Academic Art": ["Academic_Art", "Academic art"],
    "Art Deco": ["Art_Deco"],
    "Art Informel": ["Art_Informel", "Informalism"],
    "Art Nouveau": ["Art_Nouveau", "Art_Nouveau_Modern"],
    "Chinese Ink Painting": ["Chinese_Ink_Painting", "Ink Painting", "Ink_Painting"],
    "Color Field Painting": ["Color_Field_Painting"],
    "Early Renaissance": ["Early_Renaissance"],
    "High Renaissance": ["High_Renaissance"],
    "Japanese Art": ["Japanese_Art"],
    "Mannerism": ["Mannerism_Late_Renaissance"],
    "Northern Renaissance": ["Northern_Renaissance"],
    "Op Art": ["Op_Art"],
    "Pop Art": ["Pop_Art"],
    "Post-Impressionism": ["Post_Impressionism", "Post-Impressionism"],
    "Pre-Raphaelite": ["Pre_Raphaelite", "Pre-Raphaelites"],
    "Proto Renaissance": ["Proto_Renaissance", "Proto-Renaissance"],
    "Social Realism": ["Social_Realism"],
    "Synthetic Cubism": ["Synthetic_Cubism"],
    "Analytical Cubism": ["Analytical_Cubism", "Analytic_Cubism"],
    "Abstract Art": ["Abstract_Art", "Abstract"],
    "Abstract Expressionism": ["Abstract_Expressionism"],
    "Action Painting": ["Action_Painting", "Action_painting"],
    "Hard Edge Painting": ["Hard_Edge_Painting"],
    "Contemporary Realism": ["Contemporary_Realism"],
    "Ukiyo-e": ["Ukiyo_e"],
}


@dataclass(frozen=True)
class CoverageRow:
    """Coverage summary for one requested style."""

    requested_style: str
    matched_folders: list[str]
    available_images: int
    sampled_images: int
    status: str


def build_expanded_dataset(
    *,
    source_roots: list[Path],
    output_root: Path,
    target_per_style: int,
    seed: int,
    allow_partial: bool,
    coverage_csv: Path,
    progress_interval: int,
    backfill_missing: bool,
    backfill_root: Path,
    backfill_max_images_per_style: int,
) -> list[CoverageRow]:
    """Scan sources, sample requested styles, and write reports."""
    allow_partial = allow_partial or backfill_missing
    sources = [path.expanduser().resolve() for path in source_roots]
    existing_sources = [path for path in sources if path.exists() and path.is_dir()]
    if not existing_sources:
        raise FileNotFoundError(
            "No source roots exist. Provide a downloaded dataset such as "
            "data/raw/wikiart_full."
        )

    style_index = _build_style_index(existing_sources)
    if backfill_missing:
        missing_styles = _missing_styles(style_index, target_per_style)
        if missing_styles:
            print(
                "Backfilling missing styles from Wikimedia Commons: "
                + ", ".join(missing_styles),
                flush=True,
            )
            backfill_public_styles(
                missing_styles,
                output_root=backfill_root,
                max_images_per_style=backfill_max_images_per_style,
                seed=seed,
                progress_interval=5,
                overwrite=False,
            )
            if backfill_root.expanduser().resolve() not in existing_sources:
                existing_sources.append(backfill_root.expanduser().resolve())
            style_index = _build_style_index(existing_sources)

    coverage_rows: list[CoverageRow] = []
    sampled: dict[str, list[Path]] = {}

    output = output_root.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    for style in EXPANDED_STYLES:
        paths_by_folder = _matching_paths(style_index, style)
        all_paths = _dedupe_paths(
            image_path for paths in paths_by_folder.values() for image_path in paths
        )
        status = "OK" if len(all_paths) >= target_per_style else "INSUFFICIENT"
        sample_count = min(len(all_paths), target_per_style) if allow_partial else target_per_style
        if status == "INSUFFICIENT" and not allow_partial:
            sampled_paths: list[Path] = []
        else:
            rng = random.Random(f"{seed}:{style}:expanded")
            sampled_paths = sorted(rng.sample(all_paths, sample_count))
            _copy_style_images(
                sampled_paths,
                output / _safe_style_dir(style),
                style=style,
                progress_interval=progress_interval,
            )
        sampled[style] = sampled_paths
        coverage_rows.append(
            CoverageRow(
                requested_style=style,
                matched_folders=sorted(paths_by_folder),
                available_images=len(all_paths),
                sampled_images=len(sampled_paths),
                status=status,
            )
        )
        print(
            f"{style}: available={len(all_paths)} sampled={len(sampled_paths)} "
            f"status={status}",
            flush=True,
        )

    _write_coverage_csv(coverage_rows, coverage_csv)
    _write_manifest(sampled, output / "sample_manifest.csv", existing_sources, seed)
    _write_report(
        output / "sample_report.txt",
        coverage_rows,
        source_roots=existing_sources,
        output_root=output,
        target_per_style=target_per_style,
        seed=seed,
        allow_partial=allow_partial,
    )
    return coverage_rows


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build an expanded 50-style art dataset from local sources.",
    )
    parser.add_argument(
        "--source-roots",
        type=Path,
        nargs="+",
        default=DEFAULT_SOURCE_ROOTS,
        help="One or more folder-style datasets to scan.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--coverage-csv", type=Path, default=DEFAULT_COVERAGE_CSV)
    parser.add_argument("--target-per-style", type=int, default=DEFAULT_TARGET_PER_STYLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--backfill-public",
        action="store_true",
        help="Backfill missing styles from Wikimedia Commons before sampling.",
    )
    parser.add_argument(
        "--backfill-root",
        type=Path,
        default=DEFAULT_PUBLIC_BACKFILL_ROOT,
        help="Output root for public-source backfill images.",
    )
    parser.add_argument(
        "--backfill-max-images-per-style",
        type=int,
        default=40,
        help="Maximum public images to backfill per missing style.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Copy all available images for styles below the target instead of skipping them.",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    """Run the expanded sampler."""
    args = parse_args()
    rows = build_expanded_dataset(
        source_roots=list(args.source_roots),
        output_root=args.output_root,
        target_per_style=args.target_per_style,
        seed=args.seed,
        allow_partial=args.allow_partial,
        coverage_csv=args.coverage_csv,
        progress_interval=args.progress_interval,
        backfill_missing=args.backfill_public,
        backfill_root=args.backfill_root,
        backfill_max_images_per_style=args.backfill_max_images_per_style,
    )
    ok_count = sum(1 for row in rows if row.status == "OK")
    insufficient_count = len(rows) - ok_count
    total_sampled = sum(row.sampled_images for row in rows)
    print(f"Styles requested: {len(rows)}")
    print(f"Styles OK: {ok_count}")
    print(f"Styles insufficient: {insufficient_count}")
    print(f"Total sampled: {total_sampled}")
    print(f"Coverage CSV: {args.coverage_csv.expanduser().resolve()}")
    print(f"Output root: {args.output_root.expanduser().resolve()}")
    return 0 if insufficient_count == 0 else 2


def _build_style_index(source_roots: list[Path]) -> dict[str, dict[str, list[Path]]]:
    """Index images by normalized first-level folder name."""
    index: dict[str, dict[str, list[Path]]] = {}
    for root in source_roots:
        source = _select_folder_dataset_root(root)
        for folder in sorted(path for path in source.iterdir() if path.is_dir()):
            images = _image_paths(folder)
            if not images:
                continue
            normalized = _normalize_style(folder.name)
            index.setdefault(normalized, {})[str(folder)] = images
    return index


def _missing_styles(
    style_index: dict[str, dict[str, list[Path]]],
    target_per_style: int,
) -> list[str]:
    """Return styles that do not yet have enough images for a balanced sample."""
    missing: list[str] = []
    for style in EXPANDED_STYLES:
        paths_by_folder = _matching_paths(style_index, style)
        all_paths = _dedupe_paths(
            image_path for paths in paths_by_folder.values() for image_path in paths
        )
        if len(all_paths) < target_per_style:
            missing.append(style)
    return missing


def _select_folder_dataset_root(root: Path) -> Path:
    """Handle archives that unzip into one nested dataset directory."""
    if any(path.is_dir() and _image_paths(path) for path in root.iterdir()):
        return root
    child_dirs = [path for path in root.iterdir() if path.is_dir()]
    if len(child_dirs) == 1:
        child = child_dirs[0]
        if any(path.is_dir() and _image_paths(path) for path in child.iterdir()):
            return child
    return root


def _matching_paths(
    style_index: dict[str, dict[str, list[Path]]],
    requested_style: str,
) -> dict[str, list[Path]]:
    matches: dict[str, list[Path]] = {}
    for alias in _style_aliases(requested_style):
        normalized = _normalize_style(alias)
        for folder, paths in style_index.get(normalized, {}).items():
            matches[folder] = paths
    return matches


def _style_aliases(style: str) -> list[str]:
    return [style, style.replace(" ", "_"), *STYLE_ALIASES.get(style, [])]


def _image_paths(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def _dedupe_paths(paths: object) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = str(Path(path).resolve()).lower()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(Path(path))
    return sorted(unique)


def _copy_style_images(
    image_paths: list[Path],
    output_style_dir: Path,
    *,
    style: str,
    progress_interval: int,
) -> None:
    output_style_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    for index, image_path in enumerate(image_paths, start=1):
        target_name = _unique_target_name(image_path, used_names)
        shutil.copy2(image_path, output_style_dir / target_name)
        if progress_interval > 0 and index % progress_interval == 0:
            print(f"copied {style}: {index}/{len(image_paths)}", flush=True)


def _unique_target_name(image_path: Path, used_names: set[str]) -> str:
    candidate = image_path.name
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()
    candidate = f"{image_path.stem}_{digest[:10]}{image_path.suffix.lower()}"
    used_names.add(candidate)
    return candidate


def _safe_style_dir(style: str) -> str:
    return style.replace(" ", "_").replace("-", "_")


def _normalize_style(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in normalized.split("_") if part)


def _write_coverage_csv(rows: list[CoverageRow], output_csv: Path) -> None:
    output_path = output_csv.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "requested_style",
                "available_images",
                "sampled_images",
                "status",
                "matched_folders",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.requested_style,
                    row.available_images,
                    row.sampled_images,
                    row.status,
                    " | ".join(row.matched_folders),
                ]
            )


def _write_manifest(
    sampled: dict[str, list[Path]],
    manifest_path: Path,
    source_roots: list[Path],
    seed: int,
) -> None:
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "style", "source_path"])
        for style, paths in sorted(sampled.items()):
            for path in paths:
                writer.writerow([seed, style, _display_source_path(path, source_roots)])


def _display_source_path(path: Path, source_roots: list[Path]) -> str:
    for root in source_roots:
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def _write_report(
    report_path: Path,
    rows: list[CoverageRow],
    *,
    source_roots: list[Path],
    output_root: Path,
    target_per_style: int,
    seed: int,
    allow_partial: bool,
) -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    ok_rows = [row for row in rows if row.status == "OK"]
    insufficient_rows = [row for row in rows if row.status != "OK"]
    lines = [
        "Expanded 50-style dataset report",
        f"Sampling time: {now}",
        f"Random seed: {seed}",
        f"Target per style: {target_per_style}",
        f"Allow partial: {allow_partial}",
        f"Output root: {output_root}",
        "Source roots:",
        *[f"- {root}" for root in source_roots],
        "",
        f"Styles requested: {len(rows)}",
        f"Styles OK: {len(ok_rows)}",
        f"Styles insufficient: {len(insufficient_rows)}",
        f"Total sampled: {sum(row.sampled_images for row in rows)}",
        "",
        "Per-style coverage:",
    ]
    for row in rows:
        lines.append(
            f"- {row.requested_style}: available={row.available_images}, "
            f"sampled={row.sampled_images}, status={row.status}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
