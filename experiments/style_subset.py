"""Create reproducible style subsets from WikiArt-style folder datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

DEFAULT_STYLES = (
    "impressionism",
    "baroque",
    "cubism",
    "realism",
    "renaissance",
    "romanticism",
)

STYLE_ALIASES = {
    "renaissance": {
        "renaissance",
        "early_renaissance",
        "high_renaissance",
        "northern_renaissance",
        "mannerism_late_renaissance",
    },
}


def create_style_subset(
    source_root: Path,
    output_root: Path,
    styles: tuple[str, ...] = DEFAULT_STYLES,
) -> dict[str, int]:
    """Copy selected first-level style folders into a managed subset."""
    source = source_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    if not source.exists() or not source.is_dir():
        msg = f"source_root must be an existing directory: {source}"
        raise FileNotFoundError(msg)

    normalized_targets = _target_names(styles)
    source = _select_source_root(source, normalized_targets)
    copied_counts: dict[str, int] = {}
    output.mkdir(parents=True, exist_ok=True)
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        normalized = _normalize_style_name(class_dir.name)
        if normalized not in normalized_targets:
            continue
        destination = output / class_dir.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(class_dir, destination)
        copied_counts[class_dir.name] = sum(
            1
            for path in destination.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    return copied_counts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a selected-style dataset subset.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--styles", nargs="+", default=list(DEFAULT_STYLES))
    return parser.parse_args()


def main() -> None:
    """Run the style-subset command."""
    args = parse_args()
    counts = create_style_subset(
        args.source_root,
        args.output_root,
        tuple(args.styles),
    )
    print(f"Subset root: {args.output_root.expanduser().resolve()}")
    for style_name, count in counts.items():
        print(f"{style_name}: {count}")
    if not counts:
        raise SystemExit("No matching style folders were found.")


def _target_names(styles: tuple[str, ...]) -> set[str]:
    targets: set[str] = set()
    for style in styles:
        normalized = _normalize_style_name(style)
        targets.add(normalized)
        targets.update(STYLE_ALIASES.get(normalized, set()))
    return targets


def _select_source_root(source: Path, targets: set[str]) -> Path:
    """Return the directory containing the requested style folders."""
    if any(_normalize_style_name(path.name) in targets for path in source.iterdir() if path.is_dir()):
        return source
    child_dirs = [path for path in source.iterdir() if path.is_dir()]
    if len(child_dirs) == 1:
        child = child_dirs[0]
        if any(_normalize_style_name(path.name) in targets for path in child.iterdir() if path.is_dir()):
            return child
    return source


def _normalize_style_name(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in normalized.split("_") if part)


if __name__ == "__main__":
    main()
