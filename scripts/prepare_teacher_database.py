"""Prepare a deployment SQLite database for the teacher-facing server.

The source development database may contain paths such as ``D:\\PCA\\data\\...``
and records for datasets that will not be uploaded. This script copies the
database, keeps the selected 50-style datasets, rewrites image paths for the
server container, and vacuums the result.
"""

from __future__ import annotations

import argparse
import posixpath
import shutil
import sqlite3
from pathlib import Path


DEFAULT_DATASETS = ("demo_1000_dataset", "expanded_50_style_dataset")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare ArtInsight SQLite database for Aliyun deployment.",
    )
    parser.add_argument(
        "--source",
        default="data/artinsight.sqlite3",
        help="Source SQLite database. Defaults to data/artinsight.sqlite3.",
    )
    parser.add_argument(
        "--output",
        default="data/artinsight_teacher.sqlite3",
        help="Output SQLite database. Defaults to data/artinsight_teacher.sqlite3.",
    )
    parser.add_argument(
        "--server-root",
        default="/opt/artinsight",
        help="Project root inside the server/container. Defaults to /opt/artinsight.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset names to keep and enable.",
    )
    parser.add_argument(
        "--keep-other-datasets",
        action="store_true",
        help="Only disable unselected datasets instead of deleting their records.",
    )
    return parser.parse_args()


def main() -> int:
    """Copy and rewrite a deployment database."""
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source database does not exist: {source}")
    if source == output:
        raise ValueError("Refusing to modify the source database in place.")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)

    dataset_names = tuple(dict.fromkeys(args.datasets))
    with sqlite3.connect(output) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        selected_ids = load_selected_dataset_ids(connection, dataset_names)
        if not selected_ids:
            raise ValueError(f"No selected datasets were found: {dataset_names}")

        if args.keep_other_datasets:
            disable_unselected_datasets(connection, selected_ids)
        else:
            prune_unselected_datasets(connection, selected_ids)

        rewrite_paths(connection, dataset_names, args.server_root.rstrip("/"))
        connection.commit()

    with sqlite3.connect(output) as connection:
        connection.execute("VACUUM")

    print(f"Prepared deployment database: {output}")
    return 0


def load_selected_dataset_ids(
    connection: sqlite3.Connection,
    dataset_names: tuple[str, ...],
) -> tuple[int, ...]:
    """Return IDs for the requested dataset names."""
    placeholders = ",".join("?" for _ in dataset_names)
    rows = connection.execute(
        f"SELECT id FROM datasets WHERE name IN ({placeholders})",
        dataset_names,
    ).fetchall()
    return tuple(int(row[0]) for row in rows)


def disable_unselected_datasets(
    connection: sqlite3.Connection,
    selected_ids: tuple[int, ...],
) -> None:
    """Enable selected datasets and disable every other dataset."""
    placeholders = ",".join("?" for _ in selected_ids)
    connection.execute("UPDATE datasets SET is_enabled = 0")
    connection.execute(
        f"UPDATE datasets SET is_enabled = 1 WHERE id IN ({placeholders})",
        selected_ids,
    )


def prune_unselected_datasets(
    connection: sqlite3.Connection,
    selected_ids: tuple[int, ...],
) -> None:
    """Remove datasets and dependent rows that will not be uploaded."""
    placeholders = ",".join("?" for _ in selected_ids)
    connection.execute(
        f"""
        DELETE FROM features
        WHERE artwork_id IN (
            SELECT id FROM artworks WHERE dataset_id NOT IN ({placeholders})
        )
        """,
        selected_ids,
    )
    connection.execute(
        f"""
        DELETE FROM pca_results
        WHERE artwork_id IN (
            SELECT id FROM artworks WHERE dataset_id NOT IN ({placeholders})
        )
        """,
        selected_ids,
    )
    connection.execute(
        f"DELETE FROM artworks WHERE dataset_id NOT IN ({placeholders})",
        selected_ids,
    )
    connection.execute(
        f"DELETE FROM history WHERE dataset_id NOT IN ({placeholders})",
        selected_ids,
    )
    connection.execute(
        f"DELETE FROM datasets WHERE id NOT IN ({placeholders})",
        selected_ids,
    )
    connection.execute("UPDATE datasets SET is_enabled = 1")


def rewrite_paths(
    connection: sqlite3.Connection,
    dataset_names: tuple[str, ...],
    server_root: str,
) -> None:
    """Rewrite dataset and artwork paths to the server/container layout."""
    for dataset_name in dataset_names:
        server_dataset_root = posixpath.join(server_root, "data", dataset_name)
        connection.execute(
            "UPDATE datasets SET path = ? WHERE name = ?",
            (server_dataset_root, dataset_name),
        )
        rows = connection.execute(
            """
            SELECT artworks.id, artworks.image_path
            FROM artworks
            JOIN datasets ON artworks.dataset_id = datasets.id
            WHERE datasets.name = ?
            """,
            (dataset_name,),
        ).fetchall()
        for artwork_id, image_path in rows:
            relative_path = path_after_dataset_name(str(image_path), dataset_name)
            new_path = posixpath.join(server_dataset_root, relative_path)
            connection.execute(
                "UPDATE artworks SET image_path = ? WHERE id = ?",
                (new_path, int(artwork_id)),
            )


def path_after_dataset_name(image_path: str, dataset_name: str) -> str:
    """Return the artwork path relative to its dataset directory."""
    normalized = image_path.replace("\\", "/")
    marker = f"/{dataset_name}/"
    marker_index = normalized.rfind(marker)
    if marker_index >= 0:
        return normalized[marker_index + len(marker):]
    if normalized.startswith(f"{dataset_name}/"):
        return normalized[len(dataset_name) + 1:]
    raise ValueError(
        f"Cannot find dataset segment {dataset_name!r} in image path: {image_path}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
