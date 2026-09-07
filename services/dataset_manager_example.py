"""Example usage for dataset import and statistics.

Run with:
    python -m services.dataset_manager_example D:/path/to/WikiArt
"""

from __future__ import annotations

import argparse
import logging

from services.dataset_manager import DatasetManager

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Import a dataset and log basic statistics."""
    parser = argparse.ArgumentParser(description="Import an ArtInsight dataset folder.")
    parser.add_argument("dataset_path", help="Path to a dataset root such as WikiArt.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    manager = DatasetManager.from_config()
    result = manager.import_dataset(args.dataset_path)
    statistics = manager.get_statistics(result.dataset_id)
    LOGGER.info("Imported dataset: %s", result)
    LOGGER.info("Dataset statistics: %s", statistics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())