"""Tests for the algorithm experiment GUI page request wiring."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from views.pages.algorithm_experiment_page import AlgorithmExperimentPage


@dataclass(frozen=True)
class _DummyPaths:
    """Minimal path settings used by the page test."""

    export_dir: Path


@dataclass(frozen=True)
class _DummyConfig:
    """Minimal controller config used by the page test."""

    paths: _DummyPaths


class _DummyController:
    """Minimal controller for request-building tests."""

    def __init__(self, export_dir: Path) -> None:
        """Create a dummy controller with only config paths."""
        self.config = _DummyConfig(paths=_DummyPaths(export_dir=export_dir))


class AlgorithmExperimentPageTestCase(unittest.TestCase):
    """Validate GUI request construction for paper exports."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a shared Qt application for widget tests."""
        cls._app = QApplication.instance() or QApplication([])

    def test_request_contains_seed_and_paper_figure_directory(self) -> None:
        """Experiment requests should carry seed and paper-export figure path."""
        export_dir = Path("data/test_exports").resolve()
        page = AlgorithmExperimentPage(_DummyController(export_dir))  # type: ignore[arg-type]

        page._seed_input.setValue(2024)
        request = page._build_request()

        self.assertEqual(request.random_state, 2024)
        self.assertEqual(
            request.output_dir,
            export_dir / "paper_experiments" / "baseline_seed_2024" / "figures",
        )
        self.assertFalse(page._export_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
