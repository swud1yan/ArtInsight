"""Tests for the standalone PCA algorithm laboratory page."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from services.ml.pca_algorithm_demo import PCAAlgorithmDemoBuilder
from views.pages.algorithm_lab_page import AlgorithmLabPage


class DummyController:
    """Minimal controller exposing PCA demo steps for the page test."""

    def get_pca_algorithm_demo_steps(self) -> object:
        """Return deterministic PCA demo steps."""
        return PCAAlgorithmDemoBuilder().build_demo_steps()


class AlgorithmLabPageTestCase(unittest.TestCase):
    """Validate standalone algorithm lab page navigation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a shared Qt application for widget tests."""
        cls._app = QApplication.instance() or QApplication([])

    def test_page_loads_demo_steps_and_navigates(self) -> None:
        """The page should display deterministic demo steps and next button."""
        page = AlgorithmLabPage(DummyController())  # type: ignore[arg-type]

        page._next_button.click()

        self.assertEqual(len(page._steps), 5)
        self.assertTrue(page._previous_button.isEnabled())
        self.assertTrue(page._next_button.isEnabled())

    def test_demo_builder_outputs_projection_step(self) -> None:
        """The demo builder should include PCA projection teaching content."""
        steps = PCAAlgorithmDemoBuilder().build_demo_steps()

        self.assertEqual(len(steps), 5)
        self.assertIn("投影", steps[-1].title)
        self.assertEqual(steps[0].points.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
