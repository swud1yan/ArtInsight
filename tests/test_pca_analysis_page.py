"""Tests for the PCA analysis page embedded algorithm laboratory."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from controllers.gui_dtos import PCALabStepData
from services.ml.pca_algorithm_demo import PCAAlgorithmDemoBuilder
from views.pages.pca_analysis_page import PCAAnalysisPage


class DummyPCAAnalysisController:
    """Minimal controller exposing teaching steps required at page startup."""

    def get_pca_algorithm_lab_demo_steps(self) -> object:
        """Return deterministic PCA teaching steps for the embedded lab."""
        return [
            PCALabStepData(
                title=step.title,
                description=step.explanation,
                column_headers=list(step.table_headers),
                rows=[list(row) for row in step.table_rows],
                note=step.annotation,
            )
            for step in PCAAlgorithmDemoBuilder().build_demo_steps()
        ]


class PCAAnalysisPageTestCase(unittest.TestCase):
    """Validate PCA analysis startup behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a shared Qt application for widget tests."""
        cls._app = QApplication.instance() or QApplication([])

    def test_embedded_lab_has_default_demo_without_trained_model(self) -> None:
        """The algorithm lab tab should be usable before loading PCA artifacts."""
        page = PCAAnalysisPage(DummyPCAAnalysisController())  # type: ignore[arg-type]

        self.assertGreater(len(page._lab_widget._steps), 0)
        self.assertTrue(page._lab_widget._next_button.isEnabled())
        self.assertIn("原始二维样本", page._lab_widget._title_label.text())


if __name__ == "__main__":
    unittest.main()
