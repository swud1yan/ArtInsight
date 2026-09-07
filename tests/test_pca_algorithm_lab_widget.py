"""Tests for the PCA algorithm laboratory widget."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from controllers.gui_dtos import PCALabStepData
from views.pca_algorithm_lab_widget import PCAAlgorithmLabWidget


class PCAAlgorithmLabWidgetTestCase(unittest.TestCase):
    """Validate that the laboratory widget renders step DTOs."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a shared Qt application for widget tests."""
        cls._app = QApplication.instance() or QApplication([])

    def test_widget_loads_and_navigates_steps(self) -> None:
        """Loading steps should enable next/previous navigation safely."""
        widget = PCAAlgorithmLabWidget()
        steps = [
            PCALabStepData(
                title="Step 1",
                description="First",
                column_headers=["A", "B"],
                rows=[["r1", "1.0"]],
                note="note 1",
            ),
            PCALabStepData(
                title="Step 2",
                description="Second",
                column_headers=["A", "B"],
                rows=[["r2", "2.0"]],
                note="note 2",
            ),
        ]

        widget.set_steps(steps)
        widget._next_button.click()

        self.assertTrue(widget._previous_button.isEnabled())
        self.assertFalse(widget._next_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
