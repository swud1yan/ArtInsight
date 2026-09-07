"""Tests for the modern Home dashboard page."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.navigation import NavigationRoute
from controllers.gui_dtos import AppSummary, PCAMapPoint
from views.pages.home_page import HomeDashboardData, HomePage


class DummyDashboardController:
    """Minimal controller for dashboard page tests."""

    def __init__(
        self,
        summary: AppSummary,
        map_points: list[PCAMapPoint] | None = None,
    ) -> None:
        """Create a dummy controller with deterministic responses."""
        self._summary = summary
        self._map_points = map_points or []

    def get_summary(self) -> AppSummary:
        """Return dashboard summary data."""
        return self._summary

    def get_pca_map_points(self) -> list[PCAMapPoint]:
        """Return PCA map points."""
        return self._map_points


class HomeDashboardPageTestCase(unittest.TestCase):
    """Validate dashboard states and navigation signals."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a shared Qt application."""
        cls._app = QApplication.instance() or QApplication([])

    def test_empty_dashboard_guides_user_to_import_dataset(self) -> None:
        """Empty state should guide users to the dataset page."""
        page = HomePage(
            DummyDashboardController(
                AppSummary(
                    dataset_count=0,
                    artwork_count=0,
                    style_count=0,
                    feature_count=0,
                    model_available=False,
                    latest_action="尚未记录实验操作",
                ),
            ),  # type: ignore[arg-type]
        )

        page._handle_success(page._load_dashboard_data())

        self.assertEqual(
            page.map_panel.stack.currentWidget(),
            page.map_panel.empty,
        )
        self.assertIn("WikiArt", page.map_panel.empty._label.text())
        self.assertEqual(page._next_workflow_index(page._controller.get_summary()), 0)

    def test_completed_dashboard_highlights_classification_step(self) -> None:
        """A ready model should suggest the next user-facing inference step."""
        summary = AppSummary(
            dataset_count=1,
            artwork_count=200,
            style_count=5,
            feature_count=200,
            model_available=True,
            latest_action="style_model_trained",
            pca_completed=True,
            pca_component_count=10,
            cumulative_variance_ratio=0.93,
            latest_accuracy=0.82,
        )
        page = HomePage(DummyDashboardController(summary))  # type: ignore[arg-type]

        page._handle_success(HomeDashboardData(summary=summary, map_points=[]))

        self.assertEqual(page._next_workflow_index(summary), 5)

    def test_start_button_emits_dataset_navigation_route(self) -> None:
        """The primary quick-start button should request dataset navigation."""
        page = HomePage(
            DummyDashboardController(
                AppSummary(
                    dataset_count=0,
                    artwork_count=0,
                    style_count=0,
                    feature_count=0,
                    model_available=False,
                    latest_action="尚未记录实验操作",
                ),
            ),  # type: ignore[arg-type]
        )
        emitted: list[str] = []
        page.navigation_requested.connect(emitted.append)

        page.start_button.click()

        self.assertEqual(emitted, [NavigationRoute.DATASET.value])


if __name__ == "__main__":
    unittest.main()
