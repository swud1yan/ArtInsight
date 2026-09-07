"""Smoke tests for the interactive PCA style map chart."""

import os
from unittest import TestCase

from PySide6.QtWidgets import QApplication

from controllers.gui_dtos import PCAMapPoint
from views.style_map_chart import PCAStyleMapChart
from views.style_map_state import map_point_to_scatter_point
from views.style_map_widgets import StyleMapSidebar

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class PCAStyleMapChartTests(TestCase):
    """Verify map chart state transitions render without errors."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a shared Qt application for widget smoke tests."""
        cls._app = QApplication.instance() or QApplication([])

    def test_filter_highlight_and_3d_rendering(self) -> None:
        """The chart should render 2-D, filtered, highlighted, and 3-D states."""
        points = _sample_points()
        chart = PCAStyleMapChart()

        chart.draw_pca_map(points)
        chart.set_hidden_labels({"B"})
        chart.highlight_point(map_point_to_scatter_point(points[0]))
        chart.clear_highlight()
        chart.draw_pca_map_3d(points)

        self.assertEqual(len(chart.figure.axes), 1)

    def test_sidebar_displays_translated_labels_but_returns_raw_style(self) -> None:
        """The style filter should show bilingual labels and keep raw filter data."""
        sidebar = StyleMapSidebar()

        sidebar.set_styles(["Baroque", "Impressionism"])
        index = sidebar._style_filter.findData("Baroque")
        sidebar._style_filter.setCurrentIndex(index)

        self.assertIn("巴洛克", sidebar._style_filter.currentText())
        self.assertEqual(sidebar.selected_style(), "Baroque")


def _sample_points() -> list[PCAMapPoint]:
    """Return deterministic points for chart smoke tests."""
    return [
        PCAMapPoint(1, 0.1, 0.2, 0.3, "A", "a.jpg", "Alpha", "Artist A"),
        PCAMapPoint(2, 1.1, 1.2, 1.3, "B", "b.jpg", "Beta", "Artist B"),
    ]
