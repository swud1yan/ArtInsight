"""Interactive Matplotlib chart for PCA style maps."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from controllers.gui_dtos import PCAMapPoint
from visualization.matplotlib_style import configure_matplotlib_fonts
from visualization.pca_chart_models import HoverAnnotation, PCAScatterPoint
from visualization.pca_charts import (
    draw_empty_chart,
    draw_pca_scatter_2d,
    draw_pca_scatter_3d,
)
from views.ui_helpers import create_button
from views.zoom_dialogs import show_enlarged_figure


configure_matplotlib_fonts()


class PCAStyleMapChart(QWidget):
    """Matplotlib widget specialized for interactive PCA style maps."""

    def __init__(self) -> None:
        """Create an interactive PCA style map chart."""
        super().__init__()
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._dialog_title = "PCA风格地图"
        self._zoom_button = create_button("放大查看", "search")
        self._zoom_button.clicked.connect(self._show_zoom_dialog)
        self._hover_annotation: HoverAnnotation | None = None
        self._base_points: list[PCAScatterPoint] = []
        self._highlight_point: PCAScatterPoint | None = None
        self._hidden_labels: set[str] = set()
        self._map_mode = "2d"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.addWidget(self.toolbar)
        toolbar_row.addStretch(1)
        toolbar_row.addWidget(self._zoom_button)
        layout.addLayout(toolbar_row)
        layout.addWidget(self.canvas)

    def draw_empty(self, title: str, message: str = "暂无数据") -> None:
        """Draw a centered no-data message."""
        self._dialog_title = title
        self._disconnect_hover()
        self.toolbar.hide()
        draw_empty_chart(self.figure, title, message)
        self.canvas.draw_idle()

    def draw_pca_map(self, points: list[PCAMapPoint]) -> None:
        """Draw a 2-D PCA scatter plot with hover annotations."""
        self._base_points = self._to_scatter_points(points)
        self._highlight_point = None
        self._map_mode = "2d"
        self._dialog_title = "PCA二维风格地图"
        self._render_map()

    def draw_pca_map_3d(self, points: list[PCAMapPoint]) -> None:
        """Draw a rotatable 3-D PCA scatter plot."""
        self._base_points = self._to_scatter_points(points)
        self._highlight_point = None
        self._map_mode = "3d"
        self._dialog_title = "PCA三维风格地图"
        self._render_map()

    def set_hidden_labels(self, labels: set[str]) -> None:
        """Hide the specified style labels and refresh the map."""
        self._hidden_labels = set(labels)
        self._render_map()

    def highlight_point(self, point: PCAScatterPoint | None) -> None:
        """Highlight a single uploaded or searched point."""
        self._highlight_point = point
        self._render_map()

    def clear_highlight(self) -> None:
        """Remove any highlighted point."""
        self._highlight_point = None
        self._render_map()

    def _render_map(self) -> None:
        """Render the current map state from cached base points."""
        self._disconnect_hover()
        points = [
            point
            for point in self._base_points
            if point.label not in self._hidden_labels
        ]
        if not points:
            self.toolbar.hide()
            draw_empty_chart(self.figure, "PCA二维风格地图", "暂无可显示的作品点")
            self.canvas.draw_idle()
            return
        self.toolbar.show()
        if self._map_mode == "3d" and all(point.z_value is not None for point in points):
            draw_pca_scatter_3d(self.figure, points)
        else:
            self._hover_annotation = draw_pca_scatter_2d(self.figure, points)
        if self._highlight_point is not None:
            self._draw_highlight(self._highlight_point)
        self.canvas.draw_idle()

    def _show_zoom_dialog(self) -> None:
        """Open the current PCA map in a larger dialog."""
        points = [
            point
            for point in self._base_points
            if point.label not in self._hidden_labels
        ]
        if not points:
            show_enlarged_figure(self, self.figure, self._dialog_title)
            return
        figure = Figure(figsize=(12, 7), tight_layout=True)
        if self._map_mode == "3d" and all(point.z_value is not None for point in points):
            draw_pca_scatter_3d(figure, points, legend_mode="full")
        else:
            draw_pca_scatter_2d(figure, points, interactive=False, legend_mode="full")
        show_enlarged_figure(self, figure, self._dialog_title)

    def _draw_highlight(self, point: PCAScatterPoint) -> None:
        """Overlay a prominent marker for a highlighted point."""
        axis = self.figure.axes[0]
        if self._map_mode == "3d" and point.z_value is not None:
            axis.scatter(
                [point.x_value],
                [point.y_value],
                [point.z_value],
                s=220,
                color="#D62828",
                marker="*",
                edgecolors="white",
                linewidths=1.5,
                label="你的图片",
                zorder=8,
            )
        else:
            axis.scatter(
                [point.x_value],
                [point.y_value],
                s=220,
                color="#D62828",
                marker="*",
                edgecolors="white",
                linewidths=1.5,
                label="你的图片",
                zorder=8,
            )
        axis.legend(loc="best", title="风格")

    def _disconnect_hover(self) -> None:
        """Disconnect an existing hover annotation callback."""
        if self._hover_annotation is None:
            return
        self.canvas.mpl_disconnect(self._hover_annotation.connection_id)
        self._hover_annotation = None

    def _to_scatter_points(self, points: list[PCAMapPoint]) -> list[PCAScatterPoint]:
        """Convert GUI PCA map DTOs to visualization scatter points."""
        return [
            PCAScatterPoint(
                sample_id=str(point.artwork_id),
                label=point.style_label,
                x_value=point.x_value,
                y_value=point.y_value,
                z_value=point.z_value,
                title=point.title,
                artist_name=point.artist_name,
                image_path=point.image_path,
            )
            for point in points
        ]
