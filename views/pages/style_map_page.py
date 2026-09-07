"""Interactive PCA style map page."""

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
)

from app.navigation import NavigationRoute
from controllers.gui_dtos import PCAMapPoint, UploadedMapPoint
from controllers.page_data_controller import PageDataController
from services.style_translations import format_style_label
from views.dashboard_widgets import EmptyActionPanel
from views.pages.base_page import BasePage
from views.ui_helpers import choose_image_file, create_panel
from views.style_map_chart import PCAStyleMapChart
from views.style_map_state import (
    StyleMapState,
    map_point_to_scatter_point,
    uploaded_to_scatter_point,
)
from views.style_map_toolbar import StyleMapToolbar
from views.style_map_widgets import StyleMapSidebar


class StyleMapPage(BasePage):
    """Visualize artworks projected onto PCA space with map interactions."""

    page_title = "风格地图"
    page_subtitle = (
        "Step 7：在 PCA 空间中查看艺术作品分布，筛选风格并定位上传图片。"
    )

    def __init__(self, controller: PageDataController) -> None:
        """Create PCA style map widgets."""
        super().__init__(controller)
        self._toolbar = StyleMapToolbar()
        self._sidebar = StyleMapSidebar()
        self._state = StyleMapState()
        self._status_label = QLabel("尚未加载 PCA 地图。")
        self._status_label.setObjectName("hintLabel")
        self._chart = PCAStyleMapChart()
        self._content_stack = QStackedWidget()
        self._empty_panel = EmptyActionPanel(
            "请先导入真实艺术数据集并完成 PCA 训练。风格地图不会用 synthetic 数据生成论文截图。",
            "去导入数据集",
        )
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build map controls, chart area, and side filters."""
        self._toolbar.load_requested.connect(self.refresh)
        self._toolbar.upload_requested.connect(self._choose_uploaded_image)
        self._toolbar.clear_requested.connect(self._clear_highlight)
        self._toolbar.show_2d_requested.connect(self._render_2d)
        self._toolbar.show_3d_requested.connect(self._render_3d)
        self._sidebar.style_changed.connect(self._apply_style_filter)
        self._sidebar.search_requested.connect(self._highlight_search_result)
        self._empty_panel.button.clicked.connect(
            lambda: self.request_navigation(NavigationRoute.DATASET),
        )
        self._root_layout.addWidget(self._toolbar)
        self._content_stack.addWidget(self._empty_panel)
        self._content_stack.addWidget(self._create_map_content())
        self._root_layout.addWidget(self._content_stack, stretch=1)
        self._show_empty_state()

    def _create_map_content(self) -> QFrame:
        """Create chart and side controls shown after map data is loaded."""
        panel = QFrame()
        panel.setObjectName("transparentPanel")
        content_row = QHBoxLayout(panel)
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(16)
        content_row.addWidget(self._create_chart_panel(), stretch=1)
        content_row.addWidget(self._sidebar)
        return panel

    def _create_chart_panel(self) -> QFrame:
        """Create the chart panel containing status text and canvas."""
        panel = create_panel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self._status_label)
        layout.addWidget(self._chart, stretch=1)
        return panel

    def refresh(self) -> None:
        """Load projected artwork points and redraw the map."""
        self._toolbar.set_loading(True)
        self._status_label.setText("正在加载 PCA 地图...")
        self.run_task(
            "PCA地图加载失败",
            self._controller.get_pca_map_points,
            self._handle_success,
            self._finish_task,
            on_failure=lambda _error: self._show_empty_state(),
        )

    def _handle_success(self, result: object) -> None:
        """Render projected PCA map points."""
        if not self._is_point_list(result):
            self.show_error("PCA地图结果异常", "后端返回了无法识别的地图结果。")
            self._show_empty_state()
            return
        points = list(result)
        self._state.set_points(points)
        if not points:
            self._show_empty_state()
            return
        self._content_stack.setCurrentIndex(1)
        self._populate_style_filter()
        self._render_2d()
        self._update_status()
        self._update_legend()

    def _finish_task(self) -> None:
        """Restore map loading controls."""
        self._toolbar.set_loading(False)

    def _choose_uploaded_image(self) -> None:
        """Ask the user for an image and project it into the current map."""
        if not self._state.points:
            self.show_info("尚未加载地图", "请先加载 PCA 地图后再定位上传图片。")
            return
        image_path = choose_image_file(
            self,
            self._controller.config.paths.dataset_root,
        )
        if image_path is None:
            return
        self._toolbar.set_upload_enabled(False)
        self._status_label.setText(f"正在定位上传图片：{image_path.name}")
        self.run_task(
            "上传图片定位失败",
            self._controller.project_uploaded_image,
            self._handle_upload_success,
            self._finish_upload,
            image_path,
        )

    def _handle_upload_success(self, result: object) -> None:
        """Highlight an uploaded image after backend projection."""
        if not isinstance(result, UploadedMapPoint):
            self.show_error("上传图片定位异常", "后端返回了无法识别的上传点位。")
            return
        self._chart.highlight_point(uploaded_to_scatter_point(result))
        self._status_label.setText(
            f"上传图片已定位：{format_style_label(result.style_label)}，置信度 {result.confidence:.1%}。",
        )

    def _finish_upload(self) -> None:
        """Restore upload button after projection completes."""
        self._toolbar.set_upload_enabled(bool(self._state.points))

    def _render_2d(self) -> None:
        """Render loaded points in the first two principal components."""
        self._chart.draw_pca_map(self._state.points)
        self._apply_current_filter()
        self._toolbar.set_map_actions_enabled(bool(self._state.points))
        self._toolbar.set_3d_enabled(self._state.has_3d_points)

    def _render_3d(self) -> None:
        """Render loaded points in the first three principal components."""
        if not self._state.has_3d_points:
            self.show_info("三维视图不可用", "当前 PCA 模型没有保留第三主成分。")
            return
        self._chart.draw_pca_map_3d(self._state.points)
        self._apply_current_filter()

    def _apply_style_filter(self, _label: str) -> None:
        """Show only the selected style on the current map."""
        self._state.active_style = self._sidebar.selected_style()
        self._apply_current_filter()
        self._update_status()

    def _highlight_search_result(self, query_text: str) -> None:
        """Find an artwork by ID or keyword and highlight it on the map."""
        query = query_text.strip().lower()
        if not query:
            self.show_info("搜索条件为空", "请输入作品ID、标题、作者或风格关键字。")
            return
        point = self._state.find_point(query)
        if point is None:
            self.show_info("未找到作品", "当前地图中没有匹配的作品点。")
            return
        self._chart.highlight_point(map_point_to_scatter_point(point))
        self._status_label.setText(
            f"已高亮作品 #{point.artwork_id}：{point.title} / {format_style_label(point.style_label)}",
        )

    def _clear_highlight(self) -> None:
        """Clear uploaded or searched map highlight."""
        self._chart.clear_highlight()
        self._update_status()

    def _apply_current_filter(self) -> None:
        """Apply the selected style filter to the chart widget."""
        if not self._state.points:
            return
        self._chart.set_hidden_labels(self._state.hidden_labels())

    def _populate_style_filter(self) -> None:
        """Fill style filter values from loaded map data."""
        self._sidebar.set_styles(self._state.style_labels)
        self._state.active_style = self._sidebar.selected_style()

    def _update_status(self) -> None:
        """Update compact status text for loaded and filtered points."""
        self._status_label.setText(self._state.status_text())

    def _update_legend(self) -> None:
        """Update textual legend and map summary."""
        if not self._state.points:
            self._sidebar.set_legend_counts({})
            return
        self._sidebar.set_legend_counts(self._state.style_counts)

    def _show_empty_state(self) -> None:
        """Show a user-actionable empty state instead of a blank chart."""
        self._content_stack.setCurrentWidget(self._empty_panel)
        self._toolbar.set_map_actions_enabled(False)
        self._toolbar.set_3d_enabled(False)
        self._toolbar.set_upload_enabled(False)

    def _is_point_list(self, result: object) -> bool:
        """Return whether the result is a list of PCA map points."""
        return isinstance(result, list) and all(
            isinstance(point, PCAMapPoint) for point in result
        )
