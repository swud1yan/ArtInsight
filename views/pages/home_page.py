"""Modern workflow dashboard page."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QWidget

from controllers.gui_dtos import AppSummary, PCAMapPoint
from controllers.page_data_controller import PageDataController
from views.home_dashboard_sections import (
    DashboardHero,
    DashboardMapPreviewPanel,
    DashboardOverviewPanel,
    DashboardRecentPanel,
    DashboardStatusPanel,
    DashboardWorkflowPanel,
    next_workflow_index,
)
from views.pages.base_page import BasePage


@dataclass(frozen=True)
class HomeDashboardData:
    """Combined dashboard data used by the home page."""

    summary: AppSummary
    map_points: list[PCAMapPoint]


class HomePage(BasePage):
    """Wizard-style dashboard for the full ArtInsight workflow."""

    page_title = "首页"
    page_subtitle = "从数据导入到论文报告导出的实验工作台"

    def __init__(self, controller: PageDataController) -> None:
        """Create the modern dashboard page."""
        super().__init__(controller)
        self._hero = DashboardHero()
        self._status_panel = DashboardStatusPanel()
        self._workflow_panel = DashboardWorkflowPanel()
        self._overview_panel = DashboardOverviewPanel()
        self._recent_panel = DashboardRecentPanel()
        self._map_panel = DashboardMapPreviewPanel()
        self._setup_ui()
        QTimer.singleShot(0, self.refresh)

    def _setup_ui(self) -> None:
        """Build dashboard sections."""
        self._hero.refresh_requested.connect(self.refresh)
        self._hero.navigation_requested.connect(self.navigation_requested)
        self._workflow_panel.navigation_requested.connect(self.navigation_requested)
        self._map_panel.navigation_requested.connect(self.navigation_requested)

        self._root_layout.addWidget(self._hero)
        self._root_layout.addWidget(self._status_panel)
        self._root_layout.addWidget(self._workflow_panel)

        content_row = QWidget()
        content_layout = QHBoxLayout(content_row)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)
        content_layout.addWidget(self._overview_panel, stretch=2)
        content_layout.addWidget(self._recent_panel, stretch=1)
        self._root_layout.addWidget(content_row)
        self._root_layout.addWidget(self._map_panel, stretch=1)

    def refresh(self) -> None:
        """Refresh dashboard data from the controller."""
        self._hero.refresh_button.setEnabled(False)
        self.run_task(
            "首页刷新失败",
            self._load_dashboard_data,
            self._handle_success,
            self._finish_task,
        )

    def _load_dashboard_data(self) -> HomeDashboardData:
        """Load dashboard data outside the GUI thread."""
        summary = self._controller.get_summary()
        map_points: list[PCAMapPoint] = []
        if summary.pca_completed:
            try:
                map_points = self._controller.get_pca_map_points()
            except (FileNotFoundError, ValueError, TypeError, AttributeError):
                map_points = []
        return HomeDashboardData(summary=summary, map_points=map_points)

    def _handle_success(self, result: object) -> None:
        """Render loaded dashboard data."""
        if not isinstance(result, HomeDashboardData):
            self.show_error("首页结果异常", "后端返回了无法识别的首页统计结果。")
            return
        summary = result.summary
        self._status_panel.render(summary)
        self._workflow_panel.render(summary)
        self._overview_panel.render(summary)
        self._recent_panel.render(summary)
        self._map_panel.render(summary, result.map_points)

    def _finish_task(self) -> None:
        """Restore dashboard refresh controls."""
        self._hero.refresh_button.setEnabled(True)

    def _next_workflow_index(self, summary: AppSummary) -> int:
        """Return the zero-based index of the next suggested workflow step."""
        return next_workflow_index(summary)

    @property
    def start_button(self) -> object:
        """Return the primary start button for tests and automation."""
        return self._hero.start_button

    @property
    def map_panel(self) -> DashboardMapPreviewPanel:
        """Return the map preview panel for tests and automation."""
        return self._map_panel
