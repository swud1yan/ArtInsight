"""Statistical analysis page."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.navigation import NavigationRoute
from controllers.gui_dtos import AppSummary, PCAViewData
from controllers.page_data_controller import PageDataController
from services.style_translations import format_style_distribution
from views.chart_widgets import MatplotlibChart
from views.dashboard_widgets import DashboardCard, EmptyActionPanel, OverviewMetricWidget
from views.pages.base_page import BasePage
from views.ui_helpers import create_button, create_panel
from views.widgets import button_row


@dataclass(frozen=True)
class StatisticsViewData:
    """Combined data needed by the statistics page."""

    summary: AppSummary
    style_counts: dict[str, int]
    artist_counts: dict[str, int]
    pca_data: PCAViewData | None


class StatisticsPage(BasePage):
    """Display dataset, author, style, and PCA statistics charts."""

    page_title = "统计分析"
    page_subtitle = (
        "汇总真实艺术数据集的类别分布、作者分布和 PCA 方差贡献，"
        "用于检查实验数据健康度与论文图表素材。"
    )

    def __init__(self, controller: PageDataController) -> None:
        """Create statistics charts."""
        super().__init__(controller)
        self._refresh_button: QPushButton
        self._style_chart = MatplotlibChart()
        self._artist_chart = MatplotlibChart()
        self._pca_chart = MatplotlibChart()
        self._status_label = QLabel("点击刷新加载统计分析。")
        self._status_label.setObjectName("hintLabel")
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setTextVisible(False)
        self._progress.hide()
        self._artwork_metric = OverviewMetricWidget("作品数量")
        self._style_metric = OverviewMetricWidget("风格数量")
        self._artist_metric = OverviewMetricWidget("作者数量")
        self._pca_metric = OverviewMetricWidget("PCA状态")
        self._content_stack = QStackedWidget()
        self._empty_panel = EmptyActionPanel(
            "请先导入 WikiArt 或 Kaggle 真实艺术数据集。统计分析只展示真实数据结果，"
            "不会使用 synthetic 数据生成论文指标。",
            "立即导入数据集",
        )
        self._setup_ui()
        self._render_empty_metrics()

    def _setup_ui(self) -> None:
        """Build statistical chart layout."""
        self._refresh_button = create_button("刷新统计", "refresh", primary=True)
        self._refresh_button.clicked.connect(self.refresh)
        self._empty_panel.button.clicked.connect(
            lambda: self.request_navigation(NavigationRoute.DATASET),
        )
        self._root_layout.addWidget(button_row(self._refresh_button))
        self._root_layout.addWidget(self._create_metric_panel())
        self._content_stack.addWidget(self._empty_panel)
        self._content_stack.addWidget(self._create_chart_panel())
        self._root_layout.addWidget(self._content_stack, stretch=1)

        status_panel = create_panel()
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(16, 12, 16, 12)
        status_layout.setSpacing(8)
        status_layout.addWidget(self._status_label)
        status_layout.addWidget(self._progress)
        self._root_layout.addWidget(status_panel)

    def refresh(self) -> None:
        """Refresh all statistical charts."""
        self._set_busy(True, "正在加载统计数据...")
        self.run_task(
            "统计刷新失败",
            self._load_statistics,
            self._handle_success,
            self._finish_task,
        )

    def _load_statistics(self) -> StatisticsViewData:
        """Load statistics data outside the GUI thread."""
        summary = self._controller.get_summary()
        style_counts = self._controller.style_distribution()
        artist_counts = self._controller.artist_distribution(limit=12)
        try:
            pca_data = self._controller.get_pca_view_data()
        except FileNotFoundError:
            pca_data = None
        return StatisticsViewData(summary, style_counts, artist_counts, pca_data)

    def _handle_success(self, result: object) -> None:
        """Draw statistical charts on the GUI thread."""
        if not isinstance(result, StatisticsViewData):
            self.show_error("统计结果异常", "后端返回了无法识别的统计结果。")
            return
        self._render_metrics(result)
        if not self._has_dataset(result):
            self._content_stack.setCurrentWidget(self._empty_panel)
            self._status_label.setText("请先导入真实艺术数据集后再查看统计分析。")
            return
        self._content_stack.setCurrentIndex(1)
        style_counts = format_style_distribution(result.style_counts)
        self._style_chart.draw_bar(
            "风格类别分布",
            list(style_counts.keys()),
            list(style_counts.values()),
        )
        self._artist_chart.draw_bar(
            "作者作品数 Top 12",
            list(result.artist_counts.keys()),
            list(result.artist_counts.values()),
        )
        self._draw_pca_chart(result.pca_data)
        self._status_label.setText("统计图已更新，可用于论文图表页面生成 300dpi 图像。")

    def _finish_task(self) -> None:
        """Restore statistics refresh controls."""
        self._set_busy(False)

    def _create_metric_panel(self) -> QFrame:
        """Create compact dataset health metrics."""
        panel = create_panel()
        layout = QGridLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)
        for index, metric in enumerate(
            (
                self._artwork_metric,
                self._style_metric,
                self._artist_metric,
                self._pca_metric,
            ),
        ):
            layout.addWidget(metric, 0, index)
        return panel

    def _create_chart_panel(self) -> QWidget:
        """Create the chart grid shown when real data exists."""
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(16)
        layout.addWidget(self._chart_card("风格分布", self._style_chart), 0, 0)
        layout.addWidget(self._chart_card("作者分布", self._artist_chart), 0, 1)
        layout.addWidget(self._chart_card("PCA贡献率", self._pca_chart), 1, 0, 1, 2)
        return panel

    def _chart_card(self, title: str, chart: MatplotlibChart) -> DashboardCard:
        """Wrap a chart in a dashboard card."""
        card = DashboardCard(title)
        chart.setMinimumHeight(260)
        card.content_layout.addWidget(chart, stretch=1)
        return card

    def _draw_pca_chart(self, pca_data: PCAViewData | None) -> None:
        """Draw PCA contribution chart or a no-data hint."""
        if pca_data is None:
            self._pca_chart.draw_empty("PCA贡献率", "请先完成特征提取与模型训练")
            return
        self._pca_chart.draw_explained_variance(
            pca_data.explained_variance_ratio,
            pca_data.cumulative_variance_ratio,
            max_components=20,
        )

    def _render_metrics(self, data: StatisticsViewData) -> None:
        """Render non-misleading metric-card values."""
        summary = data.summary
        self._artwork_metric.set_display_value(
            f"{summary.artwork_count:,}" if summary.artwork_count else "待导入",
        )
        self._style_metric.set_display_value(
            f"{summary.style_count:,}" if summary.style_count else "待检查",
        )
        self._artist_metric.set_display_value(
            f"{len(data.artist_counts):,}" if data.artist_counts else "待检查",
        )
        if data.pca_data is None:
            self._pca_metric.set_display_value("待训练")
        else:
            self._pca_metric.set_display_value(f"{data.pca_data.n_components} 个主成分")

    def _render_empty_metrics(self) -> None:
        """Initialize metric cards without meaningless zero values."""
        self._artwork_metric.set_display_value("待导入")
        self._style_metric.set_display_value("待检查")
        self._artist_metric.set_display_value("待检查")
        self._pca_metric.set_display_value("待训练")
        self._content_stack.setCurrentWidget(self._empty_panel)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        """Toggle controls while statistics are loading."""
        self._refresh_button.setEnabled(not busy)
        self._empty_panel.button.setEnabled(not busy)
        self._progress.setRange(0, 0 if busy else 1)
        self._progress.setValue(0 if busy else 1)
        self._progress.setVisible(busy)
        if message is not None:
            self._status_label.setText(message)

    def _has_dataset(self, data: StatisticsViewData) -> bool:
        """Return whether statistics are backed by real imported image records."""
        return data.summary.artwork_count > 0 or bool(data.style_counts)
