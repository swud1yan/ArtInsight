"""Home-dashboard section widgets."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.navigation import NavigationRoute
from controllers.gui_dtos import AppSummary, PCAMapPoint
from views.chart_widgets import MatplotlibChart
from views.dashboard_widgets import (
    DashboardCard,
    EmptyActionPanel,
    OverviewMetricWidget,
    RecentExperimentRow,
    StatusStepWidget,
)
from views.ui_helpers import create_button
from visualization.pca_chart_models import PCAScatterPoint


@dataclass(frozen=True)
class WorkflowStep:
    """A top-level analysis step shown on the dashboard."""

    number: int
    title: str
    route: NavigationRoute


WORKFLOW_STEPS: tuple[WorkflowStep, ...] = (
    WorkflowStep(1, "导入数据集", NavigationRoute.DATASET),
    WorkflowStep(2, "数据集检查", NavigationRoute.DATASET),
    WorkflowStep(3, "特征提取", NavigationRoute.FEATURE_EXTRACTION),
    WorkflowStep(4, "模型训练", NavigationRoute.MODEL_TRAINING),
    WorkflowStep(5, "PCA分析", NavigationRoute.PCA_ANALYSIS),
    WorkflowStep(6, "风格分类", NavigationRoute.STYLE_CLASSIFICATION),
    WorkflowStep(7, "相似检索", NavigationRoute.SIMILARITY_RETRIEVAL),
    WorkflowStep(8, "实验统计", NavigationRoute.STATISTICS),
    WorkflowStep(9, "风格地图", NavigationRoute.STYLE_MAP),
    WorkflowStep(10, "导出报告", NavigationRoute.REPORT_EXPORT),
)


class DashboardHero(QFrame):
    """Primary quick-start panel for the home dashboard."""

    navigation_requested = Signal(str)
    refresh_requested = Signal()

    def __init__(self) -> None:
        """Create the hero section."""
        super().__init__()
        self.setObjectName("dashboardHero")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(18)

        text_block = QWidget()
        text_layout = QVBoxLayout(text_block)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(8)
        title = QLabel("ArtInsight 实验工作台")
        title.setObjectName("dashboardHeroTitle")
        subtitle = QLabel(
            "按向导流程完成真实艺术数据集分析、PCA降维、风格识别、相似检索与论文报告导出。",
        )
        subtitle.setObjectName("dashboardHeroSubtitle")
        subtitle.setWordWrap(True)
        text_layout.addWidget(title)
        text_layout.addWidget(subtitle)

        self.start_button = create_button("开始新的分析", "play", primary=True)
        self.start_button.setObjectName("heroPrimaryButton")
        self.start_button.clicked.connect(
            lambda: self.navigation_requested.emit(NavigationRoute.DATASET.value),
        )
        self.refresh_button = create_button("刷新状态", "refresh")
        self.refresh_button.clicked.connect(self.refresh_requested)

        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.refresh_button)
        button_layout.addStretch(1)
        text_layout.addWidget(button_container)

        layout.addWidget(text_block, stretch=1)


class DashboardStatusPanel(DashboardCard):
    """Current project status checklist."""

    def __init__(self) -> None:
        """Create status items."""
        super().__init__("当前项目状态", "绿色表示该阶段已准备好，红色表示需要继续完成。")
        layout = QGridLayout()
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        self.content_layout.addLayout(layout)
        self._items = {
            "dataset": StatusStepWidget(
                "数据集已导入",
                "已找到真实艺术图像记录",
                "请先导入WikiArt数据集",
            ),
            "feature": StatusStepWidget(
                "特征已提取",
                "颜色与纹理特征可用于训练",
                "请完成RGB/HSV/HOG/LBP特征提取",
            ),
            "model": StatusStepWidget(
                "模型已训练",
                "分类与检索模型可调用",
                "请完成模型训练",
            ),
            "pca": StatusStepWidget(
                "PCA已完成",
                "主成分与贡献率可视化就绪",
                "请加载训练模型中的PCA结果",
            ),
        }
        for index, item in enumerate(self._items.values()):
            layout.addWidget(item, 0, index)

    def render(self, summary: AppSummary) -> None:
        """Render project status values."""
        self._items["dataset"].set_done(summary.artwork_count > 0)
        self._items["feature"].set_done(summary.feature_count > 0)
        self._items["pca"].set_done(summary.pca_completed)
        self._items["model"].set_done(summary.model_available)


class DashboardWorkflowPanel(DashboardCard):
    """Ten-step wizard overview."""

    navigation_requested = Signal(str)

    def __init__(self) -> None:
        """Create workflow cards."""
        super().__init__("快速开始", "按照从左到右的流程完成一次完整实验。")
        layout = QGridLayout()
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        self.content_layout.addLayout(layout)
        self._cards: list[QFrame] = []
        for index, step in enumerate(WORKFLOW_STEPS):
            card = self._create_step_card(step)
            self._cards.append(card)
            layout.addWidget(card, index // 5, index % 5)

    def render(self, summary: AppSummary) -> None:
        """Highlight the next suggested workflow card."""
        active_index = next_workflow_index(summary)
        for index, card in enumerate(self._cards):
            card.setObjectName(
                "workflowStepActive" if index == active_index else "workflowStep",
            )
            card.style().unpolish(card)
            card.style().polish(card)

    def _create_step_card(self, step: WorkflowStep) -> QFrame:
        """Create one clickable workflow card."""
        card = QFrame()
        card.setObjectName("workflowStep")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(4)
        number = QLabel(f"Step {step.number}")
        number.setObjectName("workflowStepNumber")
        title = QLabel(step.title)
        title.setObjectName("workflowStepTitle")
        title.setWordWrap(True)
        card_layout.addWidget(number)
        card_layout.addWidget(title)
        card.mousePressEvent = (  # type: ignore[method-assign]
            lambda _event, route=step.route: self.navigation_requested.emit(
                route.value,
            )
        )
        return card


class DashboardOverviewPanel(DashboardCard):
    """Data overview metric section."""

    def __init__(self) -> None:
        """Create overview metrics."""
        super().__init__("数据概览")
        layout = QGridLayout()
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        self.content_layout.addLayout(layout)
        self._metrics = {
            "artwork": OverviewMetricWidget("作品数量"),
            "style": OverviewMetricWidget("风格数量"),
            "feature": OverviewMetricWidget("特征数量"),
            "component": OverviewMetricWidget("主成分数量"),
            "variance": OverviewMetricWidget("累计贡献率"),
        }
        for index, metric in enumerate(self._metrics.values()):
            layout.addWidget(metric, index // 3, index % 3)

    def render(self, summary: AppSummary) -> None:
        """Render overview metrics."""
        self._metrics["artwork"].set_display_value(
            count_or_pending(summary.artwork_count, "未导入"),
        )
        self._metrics["style"].set_display_value(
            count_or_pending(summary.style_count, "待识别"),
        )
        self._metrics["feature"].set_display_value(
            count_or_pending(summary.feature_count, "待提取"),
        )
        self._metrics["component"].set_display_value(
            count_or_pending(summary.pca_component_count, "待分析"),
        )
        self._metrics["variance"].set_display_value(
            percent_or_pending(summary.cumulative_variance_ratio, "待分析"),
        )


class DashboardRecentPanel(DashboardCard):
    """Recent experiment summary section."""

    def __init__(self) -> None:
        """Create recent experiment rows."""
        super().__init__("最近实验")
        self._rows = {
            "time": RecentExperimentRow("最近一次实验时间"),
            "dataset": RecentExperimentRow("使用的数据集"),
            "components": RecentExperimentRow("PCA维数"),
            "accuracy": RecentExperimentRow("Accuracy"),
            "precision": RecentExperimentRow("Precision"),
            "recall": RecentExperimentRow("Recall"),
        }
        for row in self._rows.values():
            self.content_layout.addWidget(row)

    def render(self, summary: AppSummary) -> None:
        """Render recent experiment fields."""
        self._rows["time"].set_value(summary.latest_experiment_time)
        self._rows["dataset"].set_value(summary.latest_dataset_name)
        self._rows["components"].set_value(
            count_or_pending(summary.pca_component_count, "待分析"),
        )
        self._rows["accuracy"].set_value(
            percent_or_pending(summary.latest_accuracy, "待评估"),
        )
        self._rows["precision"].set_value(
            percent_or_pending(summary.latest_precision, "待评估"),
        )
        self._rows["recall"].set_value(
            percent_or_pending(summary.latest_recall, "待评估"),
        )


class DashboardMapPreviewPanel(DashboardCard):
    """PCA style-map preview section."""

    navigation_requested = Signal(str)

    def __init__(self) -> None:
        """Create the map preview panel."""
        super().__init__("风格地图预览", "PCA二维散点图用于观察不同艺术风格在主成分空间中的分布。")
        self.stack = QStackedWidget()
        self.chart = MatplotlibChart()
        self.empty = EmptyActionPanel(
            "请先导入WikiArt数据集，完成特征提取与PCA分析后即可预览风格地图。",
            "立即导入",
        )
        self.empty.button.clicked.connect(
            lambda: self.navigation_requested.emit(NavigationRoute.DATASET.value),
        )
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self.chart)
        self.content_layout.addWidget(self.stack, stretch=1)

    def render(self, summary: AppSummary, map_points: list[PCAMapPoint]) -> None:
        """Render the preview chart or an action-oriented empty state."""
        if map_points:
            self._draw_points(map_points)
            return
        if summary.artwork_count <= 0:
            self.empty.set_message("请先导入WikiArt数据集，然后系统会在这里显示PCA二维风格地图。")
            self.empty.set_button_text("立即导入")
            self._rewire_empty_button(NavigationRoute.DATASET)
        else:
            self.empty.set_message("数据集已就绪。请先完成特征提取和模型训练，然后查看PCA二维风格地图。")
            self.empty.set_button_text("进入模型训练")
            self._rewire_empty_button(NavigationRoute.MODEL_TRAINING)
        self.stack.setCurrentWidget(self.empty)

    def _draw_points(self, map_points: list[PCAMapPoint]) -> None:
        """Draw a bounded number of PCA points for a fast home preview."""
        chart_points = [
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
            for point in map_points[:800]
        ]
        self.chart.draw_pca_scatter(chart_points, "PCA二维风格地图预览")
        self.stack.setCurrentWidget(self.chart)

    def _rewire_empty_button(self, route: NavigationRoute) -> None:
        """Connect the empty-state button to a single target route."""
        try:
            self.empty.button.clicked.disconnect()
        except RuntimeError:
            pass
        self.empty.button.clicked.connect(
            lambda: self.navigation_requested.emit(route.value),
        )


def next_workflow_index(summary: AppSummary) -> int:
    """Return the zero-based index of the next suggested workflow step."""
    if summary.artwork_count <= 0:
        return 0
    if summary.feature_count <= 0:
        return 2
    if not summary.model_available:
        return 3
    if not summary.pca_completed:
        return 4
    return 5


def count_or_pending(value: int, pending_text: str) -> str:
    """Return a count when positive, otherwise an action-oriented state."""
    return f"{value:,}" if value > 0 else pending_text


def percent_or_pending(value: float | None, pending_text: str) -> str:
    """Return a percentage when present, otherwise a pending state."""
    if value is None:
        return pending_text
    return f"{value:.1%}"
