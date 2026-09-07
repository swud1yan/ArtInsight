"""PCA analysis page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.navigation import NavigationRoute
from controllers.gui_dtos import PCABenefitViewData, PCAExplanationData
from controllers.page_data_controller import PageDataController
from views.dashboard_widgets import EmptyActionPanel, OverviewMetricWidget
from views.pca_algorithm_lab_widget import PCAAlgorithmLabWidget
from views.pca_analysis_state import PCAAnalysisState, PCAComponentRow
from views.pca_interpretation_widget import PCAInterpretationWidget
from views.pages.base_page import BasePage
from views.table_models import ReadOnlyTableModel, TableColumn
from views.ui_helpers import configure_table, create_button, create_panel
from views.widgets import button_row
from views.zoom_dialogs import show_enlarged_table


class PCAAnalysisPage(BasePage):
    """Display fitted PCA model eigenvalues and variance contribution."""

    page_title = "PCA分析"
    page_subtitle = (
        "Step 5：解释已训练模型中的自研 PCA 贡献率、主成分数量选择与重建误差，"
        "用于算法说明和论文结果可视化。"
    )

    def __init__(self, controller: PageDataController) -> None:
        """Create PCA analysis widgets."""
        super().__init__(controller)
        self._load_button: QPushButton
        self._state = PCAAnalysisState()
        self._summary_label = QLabel("尚未加载 PCA 模型。")
        self._summary_label.setObjectName("hintLabel")
        self._benefit_widgets = {
            "raw_accuracy": OverviewMetricWidget("Raw Feature Accuracy"),
            "pca_accuracy": OverviewMetricWidget("PCA Accuracy"),
            "accuracy_drop": OverviewMetricWidget("Accuracy Drop"),
            "compression": OverviewMetricWidget("Feature Compression"),
            "time": OverviewMetricWidget("Time Reduction"),
            "memory": OverviewMetricWidget("Memory Reduction"),
        }
        self._benefit_note = QLabel(
            "PCA 的价值不只看 Accuracy；维数降低、速度提高、内存减少和可解释性，"
            "才是降维在艺术风格分析中的核心收益。",
        )
        self._benefit_note.setObjectName("hintLabel")
        self._benefit_note.setWordWrap(True)
        self._interpretation_widget = PCAInterpretationWidget(self._state)
        self._interpretation_widget.set_empty_action(
            "请先完成模型训练。PCA 分析页面读取训练产物中的自研 PCA 参数，不会使用 synthetic 数据生成结果。",
            "去模型训练",
            lambda: self.request_navigation(NavigationRoute.MODEL_TRAINING),
        )
        self._lab_widget = PCAAlgorithmLabWidget()
        self._tabs = QTabWidget()
        self._table_panel: QWidget | None = None
        self._table_stack = QStackedWidget()
        self._table_empty = EmptyActionPanel(
            "请先完成模型训练。加载 PCA 结果后，"
            "这里会显示特征值、解释方差和累计贡献率。",
            "去模型训练",
        )
        self._table_empty.button.clicked.connect(
            lambda: self.request_navigation(NavigationRoute.MODEL_TRAINING),
        )
        self._table_model = ReadOnlyTableModel[PCAComponentRow](
            [
                TableColumn("主成分", lambda row: row.index),
                TableColumn("特征值", self._format_eigenvalue, Qt.AlignRight),
                TableColumn("解释方差", self._format_explained, Qt.AlignRight),
                TableColumn("累计贡献率", self._format_cumulative, Qt.AlignRight),
            ],
        )
        self._setup_ui()
        self._load_default_lab_demo()

    def _setup_ui(self) -> None:
        """Build PCA charts and table."""
        self._load_button = create_button("加载PCA结果", "chart", primary=True)
        self._load_button.clicked.connect(self.refresh)
        self._root_layout.addWidget(button_row(self._load_button))
        self._tabs.setObjectName("analysisTabs")
        self._tabs.addTab(self._create_interpretation_tab(), "贡献率与重建")
        self._tabs.addTab(self._lab_widget, "算法实验室")
        self._root_layout.addWidget(self._tabs, stretch=1)

    def _load_default_lab_demo(self) -> None:
        """Show a deterministic teaching demo before trained PCA results exist."""
        self._lab_widget.set_steps(self._controller.get_pca_algorithm_lab_demo_steps())

    def _create_interpretation_tab(self) -> QWidget:
        """Create the selected-K interpretation tab."""
        tab = QWidget()
        tab.setObjectName("tabPage")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        layout.addWidget(self._interpretation_widget, stretch=2)
        layout.addWidget(self._create_benefit_panel())
        self._table_panel = create_panel()
        table_layout = QVBoxLayout(self._table_panel)
        table_layout.setContentsMargins(16, 16, 16, 16)
        table_layout.setSpacing(12)
        self._table_view = QTableView()
        configure_table(self._table_view)
        self._table_view.setModel(self._table_model)
        zoom_table_button = create_button("放大表格", "search")
        zoom_table_button.clicked.connect(
            lambda: show_enlarged_table(self, self._table_view, "PCA主成分表"),
        )
        table_layout.addWidget(button_row(zoom_table_button))
        table_layout.addWidget(self._summary_label)
        self._table_stack.addWidget(self._table_empty)
        self._table_stack.addWidget(self._table_view)
        table_layout.addWidget(self._table_stack)
        self._table_panel.setVisible(False)
        layout.addWidget(self._table_panel, stretch=1)
        return tab

    def refresh(self) -> None:
        """Load PCA artifact data and update charts."""
        self._load_button.setEnabled(False)
        self._interpretation_widget.set_loading(True)
        self._summary_label.setText("正在加载 PCA 结果...")
        self.run_task(
            "PCA结果加载失败",
            self._load_analysis_payload,
            self._handle_success,
            self._finish_task,
        )

    def _load_analysis_payload(self) -> tuple[PCAExplanationData, PCABenefitViewData]:
        """Load PCA explanation and engineering-benefit metrics together."""
        return (
            self._controller.get_pca_explanation_data(),
            self._controller.get_pca_benefit_data(),
        )

    def _handle_success(self, result: object) -> None:
        """Render loaded PCA data."""
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], PCAExplanationData)
            or not isinstance(result[1], PCABenefitViewData)
        ):
            self.show_error("PCA结果异常", "后端返回了无法识别的 PCA 结果。")
            return
        data, benefit = result
        self._state.set_data(data)
        self._table_model.set_rows(self._state.table_rows())
        if self._table_panel is not None:
            self._table_panel.setVisible(True)
        self._table_stack.setCurrentWidget(self._table_view)
        self._summary_label.setText(self._state.summary_text())
        self._interpretation_widget.load_data(data)
        self._lab_widget.set_steps(data.lab_steps)
        self._render_benefits(benefit)

    def _finish_task(self) -> None:
        """Restore PCA loading controls."""
        self._load_button.setEnabled(True)
        self._interpretation_widget.set_loading(False)

    def _format_eigenvalue(self, row: PCAComponentRow) -> str:
        """Format an eigenvalue for the table."""
        return f"{row.eigenvalue:.6f}"

    def _format_explained(self, row: PCAComponentRow) -> str:
        """Format explained variance ratio."""
        return f"{row.explained_ratio * 100:.2f}%"

    def _format_cumulative(self, row: PCAComponentRow) -> str:
        """Format cumulative variance ratio."""
        return f"{row.cumulative_ratio * 100:.2f}%"

    def _create_benefit_panel(self) -> QWidget:
        """Create PCA benefit metrics panel."""
        panel = create_panel()
        layout = QGridLayout(panel)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)
        for index, widget in enumerate(self._benefit_widgets.values()):
            layout.addWidget(widget, index // 3, index % 3)
        layout.addWidget(self._benefit_note, 2, 0, 1, 3)
        return panel

    def _render_benefits(self, benefit: PCABenefitViewData) -> None:
        """Render raw-vs-PCA benefit metrics."""
        self._benefit_widgets["raw_accuracy"].set_display_value(
            self._format_optional_percent(benefit.raw_feature_accuracy),
        )
        self._benefit_widgets["pca_accuracy"].set_display_value(
            self._format_optional_percent(benefit.pca_accuracy),
        )
        self._benefit_widgets["accuracy_drop"].set_display_value(
            self._format_signed_percent(
                -benefit.accuracy_drop if benefit.accuracy_drop is not None else None,
            ),
        )
        self._benefit_widgets["compression"].set_display_value(
            (
                f"{benefit.feature_compression_ratio:.1f}x "
                f"({benefit.raw_dimension}->{benefit.pca_dimension})"
            )
            if benefit.feature_compression_ratio is not None
            else "N/A",
        )
        self._benefit_widgets["time"].set_display_value(
            self._format_optional_percent(benefit.time_reduction),
        )
        self._benefit_widgets["memory"].set_display_value(
            self._format_optional_percent(benefit.memory_reduction),
        )
        self._benefit_note.setText(
            "PCA 的价值不只看 Accuracy；"
            f"本次轻量评估使用 {benefit.sample_count} 个样本，"
            f"预测内存从 {self._format_mb(benefit.raw_memory_mb)} MB 降到 "
            f"{self._format_mb(benefit.pca_memory_mb)} MB。"
        )

    def _format_optional_percent(self, value: float | None) -> str:
        """Format nullable ratio as percent."""
        return "N/A" if value is None else f"{value * 100:.2f}%"

    def _format_signed_percent(self, value: float | None) -> str:
        """Format nullable signed ratio as percent."""
        return "N/A" if value is None else f"{value * 100:+.2f}%"

    def _format_mb(self, value: float | None) -> str:
        """Format nullable megabytes."""
        return "N/A" if value is None else f"{value:.2f}"
