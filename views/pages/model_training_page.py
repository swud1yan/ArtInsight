"""Model training page for style classification and retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from app.navigation import NavigationRoute
from controllers.gui_dtos import FeatureExtractionRunResult, TrainingViewResult
from controllers.page_data_controller import PageDataController
from views.dashboard_widgets import DashboardCard
from views.pages.base_page import BasePage
from views.pages.feature_extraction_page import FEATURE_METHODS, FeatureSummaryCard
from views.ui_helpers import create_button, create_panel
from views.widgets import button_row


@dataclass(frozen=True)
class TrainingFeatureMode:
    """One persisted feature family available to the training workflow."""

    label: str
    feature_type: str
    methods: tuple[str, ...]
    description: str


BASE_FEATURE_METHODS = tuple(option.method_name for option in FEATURE_METHODS)
TRAINING_FEATURE_MODES: tuple[TrainingFeatureMode, ...] = (
    TrainingFeatureMode(
        label="快速模式：RGB + HSV + HOG + LBP",
        feature_type="combined_quick",
        methods=BASE_FEATURE_METHODS,
        description="推荐先用于当前启用数据集的首次跑通与演示；不包含 GLCM，速度明显更稳。",
    ),
    TrainingFeatureMode(
        label="完整模式：快速特征 + GLCM",
        feature_type="combined",
        methods=BASE_FEATURE_METHODS + ("glcm",),
        description="包含灰度共生矩阵，适合最终消融实验，但会显著增加批量提取时间。",
    ),
)


class ModelTrainingPage(BasePage):
    """Train the PCA/KNN model used by classification and retrieval."""

    page_title = "模型训练"
    page_subtitle = (
        "Step 4：先将真实艺术数据集批量提取为特征矩阵，再训练标准化 + 自研 PCA + KNN 模型。"
    )

    def __init__(self, controller: PageDataController) -> None:
        """Create model training widgets."""
        super().__init__(controller)
        self._extract_button: QPushButton
        self._recompute_button: QPushButton
        self._train_button: QPushButton
        self._pca_button: QPushButton
        self._status_label = QLabel(
            "请先导入真实艺术数据集，然后按顺序完成特征入库与模型训练。"
        )
        self._status_label.setObjectName("hintLabel")
        self._status_label.setWordWrap(True)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setTextVisible(False)
        self._progress.hide()
        self._feature_mode_combo = QComboBox()
        self._auto_components_checkbox = QCheckBox("自动选择 PCA 维度")
        self._components_spin = QSpinBox()
        self._components_spin.setRange(1, 512)
        self._components_spin.setValue(200)
        self._neighbors_spin = QSpinBox()
        self._neighbors_spin.setRange(1, 50)
        self._neighbors_spin.setValue(5)
        self._folds_spin = QSpinBox()
        self._folds_spin.setRange(2, 10)
        self._folds_spin.setValue(5)
        self._sample_metric = FeatureSummaryCard("训练样本")
        self._feature_metric = FeatureSummaryCard("特征维数")
        self._style_metric = FeatureSummaryCard("风格类别")
        self._component_metric = FeatureSummaryCard("主成分数")
        self._accuracy_metric = FeatureSummaryCard("CV Accuracy")
        self._model_metric = FeatureSummaryCard("模型状态")
        self._setup_ui()
        self._reset_metrics()

    def _setup_ui(self) -> None:
        """Build training workflow controls."""
        self._extract_button = create_button("1. 继续提取缺失特征", "play", primary=True)
        self._recompute_button = create_button("重新计算全部特征", "refresh", danger=True)
        self._train_button = create_button("2. 开始训练模型", "chart", primary=True)
        self._pca_button = create_button("查看PCA分析", "chart")
        self._extract_button.clicked.connect(self._extract_features)
        self._recompute_button.clicked.connect(self._confirm_recompute_features)
        self._train_button.clicked.connect(self._train_model)
        self._pca_button.clicked.connect(
            lambda: self.request_navigation(NavigationRoute.PCA_ANALYSIS),
        )
        self._root_layout.addWidget(
            button_row(
                self._extract_button,
                self._train_button,
                self._pca_button,
                self._recompute_button,
            ),
        )
        self._root_layout.addWidget(self._create_workspace(), stretch=1)

    def _create_workspace(self) -> QFrame:
        """Create the training workspace."""
        panel = create_panel()
        layout = QGridLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(16)
        layout.addWidget(self._create_pipeline_card(), 0, 0)
        layout.addWidget(self._create_config_card(), 0, 1)
        layout.addWidget(self._create_metric_panel(), 1, 0, 1, 2)
        layout.addWidget(self._create_status_panel(), 2, 0, 1, 2)
        return panel

    def _create_pipeline_card(self) -> DashboardCard:
        """Create the workflow explanation card."""
        card = DashboardCard(
            "训练流程",
            "该页面生成分类、检索、PCA解释和风格地图共用的模型 artifact；默认支持断点续跑。",
        )
        steps = QLabel(
            "1. 从数据库读取已导入作品\n"
            "2. 对每张图像执行 224x224 标准预处理\n"
            "3. 默认只提取缺失特征，第二次运行会自动续跑\n"
            "4. 使用 StandardScaler 标准化特征\n"
            "5. 使用自研 PCA 降维，再训练 KNN 分类器与近邻检索索引"
        )
        steps.setObjectName("hintLabel")
        steps.setWordWrap(True)
        card.content_layout.addWidget(steps)
        return card

    def _create_config_card(self) -> DashboardCard:
        """Create compact training parameter controls."""
        card = DashboardCard(
            "训练参数",
            "默认快速模式适合先跑通当前启用数据集；PCA主成分建议从50起步，可做50/100/200对比。",
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for mode in TRAINING_FEATURE_MODES:
            self._feature_mode_combo.addItem(mode.label)
        self._feature_mode_combo.currentIndexChanged.connect(self._update_mode_hint)
        self._auto_components_checkbox.toggled.connect(self._update_mode_hint)
        grid.addWidget(QLabel("特征模式"), 0, 0)
        grid.addWidget(self._feature_mode_combo, 0, 1)
        grid.addWidget(QLabel("PCA主成分/上限"), 1, 0)
        grid.addWidget(self._components_spin, 1, 1)
        grid.addWidget(self._auto_components_checkbox, 2, 0, 1, 2)
        grid.addWidget(QLabel("KNN邻居数"), 3, 0)
        grid.addWidget(self._neighbors_spin, 3, 1)
        grid.addWidget(QLabel("交叉验证折数"), 4, 0)
        grid.addWidget(self._folds_spin, 4, 1)
        card.content_layout.addLayout(grid)
        self._mode_hint = QLabel()
        self._mode_hint.setObjectName("hintLabel")
        self._mode_hint.setWordWrap(True)
        card.content_layout.addWidget(self._mode_hint)
        self._update_mode_hint()
        return card

    def _create_metric_panel(self) -> QFrame:
        """Create model result metric cards."""
        panel = QFrame()
        panel.setObjectName("featureMetricPanel")
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)
        for index, metric in enumerate(
            (
                self._sample_metric,
                self._feature_metric,
                self._style_metric,
                self._component_metric,
                self._accuracy_metric,
                self._model_metric,
            ),
        ):
            layout.addWidget(metric, index // 3, index % 3)
        return panel

    def _create_status_panel(self) -> QFrame:
        """Create status and progress area."""
        panel = create_panel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress)
        return panel

    def _extract_features(self) -> None:
        """Extract dataset features through the controller."""
        self._start_feature_extraction(replace_existing=False)

    def _confirm_recompute_features(self) -> None:
        """Ask before deleting and recomputing all features for the selected mode."""
        mode = self._selected_feature_mode()
        reply = QMessageBox.question(
            self,
            "确认重新计算",
            (
                f"将删除并重新计算“{mode.label}”的全部特征。\n"
                "当前启用数据集图片较多时会耗时较久；如果只是继续上次任务，请使用“继续提取缺失特征”。"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._start_feature_extraction(replace_existing=True)

    def _start_feature_extraction(self, replace_existing: bool) -> None:
        """Extract selected feature mode through the controller."""
        mode = self._selected_feature_mode()
        action = "重新计算" if replace_existing else "继续提取缺失"
        self._set_busy(
            True,
            f"正在{action}特征：{mode.label}。请保持应用打开，日志会持续记录进度。",
        )
        self.run_task(
            "批量特征提取失败",
            self._controller.extract_dataset_features,
            self._handle_feature_success,
            self._finish_task,
            list(mode.methods),
            mode.feature_type,
            replace_existing,
        )

    def _train_model(self) -> None:
        """Train the style model through the controller."""
        mode = self._selected_feature_mode()
        self._set_busy(True, "正在训练 PCA/KNN 模型，请稍候...")
        self.run_task(
            "模型训练失败",
            self._controller.train_style_model,
            self._handle_training_success,
            self._finish_task,
            self._components_spin.value(),
            self._neighbors_spin.value(),
            self._folds_spin.value(),
            mode.feature_type,
            list(mode.methods),
            self._auto_components_checkbox.isChecked(),
        )

    def _handle_feature_success(self, result: object) -> None:
        """Render batch feature extraction summary."""
        if not isinstance(result, FeatureExtractionRunResult):
            self.show_error("特征提取结果异常", "后端返回了无法识别的特征提取结果。")
            return
        self._feature_metric.set_display_value(result.feature_type)
        self._model_metric.set_display_value("待训练")
        self._status_label.setText(
            f"特征入库完成：新增 {result.processed_count}，跳过 {result.skipped_count}，"
            f"失败 {result.failed_count}。特征类型：{result.feature_type}。现在可以开始训练模型。"
        )

    def _handle_training_success(self, result: object) -> None:
        """Render model training summary."""
        if not isinstance(result, TrainingViewResult):
            self.show_error("训练结果异常", "后端返回了无法识别的训练结果。")
            return
        self._sample_metric.set_display_value(f"{result.sample_count:,}")
        self._feature_metric.set_display_value(f"{result.feature_count:,}")
        self._style_metric.set_display_value(f"{result.style_count:,}")
        self._component_metric.set_display_value(f"{result.n_components}")
        self._accuracy_metric.set_display_value(
            f"{result.cross_validation_accuracy:.1%}"
            if result.cross_validation_accuracy is not None
            else "样本不足",
        )
        self._model_metric.set_display_value("已训练")
        detail = (
            f"\n自动选维：{result.component_selection_note}"
            if result.component_selection_note
            else ""
        )
        self._status_label.setText(
            f"模型训练完成：{result.model_path}。本次保留 {result.n_components} 个 PCA 主成分；"
            "若分类置信度偏低，请提高搜索上限或改用 KNN=1 重新训练。"
            f"{detail}"
        )

    def _finish_task(self) -> None:
        """Restore controls after a training task."""
        self._set_busy(False)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        """Toggle controls while extraction or training is running."""
        for button in (
            self._extract_button,
            self._recompute_button,
            self._train_button,
            self._pca_button,
        ):
            button.setEnabled(not busy)
        for spin_box in (self._components_spin, self._neighbors_spin, self._folds_spin):
            spin_box.setEnabled(not busy)
        self._feature_mode_combo.setEnabled(not busy)
        self._auto_components_checkbox.setEnabled(not busy)
        self._progress.setRange(0, 0 if busy else 1)
        self._progress.setValue(0 if busy else 1)
        self._progress.setVisible(busy)
        if message is not None:
            self._status_label.setText(message)

    def _reset_metrics(self) -> None:
        """Initialize metric cards without misleading zeroes."""
        self._sample_metric.set_display_value("待训练")
        self._feature_metric.set_display_value("待入库")
        self._style_metric.set_display_value("待训练")
        self._component_metric.set_display_value("待训练")
        self._accuracy_metric.set_display_value("待评估")
        self._model_metric.set_display_value("未训练")

    def _selected_feature_mode(self) -> TrainingFeatureMode:
        """Return the currently selected persisted feature mode."""
        index = self._feature_mode_combo.currentIndex()
        if index < 0 or index >= len(TRAINING_FEATURE_MODES):
            return TRAINING_FEATURE_MODES[0]
        return TRAINING_FEATURE_MODES[index]

    def _update_mode_hint(self, _index: int | None = None) -> None:
        """Refresh feature-mode explanatory text."""
        mode = self._selected_feature_mode()
        component_mode = (
            "勾选自动选择时会在候选维度中寻找准确率接近最高、维度更小的折中值；"
            "未勾选时直接使用填写的 PCA 主成分数。"
        )
        self._mode_hint.setText(
            f"{mode.description}\n如果只保留 2 个主成分，PCA 分析通常只展示极低的累计解释方差；"
            f"数据库特征类型：{mode.feature_type}；"
            f"方法：{', '.join(mode.methods)}。\n{component_mode}"
        )
