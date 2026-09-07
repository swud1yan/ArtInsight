"""Feature extraction preview page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from app.navigation import NavigationRoute
from controllers.gui_dtos import FeaturePreview
from controllers.page_data_controller import PageDataController
from views.dashboard_widgets import DashboardCard
from views.pages.base_page import BasePage
from views.ui_helpers import choose_image_file, create_button, create_panel
from views.widgets import ImagePreviewLabel, button_row


@dataclass(frozen=True)
class FeatureMethodOption:
    """One user-facing feature method option."""

    label: str
    method_name: str
    description: str


@dataclass(frozen=True)
class PreprocessingPreset:
    """One preprocessing preset for single-image feature preview."""

    label: str
    operations: tuple[str, ...]
    description: str


class FeatureSummaryCard(QFrame):
    """Compact metric card for single-image feature preview."""

    def __init__(self, title: str) -> None:
        """Create a compact feature-summary card."""
        super().__init__()
        self.setObjectName("compactMetric")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(3)
        self._value_label = QLabel("待提取")
        self._value_label.setObjectName("compactMetricValue")
        title_label = QLabel(title)
        title_label.setObjectName("compactMetricTitle")
        layout.addWidget(self._value_label)
        layout.addWidget(title_label)

    def set_display_value(self, value: str) -> None:
        """Update the visible metric value."""
        self._value_label.setText(value)


FEATURE_METHODS: tuple[FeatureMethodOption, ...] = (
    FeatureMethodOption(
        label="RGB",
        method_name="rgb_hist",
        description="颜色直方图，刻画作品整体色彩分布。",
    ),
    FeatureMethodOption(
        label="HSV",
        method_name="hsv_hist",
        description="色相、饱和度、亮度统计，适合风格色彩差异分析。",
    ),
    FeatureMethodOption(
        label="HOG",
        method_name="hog",
        description="梯度方向直方图，描述边缘与构图形状。",
    ),
    FeatureMethodOption(
        label="LBP",
        method_name="lbp",
        description="局部二值模式，描述笔触和纹理细节。",
    ),
)

PREPROCESSING_PRESETS: tuple[PreprocessingPreset, ...] = (
    PreprocessingPreset(
        label="论文标准 224x224",
        operations=("resize:224x224",),
        description="统一尺寸，保持特征长度稳定，适合作为默认实验设置。",
    ),
    PreprocessingPreset(
        label="构图保真填充",
        operations=("resize_pad:224x224",),
        description="保持原图比例并补边，适合避免构图被拉伸的预览。",
    ),
    PreprocessingPreset(
        label="纹理增强",
        operations=("resize:224x224", "equalize:clahe"),
        description="增强亮度对比，便于观察 LBP/HOG 等纹理和边缘特征。",
    ),
)


class FeatureExtractionPage(BasePage):
    """Preview preprocessing and handcrafted feature extraction output."""

    page_title = "特征提取"
    page_subtitle = (
        "Step 3：选择真实艺术图像，预览 RGB、HSV、HOG、LBP 等传统视觉特征，"
        "并在确认设置后进入模型训练页面执行批量特征入库。"
    )

    def __init__(self, controller: PageDataController) -> None:
        """Create feature preview widgets."""
        super().__init__(controller)
        self._selected_image: Path | None = None
        self._extract_button: QPushButton
        self._next_button: QPushButton
        self._preview = ImagePreviewLabel("请先选择艺术作品图片")
        self._preset_combo = QComboBox()
        self._method_checks: dict[str, QCheckBox] = {}
        self._status_label = QLabel("请选择一张真实艺术作品图片后开始特征预览。")
        self._status_label.setObjectName("hintLabel")
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setTextVisible(False)
        self._progress.hide()
        self._dimension_metric = FeatureSummaryCard("特征维数")
        self._dtype_metric = FeatureSummaryCard("数据类型")
        self._method_metric = FeatureSummaryCard("特征组合")
        self._time_metric = FeatureSummaryCard("预计耗时")
        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._setup_ui()
        self._reset_result_state()

    def _setup_ui(self) -> None:
        """Build feature extraction controls."""
        select_button = create_button("选择图片", "image", primary=True)
        self._extract_button = create_button("开始提取", "play")
        self._next_button = create_button("下一步：模型训练", "chart")
        select_button.clicked.connect(self._select_image)
        self._extract_button.clicked.connect(self._extract)
        self._next_button.clicked.connect(
            lambda: self.request_navigation(NavigationRoute.MODEL_TRAINING),
        )

        self._root_layout.addWidget(
            button_row(select_button, self._extract_button, self._next_button),
        )
        self._root_layout.addWidget(self._create_workspace(), stretch=1)

    def _create_workspace(self) -> QFrame:
        """Create the full feature extraction workspace."""
        panel = create_panel()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(18)
        layout.addWidget(self._create_image_panel(), stretch=2)
        layout.addWidget(self._create_control_panel(), stretch=3)
        return panel

    def _create_image_panel(self) -> QFrame:
        """Create the image input column."""
        panel = create_panel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("输入图像")
        title.setObjectName("cardTitle")
        hint = QLabel("仅用于单张图像预览。批量特征入库由数据管线和训练流程统一管理。")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(self._preview, stretch=1)
        layout.addWidget(hint)
        return panel

    def _create_control_panel(self) -> QFrame:
        """Create method selection, status, and output panels."""
        panel = create_panel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(self._create_preset_card())
        layout.addWidget(self._create_method_card())
        layout.addWidget(self._create_metric_panel())
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress)
        self._output.setMinimumHeight(112)
        self._output.setMaximumHeight(140)
        layout.addWidget(self._output)
        return panel

    def _create_preset_card(self) -> DashboardCard:
        """Create preprocessing preset selector."""
        card = DashboardCard("预处理预设", "预设只影响当前预览，不修改数据库中的原始图片。")
        for preset in PREPROCESSING_PRESETS:
            self._preset_combo.addItem(preset.label)
        self._preset_combo.currentIndexChanged.connect(
            lambda _index: self._update_preset_hint(),
        )
        self._preset_hint = QLabel(PREPROCESSING_PRESETS[0].description)
        self._preset_hint.setObjectName("hintLabel")
        self._preset_hint.setWordWrap(True)
        card.content_layout.addWidget(self._preset_combo)
        card.content_layout.addWidget(self._preset_hint)
        return card

    def _create_method_card(self) -> DashboardCard:
        """Create feature-method checkbox card."""
        card = DashboardCard("特征方法", "答辩演示中只暴露最核心的四类传统视觉特征。")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        for index, option in enumerate(FEATURE_METHODS):
            checkbox = QCheckBox(option.label)
            checkbox.setObjectName("featureMethodCheck")
            checkbox.setChecked(True)
            checkbox.setToolTip(option.description)
            checkbox.stateChanged.connect(lambda _state: self._refresh_method_metric())
            self._method_checks[option.method_name] = checkbox
            grid.addWidget(checkbox, index // 2, index % 2)
        card.content_layout.addLayout(grid)

        method_hint = QLabel(
            "RGB/HSV 侧重色彩；HOG/LBP 侧重形状和纹理。最终向量会进行归一化并拼接。"
        )
        method_hint.setObjectName("hintLabel")
        method_hint.setWordWrap(True)
        card.content_layout.addWidget(method_hint)
        return card

    def _create_metric_panel(self) -> QFrame:
        """Create feature-preview result metric cards."""
        panel = QFrame()
        panel.setObjectName("featureMetricPanel")
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)
        for index, metric in enumerate(
            (
                self._dimension_metric,
                self._dtype_metric,
                self._method_metric,
                self._time_metric,
            ),
        ):
            layout.addWidget(metric, index // 2, index % 2)
        return panel

    def _select_image(self) -> None:
        """Choose an image for feature extraction."""
        image_path = choose_image_file(
            self,
            self._controller.config.paths.dataset_root,
        )
        if image_path is None:
            return
        self._selected_image = image_path
        self._preview.set_image_path(image_path)
        self._status_label.setText(f"当前图片：{image_path.name}")
        self._output.setPlainText("已选择图片。点击“开始提取”查看特征向量摘要。")

    def _extract(self) -> None:
        """Run feature preview through the controller."""
        if self._selected_image is None:
            self.show_info("未选择图片", "请先选择一张真实艺术作品图片。")
            return
        methods = self._selected_methods()
        if not methods:
            self.show_info("未选择特征方法", "请至少勾选一种特征方法。")
            return
        self._set_busy(True, f"正在提取：{self._selected_image.name}")
        self.run_task(
            "特征提取失败",
            self._controller.extract_feature_preview,
            self._handle_success,
            self._finish_task,
            self._selected_image,
            self._selected_operations(),
            methods,
        )

    def _handle_success(self, result: object) -> None:
        """Render a completed feature preview."""
        if not isinstance(result, FeaturePreview):
            self.show_error("特征结果异常", "后端返回了无法识别的特征结果。")
            return
        methods = self._selected_methods()
        self._dimension_metric.set_display_value(f"{result.length:,}")
        self._dtype_metric.set_display_value(result.dtype_name)
        self._method_metric.set_display_value(" + ".join(self._selected_labels()))
        self._time_metric.set_display_value("单张预览完成")
        self._status_label.setText("特征提取完成，可进入 PCA 分析或模型训练流程。")
        self._output.setPlainText(
            "特征向量摘要\n"
            f"- 预处理：{', '.join(self._selected_operations())}\n"
            f"- 方法：{', '.join(methods)}\n"
            f"- 前 12 个数值：{result.first_values}\n\n"
            "说明：这里展示的是单张图片的特征预览，用于确认特征维数和数值范围；"
            "正式论文实验必须使用真实数据集批量提取后的特征矩阵。"
        )

    def _finish_task(self) -> None:
        """Restore feature extraction controls."""
        self._set_busy(False)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        """Toggle feature extraction controls while work is running."""
        self._extract_button.setEnabled(not busy)
        self._preset_combo.setEnabled(not busy)
        for checkbox in self._method_checks.values():
            checkbox.setEnabled(not busy)
        self._progress.setRange(0, 0 if busy else 1)
        self._progress.setValue(0 if busy else 1)
        self._progress.setVisible(busy)
        if message is not None:
            self._status_label.setText(message)
        if busy:
            self._time_metric.set_display_value("计算中")

    def _selected_operations(self) -> list[str]:
        """Return preprocessing operations for the selected preset."""
        preset = PREPROCESSING_PRESETS[self._preset_combo.currentIndex()]
        return list(preset.operations)

    def _selected_methods(self) -> list[str]:
        """Return selected backend feature method names."""
        return [
            option.method_name
            for option in FEATURE_METHODS
            if self._method_checks[option.method_name].isChecked()
        ]

    def _selected_labels(self) -> list[str]:
        """Return selected user-facing feature labels."""
        return [
            option.label
            for option in FEATURE_METHODS
            if self._method_checks[option.method_name].isChecked()
        ]

    def _update_preset_hint(self) -> None:
        """Update preprocessing preset description."""
        preset = PREPROCESSING_PRESETS[self._preset_combo.currentIndex()]
        self._preset_hint.setText(preset.description)

    def _refresh_method_metric(self) -> None:
        """Reflect currently selected methods in the metric card."""
        labels = self._selected_labels()
        self._method_metric.set_display_value(" + ".join(labels) if labels else "待选择")

    def _reset_result_state(self) -> None:
        """Reset metrics to useful non-zero empty-state copy."""
        self._dimension_metric.set_display_value("待提取")
        self._dtype_metric.set_display_value("待提取")
        self._refresh_method_metric()
        self._time_metric.set_display_value("单张预览")
        self._output.setPlainText(
            "请先选择图片，并勾选 RGB、HSV、HOG、LBP 中需要预览的特征。"
        )
