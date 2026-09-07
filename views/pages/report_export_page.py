"""Report export page."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from controllers.gui_dtos import ExportResult
from controllers.page_data_controller import PageDataController
from views.dashboard_widgets import DashboardCard
from views.pages.base_page import BasePage
from views.ui_helpers import create_button, create_panel
from views.widgets import button_row


class ReportExportPage(BasePage):
    """Export a project summary report."""

    page_title = "报告导出"
    page_subtitle = "Step 10：生成符合论文实验记录要求的PDF或HTML报告。"

    def __init__(self, controller: PageDataController) -> None:
        """Create report export controls."""
        super().__init__(controller)
        self._export_button: QPushButton
        self._name_edit = QLineEdit("ArtInsight_Report")
        self._format_combo = QComboBox()
        self._format_combo.addItems(["PDF", "HTML"])
        self._status_label = QLabel("报告将自动汇总数据集、PCA结果、图表与实验结论。")
        self._status_label.setObjectName("hintLabel")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build report form and preview."""
        self._export_button = create_button("导出报告", "save", primary=True)
        self._export_button.clicked.connect(self._export)
        self._root_layout.addWidget(button_row(self._export_button))

        row = QHBoxLayout()
        row.setSpacing(16)
        row.addWidget(self._create_form_panel(), stretch=2)
        row.addWidget(self._create_preview_panel(), stretch=3)
        self._root_layout.addLayout(row, stretch=1)

    def _create_form_panel(self) -> QFrame:
        """Create export configuration controls."""
        panel = create_panel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        title = QLabel("导出配置")
        title.setObjectName("dashboardCardTitle")
        form = QFormLayout()
        form.addRow("报告名称", self._name_edit)
        form.addRow("导出格式", self._format_combo)
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        return panel

    def _create_preview_panel(self) -> QFrame:
        """Create a static paper-style report structure preview."""
        panel = DashboardCard("论文报告结构预览", "导出文件将按固定模板组织，便于答辩与论文实验复现。")
        sections = [
            ("01", "Dataset Description", "数据来源、类别分布、清洗规则"),
            ("02", "PCA Configuration", "主成分数量、解释方差、重建误差"),
            ("03", "Visualization Figures", "Scree Plot、PCA二维/三维、统计图"),
            ("04", "Classification & Retrieval", "分类结果、相似检索样例与指标摘要"),
            ("05", "Conclusion", "实验结论、局限性与可复现说明"),
        ]
        for index, title, detail in sections:
            panel.content_layout.addWidget(self._create_section_row(index, title, detail))
        return panel

    def _create_section_row(self, index: str, title: str, detail: str) -> QFrame:
        """Create one report-preview section row."""
        row = QFrame()
        row.setObjectName("reportPreviewRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)
        index_label = QLabel(index)
        index_label.setObjectName("reportPreviewIndex")
        text_box = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        detail_label = QLabel(detail)
        detail_label.setObjectName("hintLabel")
        detail_label.setWordWrap(True)
        text_box.addWidget(title_label)
        text_box.addWidget(detail_label)
        layout.addWidget(index_label)
        layout.addLayout(text_box, stretch=1)
        return row

    def _export(self) -> None:
        """Export a report through the page controller."""
        report_name = self._name_edit.text().strip()
        if not report_name:
            self.show_info("报告名称为空", "请输入报告名称后再导出。")
            return
        self._export_button.setEnabled(False)
        self._status_label.setText("正在导出报告...")
        self.run_task(
            "报告导出失败",
            self._controller.export_report,
            self._handle_success,
            self._finish_task,
            report_name,
            self._format_combo.currentText(),
        )

    def _handle_success(self, result: object) -> None:
        """Render a completed report export result."""
        if not isinstance(result, ExportResult):
            self.show_error("报告结果异常", "后端返回了无法识别的报告导出结果。")
            return
        self._status_label.setText(f"{result.message} {result.output_path}")
        self.show_info("导出完成", f"报告路径：{result.output_path}")

    def _finish_task(self) -> None:
        """Restore export controls."""
        self._export_button.setEnabled(True)
