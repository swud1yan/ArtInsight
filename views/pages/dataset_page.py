"""Dataset management page."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
)

from controllers.gui_dtos import (
    DatasetImportViewResult,
    DatasetOperationResult,
    DatasetRow,
)
from controllers.page_data_controller import PageDataController
from views.dashboard_widgets import EmptyActionPanel, OverviewMetricWidget
from views.pages.base_page import BasePage
from views.table_models import ReadOnlyTableModel, TableColumn
from views.ui_helpers import choose_folder, configure_table, create_button, create_panel
from views.widgets import button_row
from views.zoom_dialogs import show_enlarged_table


class DatasetPage(BasePage):
    """Manage imported datasets and their database records."""

    page_title = "数据集管理"
    page_subtitle = "Step 1-2：导入真实艺术数据集，检查规模、类别和路径状态。"

    def __init__(self, controller: PageDataController) -> None:
        """Create dataset management widgets."""
        super().__init__(controller)
        self._import_wikiart_button: QPushButton
        self._import_kaggle_button: QPushButton
        self._delete_button: QPushButton
        self._refresh_button: QPushButton
        self._zoom_table_button: QPushButton
        self._enable_selected_button: QPushButton
        self._disable_selected_button: QPushButton
        self._only_selected_button: QPushButton
        self._enable_all_button: QPushButton
        self._table_model = self._create_table_model()
        self._status_label = QLabel("请先导入WikiArt数据集。")
        self._status_label.setObjectName("hintLabel")
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setTextVisible(False)
        self._progress.hide()
        self._content_stack = QStackedWidget()
        self._empty_panel = EmptyActionPanel(
            "请先导入WikiArt数据集。ArtInsight 需要真实艺术图像才能进行论文级实验。",
            "导入WikiArt数据集",
        )
        self._setup_ui()
        QTimer.singleShot(0, self.refresh)

    def _setup_ui(self) -> None:
        """Build dataset management controls."""
        self._import_wikiart_button = create_button(
            "导入WikiArt数据集",
            "folder",
            primary=True,
        )
        self._import_kaggle_button = create_button("导入Kaggle数据集", "folder")
        self._refresh_button = create_button("刷新", "refresh")
        self._zoom_table_button = create_button("放大表格", "search")
        self._delete_button = create_button("删除所选", "settings", danger=True)
        self._enable_selected_button = create_button("启用所选", "play", primary=True)
        self._disable_selected_button = create_button("禁用所选", "settings")
        self._only_selected_button = create_button("仅启用所选", "search")
        self._enable_all_button = create_button("全部启用", "refresh")
        self._import_wikiart_button.clicked.connect(self._import_dataset)
        self._import_kaggle_button.clicked.connect(self._import_dataset)
        self._empty_panel.button.clicked.connect(self._import_dataset)
        self._refresh_button.clicked.connect(self.refresh)
        self._zoom_table_button.clicked.connect(
            lambda: show_enlarged_table(self, self._table_view, "数据集列表"),
        )
        self._delete_button.clicked.connect(self._delete_selected_dataset)
        self._enable_selected_button.clicked.connect(
            lambda: self._set_selected_dataset_scope(True),
        )
        self._disable_selected_button.clicked.connect(
            lambda: self._set_selected_dataset_scope(False),
        )
        self._only_selected_button.clicked.connect(self._set_only_selected_dataset_scope)
        self._enable_all_button.clicked.connect(self._enable_all_datasets)
        self._root_layout.addWidget(
            button_row(
                self._import_wikiart_button,
                self._import_kaggle_button,
                self._refresh_button,
            ),
        )
        self._root_layout.addWidget(
            button_row(
                self._enable_selected_button,
                self._disable_selected_button,
                self._only_selected_button,
                self._enable_all_button,
                self._zoom_table_button,
                self._delete_button,
            ),
        )
        self._root_layout.addWidget(self._create_overview_panel())
        self._root_layout.addWidget(self._create_content_panel(), stretch=1)

    def refresh(self) -> None:
        """Reload dataset rows from the database."""
        self._set_busy(True, "正在刷新数据集状态...")
        self.run_task(
            "数据集列表刷新失败",
            self._load_rows,
            self._handle_operation_success,
            self._finish_task,
        )

    def _import_dataset(self) -> None:
        """Ask for a folder and import it through the controller."""
        folder = choose_folder(self, self._controller.config.paths.dataset_root)
        if folder is None:
            return
        self._set_busy(True, f"正在导入：{folder}")
        self.run_task(
            "数据集导入失败",
            self._import_and_reload,
            self._handle_operation_success,
            self._finish_task,
            str(folder),
        )

    def _delete_selected_dataset(self) -> None:
        """Delete the selected dataset after user confirmation."""
        row = self._selected_dataset_row()
        if row is None:
            self.show_info("未选择数据集", "请先在数据集列表中选择一项。")
            return
        message = (
            f"确定删除数据集“{row.name}”吗？\n\n"
            "这会删除数据库记录，并删除已登记的原始图片文件。此操作不可恢复。"
        )
        reply = QMessageBox.question(self, "确认删除", message)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._set_busy(True, f"正在删除：{row.name}")
        self.run_task(
            "数据集删除失败",
            self._delete_and_reload,
            self._handle_operation_success,
            self._finish_task,
            row.dataset_id,
        )

    def _set_selected_dataset_scope(self, enabled: bool) -> None:
        """Enable or disable the selected dataset."""
        rows = self._selected_dataset_rows()
        if not rows:
            self.show_info("未选择数据集", "请先选择一个或多个数据集。")
            return
        action = "启用" if enabled else "禁用"
        self._set_busy(True, f"正在{action}所选数据集...")
        self.run_task(
            "数据集实验范围更新失败",
            self._set_many_scope_and_reload,
            self._handle_operation_success,
            self._finish_task,
            [row.dataset_id for row in rows],
            enabled,
            f"已{action} {len(rows)} 个数据集。",
        )

    def _set_only_selected_dataset_scope(self) -> None:
        """Make the selected datasets the only active experiment scope."""
        rows = self._selected_dataset_rows()
        if not rows:
            self.show_info("未选择数据集", "请先选择一个或多个数据集。")
            return
        names = "、".join(row.name for row in rows[:3])
        if len(rows) > 3:
            names += f" 等 {len(rows)} 个"
        reply = QMessageBox.question(
            self,
            "确认实验范围",
            f"将只启用“{names}”，其它数据集会保留但不参与后续分析。是否继续？",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._set_busy(True, "正在设置当前实验范围...")
        self.run_task(
            "数据集实验范围更新失败",
            self._set_exclusive_scope_and_reload,
            self._handle_operation_success,
            self._finish_task,
            [row.dataset_id for row in rows],
        )

    def _enable_all_datasets(self) -> None:
        """Enable all imported datasets."""
        rows = self._table_model.rows()
        if not rows:
            self.show_info("没有数据集", "请先导入数据集。")
            return
        self._set_busy(True, "正在启用全部数据集...")
        self.run_task(
            "数据集实验范围更新失败",
            self._set_many_scope_and_reload,
            self._handle_operation_success,
            self._finish_task,
            [row.dataset_id for row in rows],
            True,
            f"已启用全部 {len(rows)} 个数据集。",
        )

    def _create_overview_panel(self) -> object:
        """Create compact dataset statistics cards."""
        panel = create_panel()
        layout = QGridLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setHorizontalSpacing(12)
        self._dataset_metric = OverviewMetricWidget("数据集")
        self._image_metric = OverviewMetricWidget("总图片")
        self._style_metric = OverviewMetricWidget("类别数")
        self._artist_metric = OverviewMetricWidget("作者数")
        for index, metric in enumerate(
            (
                self._dataset_metric,
                self._image_metric,
                self._style_metric,
                self._artist_metric,
            ),
        ):
            layout.addWidget(metric, 0, index)
        return panel

    def _create_content_panel(self) -> object:
        """Create empty-state and table-state panels."""
        panel = create_panel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        self._table_view = QTableView()
        configure_table(self._table_view)
        self._table_view.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self._table_view.setModel(self._table_model)
        self._table_view.setColumnWidth(0, 92)
        self._table_view.setColumnWidth(1, 72)
        self._content_stack.addWidget(self._empty_panel)
        self._content_stack.addWidget(self._table_view)
        layout.addWidget(self._content_stack, stretch=1)
        status_layout = QHBoxLayout()
        status_layout.addWidget(self._status_label, stretch=1)
        status_layout.addWidget(self._progress)
        layout.addLayout(status_layout)
        return panel

    def _load_rows(self) -> DatasetOperationResult:
        """Load dataset rows in a background thread."""
        return DatasetOperationResult(rows=self._controller.list_datasets())

    def _import_and_reload(self, dataset_path: str) -> DatasetOperationResult:
        """Import a dataset and load refreshed rows in one background task."""
        result = self._controller.import_dataset(dataset_path)
        if not isinstance(result, DatasetImportViewResult):
            msg = "后端返回了无法识别的数据集导入结果。"
            raise TypeError(msg)
        message = (
            f"已导入 {result.imported_count} 张图片，"
            f"跳过 {result.skipped_count} 个文件。"
        )
        return DatasetOperationResult(
            rows=self._controller.list_datasets(),
            message=message,
        )

    def _delete_and_reload(self, dataset_id: int) -> DatasetOperationResult:
        """Delete a dataset and load refreshed rows in one background task."""
        self._controller.delete_dataset(dataset_id)
        return DatasetOperationResult(
            rows=self._controller.list_datasets(),
            message="数据集已删除。",
        )

    def _set_many_scope_and_reload(
        self,
        dataset_ids: list[int],
        enabled: bool,
        message: str,
    ) -> DatasetOperationResult:
        """Persist participation for multiple datasets and reload table rows."""
        self._controller.set_dataset_enabled_many(dataset_ids, enabled)
        return DatasetOperationResult(
            rows=self._controller.list_datasets(),
            message=message,
        )

    def _set_exclusive_scope_and_reload(
        self,
        dataset_ids: list[int],
    ) -> DatasetOperationResult:
        """Persist an exclusive experiment scope and reload rows."""
        self._controller.set_exclusive_enabled_datasets(dataset_ids)
        return DatasetOperationResult(
            rows=self._controller.list_datasets(),
            message=f"当前实验范围已更新：启用 {len(dataset_ids)} 个数据集。",
        )

    def _handle_operation_success(self, result: object) -> None:
        """Render refreshed dataset rows after a background task."""
        if not isinstance(result, DatasetOperationResult):
            self.show_error("数据集结果异常", "后端返回了无法识别的数据集结果。")
            return
        self._table_model.set_rows(result.rows)
        self._render_summary(result.rows)
        if result.message:
            self.show_info("操作完成", result.message)

    def _finish_task(self) -> None:
        """Restore controls after background dataset operation finishes."""
        self._set_busy(False)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        """Toggle buttons and progress while dataset work is running."""
        self._progress.setRange(0, 0 if busy else 1)
        self._progress.setValue(0 if busy else 1)
        self._progress.setVisible(busy)
        for button in (
            self._import_wikiart_button,
            self._import_kaggle_button,
            self._refresh_button,
            self._zoom_table_button,
            self._delete_button,
            self._enable_selected_button,
            self._disable_selected_button,
            self._only_selected_button,
            self._enable_all_button,
            self._empty_panel.button,
        ):
            button.setEnabled(not busy)
        if hasattr(self, "_table_view"):
            self._table_view.setEnabled(not busy)
        if message is not None:
            self._status_label.setText(message)

    def _render_summary(self, rows: list[DatasetRow]) -> None:
        """Render table visibility and overview metrics."""
        dataset_count = len(rows)
        active_rows = [row for row in rows if row.is_enabled]
        active_count = len(active_rows)
        image_count = sum(row.image_count for row in active_rows)
        style_count = sum(row.style_count for row in active_rows)
        artist_count = sum(row.artist_count for row in active_rows)
        self._dataset_metric.set_display_value(
            f"{active_count}/{dataset_count}" if dataset_count else "未导入",
        )
        self._image_metric.set_display_value(f"{image_count:,}" if image_count else "待导入")
        self._style_metric.set_display_value(f"{style_count:,}" if style_count else "待检查")
        self._artist_metric.set_display_value(f"{artist_count:,}" if artist_count else "待检查")
        if rows:
            self._content_stack.setCurrentWidget(self._table_view)
            self._status_label.setText(
                f"已登记 {dataset_count} 个数据集，当前启用 {active_count} 个。"
                "后续特征提取、模型训练、PCA、统计和报告只使用启用的数据集。"
            )
        else:
            self._content_stack.setCurrentWidget(self._empty_panel)
            self._status_label.setText("请先导入WikiArt数据集。")

    def _selected_dataset_row(self) -> DatasetRow | None:
        """Return the currently selected dataset row."""
        indexes = self._table_view.selectionModel().selectedRows()
        if not indexes:
            return None
        return self._table_model.row_at(indexes[0].row())

    def _selected_dataset_rows(self) -> list[DatasetRow]:
        """Return all selected dataset rows."""
        selection_model = self._table_view.selectionModel()
        if selection_model is None:
            return []
        rows: list[DatasetRow] = []
        for index in selection_model.selectedRows():
            row = self._table_model.row_at(index.row())
            if row is not None:
                rows.append(row)
        return rows

    def _create_table_model(self) -> ReadOnlyTableModel[DatasetRow]:
        """Create the dataset table model."""
        return ReadOnlyTableModel[DatasetRow](
            [
                TableColumn(
                    "实验状态",
                    lambda row: "● 启用" if row.is_enabled else "○ 禁用",
                    Qt.AlignCenter,
                ),
                TableColumn("ID", lambda row: row.dataset_id),
                TableColumn("名称", lambda row: row.name),
                TableColumn("图片", lambda row: row.image_count, Qt.AlignRight),
                TableColumn("风格", lambda row: row.style_count, Qt.AlignRight),
                TableColumn("作者", lambda row: row.artist_count, Qt.AlignRight),
                TableColumn("导入时间", lambda row: row.imported_at),
                TableColumn("路径", lambda row: row.path),
            ],
        )
