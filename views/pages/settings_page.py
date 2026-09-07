"""Application settings page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QFormLayout, QLineEdit, QPushButton, QVBoxLayout

from controllers.page_data_controller import PageDataController
from views.pages.base_page import BasePage
from views.ui_helpers import choose_folder, create_button, create_panel
from views.widgets import button_row


class SettingsPage(BasePage):
    """Edit persistent application configuration values."""

    page_title = "设置"
    page_subtitle = "修改数据库、数据目录、模型目录等配置；保存后重启应用生效。"

    def __init__(self, controller: PageDataController) -> None:
        """Create settings form widgets."""
        super().__init__(controller)
        config = controller.config
        self._dataset_root_edit = QLineEdit(str(config.paths.dataset_root))
        self._cache_dir_edit = QLineEdit(str(config.paths.cache_dir))
        self._model_dir_edit = QLineEdit(str(config.paths.model_dir))
        self._export_dir_edit = QLineEdit(str(config.paths.export_dir))
        self._database_url_edit = QLineEdit(config.database.url)
        self._save_button: QPushButton
        self._status_label = QLabel("配置修改后保存，重启应用后生效。")
        self._status_label.setObjectName("hintLabel")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build editable settings form."""
        self._save_button = create_button("保存配置", "save", primary=True)
        dataset_button = create_button("选择数据目录", "folder")
        cache_button = create_button("选择缓存目录", "folder")
        model_button = create_button("选择模型目录", "folder")
        export_button = create_button("选择导出目录", "folder")
        self._save_button.clicked.connect(self._save)
        dataset_button.clicked.connect(
            lambda: self._choose_path(self._dataset_root_edit),
        )
        cache_button.clicked.connect(lambda: self._choose_path(self._cache_dir_edit))
        model_button.clicked.connect(lambda: self._choose_path(self._model_dir_edit))
        export_button.clicked.connect(lambda: self._choose_path(self._export_dir_edit))
        self._root_layout.addWidget(
            button_row(
                self._save_button,
                dataset_button,
                cache_button,
                model_button,
                export_button,
            ),
        )

        panel = create_panel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        form = QFormLayout()
        form.addRow("数据集根目录", self._dataset_root_edit)
        form.addRow("缓存目录", self._cache_dir_edit)
        form.addRow("模型目录", self._model_dir_edit)
        form.addRow("报告导出目录", self._export_dir_edit)
        form.addRow("数据库URL", self._database_url_edit)
        layout.addLayout(form)
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        self._root_layout.addWidget(panel, stretch=1)

    def _choose_path(self, target: QLineEdit) -> None:
        """Choose a folder and write it into a line edit."""
        current = Path(target.text()).expanduser()
        folder = choose_folder(self, current)
        if folder is not None:
            target.setText(str(folder))

    def _save(self) -> None:
        """Persist editable configuration values."""
        self._save_button.setEnabled(False)
        self._status_label.setText("正在保存配置...")
        self.run_task(
            "配置保存失败",
            self._controller.save_config_values,
            self._handle_save_success,
            self._finish_save,
            self._editable_values(),
        )

    def _editable_values(self) -> dict[str, str]:
        """Collect editable settings from the form."""
        return {
            "dataset_root": self._dataset_root_edit.text(),
            "cache_dir": self._cache_dir_edit.text(),
            "model_dir": self._model_dir_edit.text(),
            "export_dir": self._export_dir_edit.text(),
            "database_url": self._database_url_edit.text(),
        }

    def _handle_save_success(self, result: object) -> None:
        """Show a success message after configuration is saved."""
        del result
        self._status_label.setText("配置已保存，重启应用后生效。")
        self.show_info("保存成功", "配置已保存，重启应用后生效。")

    def _finish_save(self) -> None:
        """Restore save button state after configuration persistence."""
        self._save_button.setEnabled(True)
