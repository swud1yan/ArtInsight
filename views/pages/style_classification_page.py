"""Style classification page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QProgressBar, QPushButton

from app.navigation import NavigationRoute
from controllers.gui_dtos import ArtworkCard, ClassificationViewResult
from controllers.page_data_controller import PageDataController
from services.style_translations import format_style_label
from views.image_workflow_widgets import ImageAnalysisWorkspace, StyleSimilarityChart
from views.pages.base_page import BasePage
from views.ui_helpers import choose_image_file, create_button
from views.widgets import button_row, show_artwork_detail


class StyleClassificationPage(BasePage):
    """Classify an uploaded artwork and show same-style examples."""

    page_title = "风格分类"
    page_subtitle = "Step 6：使用已训练的标准化、自研PCA与KNN模型预测艺术风格。"

    def __init__(self, controller: PageDataController) -> None:
        """Create style classification widgets."""
        super().__init__(controller)
        self._selected_image: Path | None = None
        self._classify_button: QPushButton
        self._confidence_bar = QProgressBar()
        self._confidence_bar.setRange(0, 100)
        self._style_chart = StyleSimilarityChart(max_rows=8)
        self._workspace = ImageAnalysisWorkspace(
            result_title="分类结果",
            empty_message="完成分类后将显示同类别示例作品。",
            columns=4,
        )
        self._workspace.add_result_widget(self._confidence_bar)
        self._workspace.add_result_widget(self._style_chart)
        self._workspace.grid.artwork_selected.connect(self._open_artwork_detail)
        self._cards_by_id: dict[int, ArtworkCard] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build classification controls."""
        select_button = create_button("选择图片", "image", primary=True)
        self._classify_button = create_button("开始分类", "play")
        select_button.clicked.connect(self._select_image)
        self._classify_button.clicked.connect(self._classify)
        self._root_layout.addWidget(button_row(select_button, self._classify_button))
        self._root_layout.addWidget(self._workspace, stretch=1)

    def _select_image(self) -> None:
        """Choose an image to classify."""
        image_path = choose_image_file(
            self,
            self._controller.config.paths.dataset_root,
        )
        if image_path is None:
            return
        self._selected_image = image_path
        self._workspace.set_image_path(image_path)
        self._workspace.set_status(f"已选择：{image_path.name}")
        self._confidence_bar.setValue(0)
        self._style_chart.set_scores([])
        self._workspace.clear_results()

    def _classify(self) -> None:
        """Run style classification for the selected image."""
        if self._selected_image is None:
            self.show_info("未选择图片", "请先选择一张图片。")
            return
        self._classify_button.setEnabled(False)
        self._workspace.set_status("正在分类，请稍候...")
        self.run_task(
            "风格分类失败",
            self._controller.classify_image,
            self._handle_success,
            self._finish_task,
            self._selected_image,
            on_failure=self._handle_failure,
        )

    def _handle_success(self, result: object) -> None:
        """Render a completed classification result."""
        if not isinstance(result, ClassificationViewResult):
            self.show_error("分类结果异常", "后端返回了无法识别的分类结果。")
            return
        confidence = int(round(result.confidence * 100))
        self._confidence_bar.setValue(confidence)
        self._workspace.set_status(
            f"预测风格：{format_style_label(result.style_label)}，置信度：{confidence}%",
        )
        self._style_chart.set_scores(result.style_scores)
        self._cards_by_id = {card.artwork_id: card for card in result.examples}
        self._workspace.set_cards([(card, "同类别作品") for card in result.examples])

    def _finish_task(self) -> None:
        """Restore classification controls."""
        self._classify_button.setEnabled(True)

    def _handle_failure(self, error: Exception) -> None:
        """Guide users to model training when no artifact exists."""
        if isinstance(error, FileNotFoundError):
            self._workspace.set_status("尚未训练风格分类模型，请先完成“模型训练”。")
            self.request_navigation(NavigationRoute.MODEL_TRAINING)

    def _open_artwork_detail(self, artwork_id: int) -> None:
        """Open the selected example artwork detail dialog."""
        card = self._cards_by_id.get(artwork_id)
        if card is not None:
            show_artwork_detail(self, card)
