"""Image upload page with quick classification and retrieval."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QPushButton

from app.navigation import NavigationRoute
from controllers.gui_dtos import (
    ArtworkCard,
    ClassificationViewResult,
    SimilarArtworkViewResult,
)
from controllers.page_data_controller import PageDataController
from services.style_translations import format_style_label
from views.image_workflow_widgets import ImageAnalysisWorkspace, StyleSimilarityChart
from views.pages.base_page import BasePage
from views.ui_helpers import choose_image_file, create_button
from views.widgets import button_row, show_artwork_detail


class UploadPage(BasePage):
    """Upload one artwork image and run common analysis actions."""

    page_title = "图片上传"
    page_subtitle = "Step 6-7：模型训练完成后，选择单张作品图片快速查看风格分类与相似作品。"

    def __init__(self, controller: PageDataController) -> None:
        """Create the upload workflow page."""
        super().__init__(controller)
        self._selected_image: Path | None = None
        self._classify_button: QPushButton
        self._retrieve_button: QPushButton
        self._style_chart = StyleSimilarityChart(max_rows=8)
        self._workspace = ImageAnalysisWorkspace(
            result_title="分析结果",
            empty_message="选择图片后可查看分类结果和相似作品。",
            columns=5,
        )
        self._workspace.add_result_widget(self._style_chart)
        self._workspace.grid.artwork_selected.connect(self._open_artwork_detail)
        self._cards_by_id: dict[int, ArtworkCard] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build upload controls and result area."""
        select_button = create_button("选择图片", "image", primary=True)
        self._classify_button = create_button("分类", "play")
        self._retrieve_button = create_button("检索相似作品", "search")
        select_button.clicked.connect(self._select_image)
        self._classify_button.clicked.connect(self._classify)
        self._retrieve_button.clicked.connect(self._retrieve)
        self._root_layout.addWidget(
            button_row(select_button, self._classify_button, self._retrieve_button),
        )
        self._root_layout.addWidget(self._workspace, stretch=1)

    def _select_image(self) -> None:
        """Open a file dialog and update the preview."""
        image_path = choose_image_file(
            self,
            self._controller.config.paths.dataset_root,
        )
        if image_path is None:
            return
        self._selected_image = image_path
        self._workspace.set_image_path(image_path)
        self._workspace.set_status(f"已选择：{image_path.name}")
        self._style_chart.set_scores([])
        self._workspace.clear_results()

    def _classify(self) -> None:
        """Classify the selected image in a background task."""
        if self._selected_image is None:
            self.show_info("未选择图片", "请先选择一张图片。")
            return
        self._set_busy(True, "正在分类，请稍候...")
        self.run_task(
            "分类失败",
            self._controller.classify_image,
            self._handle_classification_success,
            self._finish_task,
            self._selected_image,
            on_failure=self._handle_failure,
        )

    def _retrieve(self) -> None:
        """Retrieve similar artworks for the selected image in the background."""
        if self._selected_image is None:
            self.show_info("未选择图片", "请先选择一张图片。")
            return
        self._set_busy(True, "正在检索相似作品，请稍候...")
        self.run_task(
            "相似检索失败",
            self._controller.retrieve_similar_image,
            self._handle_retrieval_success,
            self._finish_task,
            self._selected_image,
            on_failure=self._handle_failure,
        )

    def _handle_classification_success(self, result: object) -> None:
        """Render a completed classification result."""
        if not isinstance(result, ClassificationViewResult):
            self.show_error("分类结果异常", "后端返回了无法识别的分类结果。")
            return
        confidence = result.confidence * 100
        self._workspace.set_status(
            f"预测风格：{format_style_label(result.style_label)}，置信度：{confidence:.2f}%",
        )
        self._style_chart.set_scores(result.style_scores)
        self._remember_cards(result.examples)
        self._workspace.set_cards([(card, "同风格样例") for card in result.examples])

    def _handle_retrieval_success(self, result: object) -> None:
        """Render completed retrieval results."""
        if not self._is_retrieval_result_list(result):
            self.show_error("检索结果异常", "后端返回了无法识别的检索结果。")
            return
        results = list(result)
        self._remember_cards([item.artwork for item in results])
        self._workspace.set_cards(
            [(item.artwork, f"距离：{item.distance:.4f}") for item in results],
        )
        self._workspace.set_status(f"已返回 {len(results)} 个相似作品。")

    def _finish_task(self) -> None:
        """Restore controls after a background task finishes."""
        self._set_busy(False)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        """Toggle action buttons while a backend task is running."""
        self._classify_button.setEnabled(not busy)
        self._retrieve_button.setEnabled(not busy)
        if message:
            self._workspace.set_status(message)

    def _handle_failure(self, error: Exception) -> None:
        """Guide users to model training when no artifact exists."""
        if isinstance(error, FileNotFoundError):
            self._workspace.set_status("尚未训练模型，请先完成“模型训练”。")
            self.request_navigation(NavigationRoute.MODEL_TRAINING)

    def _remember_cards(self, cards: list[ArtworkCard]) -> None:
        """Cache artwork cards by id for detail dialogs."""
        self._cards_by_id = {card.artwork_id: card for card in cards}

    def _is_retrieval_result_list(self, result: object) -> bool:
        """Return whether an object is a list of retrieval results."""
        return isinstance(result, list) and all(
            isinstance(item, SimilarArtworkViewResult) for item in result
        )

    def _open_artwork_detail(self, artwork_id: int) -> None:
        """Show a detail dialog for a cached artwork."""
        card = self._cards_by_id.get(artwork_id)
        if card is not None:
            show_artwork_detail(self, card)
