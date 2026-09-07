"""Similar artwork retrieval page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSpinBox, QWidget

from app.navigation import NavigationRoute
from controllers.gui_dtos import ArtworkCard, SimilarArtworkViewResult
from controllers.page_data_controller import PageDataController
from views.image_workflow_widgets import ImageAnalysisWorkspace
from views.pages.base_page import BasePage
from views.ui_helpers import choose_image_file, create_button
from views.widgets import button_row, show_artwork_detail


class SimilarityRetrievalPage(BasePage):
    """Retrieve visually similar artworks for an uploaded image."""

    page_title = "相似检索"
    page_subtitle = "Step 7：在PCA空间执行近邻检索，展示Top-N相似作品缩略图。"

    def __init__(self, controller: PageDataController) -> None:
        """Create retrieval widgets."""
        super().__init__(controller)
        self._selected_image: Path | None = None
        self._retrieve_button: QPushButton
        self._top_n_spin = QSpinBox()
        self._top_n_spin.setRange(1, 50)
        self._top_n_spin.setValue(10)
        self._workspace = ImageAnalysisWorkspace(
            result_title="检索结果",
            empty_message="完成检索后将显示Top-N相似作品。",
            columns=5,
        )
        self._workspace.add_result_widget(self._create_top_n_row())
        self._workspace.grid.artwork_selected.connect(self._open_artwork_detail)
        self._cards_by_id: dict[int, ArtworkCard] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build similar retrieval controls."""
        select_button = create_button("选择图片", "image", primary=True)
        self._retrieve_button = create_button("检索", "search")
        select_button.clicked.connect(self._select_image)
        self._retrieve_button.clicked.connect(self._retrieve)
        self._root_layout.addWidget(button_row(select_button, self._retrieve_button))
        self._root_layout.addWidget(self._workspace, stretch=1)

    def _select_image(self) -> None:
        """Choose an image to retrieve against."""
        image_path = choose_image_file(
            self,
            self._controller.config.paths.dataset_root,
        )
        if image_path is None:
            return
        self._selected_image = image_path
        self._workspace.set_image_path(image_path)
        self._workspace.set_status(f"已选择：{image_path.name}")
        self._workspace.clear_results()

    def _retrieve(self) -> None:
        """Run Top-N similar artwork retrieval."""
        if self._selected_image is None:
            self.show_info("未选择图片", "请先选择一张图片。")
            return
        self._retrieve_button.setEnabled(False)
        self._workspace.set_status("正在检索，请稍候...")
        self.run_task(
            "相似检索失败",
            self._controller.retrieve_similar_image,
            self._handle_success,
            self._finish_task,
            self._selected_image,
            top_n=self._top_n_spin.value(),
            on_failure=self._handle_failure,
        )

    def _handle_success(self, result: object) -> None:
        """Render completed retrieval results."""
        if not self._is_retrieval_result_list(result):
            self.show_error("检索结果异常", "后端返回了无法识别的检索结果。")
            return
        results = list(result)
        self._cards_by_id = {
            item.artwork.artwork_id: item.artwork
            for item in results
        }
        self._workspace.set_cards(
            [(item.artwork, f"距离：{item.distance:.4f}") for item in results],
        )
        self._workspace.set_status(f"已返回 {len(results)} 个结果。")

    def _finish_task(self) -> None:
        """Restore retrieval controls."""
        self._retrieve_button.setEnabled(True)

    def _handle_failure(self, error: Exception) -> None:
        """Guide users to model training when no artifact exists."""
        if isinstance(error, FileNotFoundError):
            self._workspace.set_status("尚未训练检索模型，请先完成“模型训练”。")
            self.request_navigation(NavigationRoute.MODEL_TRAINING)

    def _create_top_n_row(self) -> QWidget:
        """Create the compact Top-N selector."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QLabel("返回数量")
        label.setObjectName("hintLabel")
        layout.addWidget(label)
        layout.addWidget(self._top_n_spin)
        layout.addStretch(1)
        return row

    def _is_retrieval_result_list(self, result: object) -> bool:
        """Return whether an object is a list of retrieval results."""
        return isinstance(result, list) and all(
            isinstance(item, SimilarArtworkViewResult) for item in result
        )

    def _open_artwork_detail(self, artwork_id: int) -> None:
        """Open a dialog with large preview and metadata."""
        card = self._cards_by_id.get(artwork_id)
        if card is not None:
            show_artwork_detail(self, card)
