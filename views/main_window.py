"""Main application window for ArtInsight."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.navigation import NavigationRoute


NAVIGATION_GROUP_LABELS: dict[NavigationRoute, str] = {
    NavigationRoute.HOME: "工作台",
    NavigationRoute.DATASET: "01 数据准备",
    NavigationRoute.FEATURE_EXTRACTION: "02 特征与训练",
    NavigationRoute.MODEL_TRAINING: "02 特征与训练",
    NavigationRoute.PCA_ANALYSIS: "03 PCA分析",
    NavigationRoute.ALGORITHM_LAB: "03 PCA分析",
    NavigationRoute.ALGORITHM_EXPERIMENTS: "06 科研实验",
    NavigationRoute.UPLOAD: "04 应用演示",
    NavigationRoute.STYLE_CLASSIFICATION: "04 应用演示",
    NavigationRoute.SIMILARITY_RETRIEVAL: "04 应用演示",
    NavigationRoute.STYLE_MAP: "05 结果输出",
    NavigationRoute.STATISTICS: "05 结果输出",
    NavigationRoute.REPORT_EXPORT: "05 结果输出",
    NavigationRoute.SETTINGS: "系统",
}


class MainWindow(QMainWindow):
    """Desktop shell containing navigation and stacked content pages."""

    navigation_requested = Signal(str)

    def __init__(self) -> None:
        """Initialize the main window layout."""
        super().__init__()
        self._route_to_index: dict[NavigationRoute, int] = {}
        self._setup_window()
        self._setup_layout()

    def register_page(
        self,
        route: NavigationRoute,
        title: str,
        page: QWidget,
    ) -> None:
        """Register a page widget and matching navigation item."""
        if route in self._route_to_index:
            msg = f"Route has already been registered: {route.value}"
            raise ValueError(msg)

        group_label = NAVIGATION_GROUP_LABELS.get(route)
        if group_label is not None:
            self._add_group_if_needed(group_label)

        item = QListWidgetItem(title)
        item.setData(Qt.ItemDataRole.UserRole, route.value)
        self._navigation_list.addItem(item)

        page_index = self._content_stack.addWidget(page)
        self._route_to_index[route] = page_index

    def select_route(self, route: NavigationRoute) -> None:
        """Select a route in the navigation list."""
        for row in range(self._navigation_list.count()):
            item = self._navigation_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == route.value:
                self._navigation_list.setCurrentRow(row)
                return

    def display_page(self, route: NavigationRoute) -> None:
        """Display the stacked page associated with a route."""
        page_index = self._route_to_index.get(route)
        if page_index is None:
            return
        self._content_stack.setCurrentIndex(page_index)
        self._sync_navigation_selection(route)

    def _setup_window(self) -> None:
        """Apply basic main window properties."""
        self.setWindowTitle("ArtInsight")
        self.resize(1600, 900)
        self.setMinimumSize(1200, 720)

    def _setup_layout(self) -> None:
        """Build the main navigation and content layout."""
        central_widget = QWidget(self)
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar = self._create_sidebar()
        self._content_stack = QStackedWidget()
        self._content_stack.setObjectName("contentStack")

        root_layout.addWidget(sidebar)
        root_layout.addWidget(self._content_stack, stretch=1)
        self.setCentralWidget(central_widget)

    def _create_sidebar(self) -> QFrame:
        """Create the left navigation sidebar."""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(260)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(18)

        title = QLabel("ArtInsight")
        title.setObjectName("appTitle")
        subtitle = QLabel("基于PCA的艺术风格分析")
        subtitle.setObjectName("appSubtitle")

        self._navigation_list = QListWidget()
        self._navigation_list.setObjectName("navigationList")
        self._navigation_list.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._navigation_list.currentItemChanged.connect(
            self._emit_navigation_request,
        )

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self._navigation_list)
        return sidebar

    def _emit_navigation_request(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        """Emit a route value when the selected navigation item changes."""
        del previous
        if current is None:
            return
        route_value = current.data(Qt.ItemDataRole.UserRole)
        if isinstance(route_value, str):
            self.navigation_requested.emit(route_value)

    def _sync_navigation_selection(self, route: NavigationRoute) -> None:
        """Keep the sidebar selection in sync with programmatic navigation."""
        for row in range(self._navigation_list.count()):
            item = self._navigation_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == route.value:
                if self._navigation_list.currentRow() != row:
                    self._navigation_list.blockSignals(True)
                    self._navigation_list.setCurrentRow(row)
                    self._navigation_list.blockSignals(False)
                return

    def _add_group_if_needed(self, group_label: str) -> None:
        """Insert a disabled group label before a new navigation section."""
        if getattr(self, "_last_group_label", None) == group_label:
            return
        self._last_group_label = group_label
        item = QListWidgetItem(group_label)
        item.setData(Qt.ItemDataRole.UserRole, None)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        self._navigation_list.addItem(item)
