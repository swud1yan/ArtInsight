"""Common base class for ArtInsight feature pages."""

from __future__ import annotations

from collections.abc import Callable
import logging

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFrame, QMessageBox, QScrollArea, QVBoxLayout, QWidget

from app.navigation import NavigationRoute
from controllers.page_data_controller import PageDataController
from views.widgets import PageHeader
from views.workers import run_in_thread

LOGGER = logging.getLogger(__name__)


class BasePage(QWidget):
    """Base page that owns a controller reference and a common root layout."""

    navigation_requested = Signal(str)

    page_title = "页面"
    page_subtitle = ""

    def __init__(self, controller: PageDataController) -> None:
        """Create a page using the shared page-data controller."""
        super().__init__()
        self._controller = controller
        self._worker_threads: list[QThread] = []
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self._scroll_area = QScrollArea()
        self._scroll_area.setObjectName("pageScrollArea")
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._content_widget = QWidget()
        self._content_widget.setObjectName("pageContent")
        self._root_layout = QVBoxLayout(self._content_widget)
        self._root_layout.setContentsMargins(32, 28, 32, 28)
        self._root_layout.setSpacing(18)
        self._root_layout.addWidget(
            PageHeader(self.page_title, self.page_subtitle),
        )
        self._scroll_area.setWidget(self._content_widget)
        outer_layout.addWidget(self._scroll_area)

    def run_task(
        self,
        task_title: str,
        function: Callable[..., object],
        on_success: Callable[[object], None],
        on_finished: Callable[[], None] | None,
        *args: object,
        on_failure: Callable[[Exception], None] | None = None,
        **kwargs: object,
    ) -> None:
        """Run a blocking backend call on a worker thread."""

        def handle_failure(error: object) -> None:
            """Forward worker errors to the page-level error handler."""
            if isinstance(error, Exception):
                self.show_error(task_title, error)
                if on_failure is not None:
                    on_failure(error)
            else:
                self.show_error(task_title, str(error))

        def handle_finished() -> None:
            """Run the optional completion callback after the worker stops."""
            if on_finished is not None:
                on_finished()

        thread = run_in_thread(
            self,
            function,
            on_success,
            handle_failure,
            handle_finished,
            *args,
            **kwargs,
        )
        self._worker_threads.append(thread)
        thread.finished.connect(lambda: self._forget_worker(thread))

    def show_info(self, title: str, message: str) -> None:
        """Show a non-blocking user-facing information message."""
        QMessageBox.information(self, title, message)

    def show_error(self, title: str, error: Exception | str) -> None:
        """Log an exception and show a friendly error message."""
        if isinstance(error, Exception):
            LOGGER.error("%s: %s", title, error)
            message = str(error)
        else:
            message = error
        QMessageBox.warning(self, title, message)

    def request_navigation(self, route: NavigationRoute) -> None:
        """Ask the shell to display another top-level page."""
        self.navigation_requested.emit(route.value)

    def _forget_worker(self, thread: QThread) -> None:
        """Drop a completed worker thread reference."""
        if thread in self._worker_threads:
            self._worker_threads.remove(thread)
