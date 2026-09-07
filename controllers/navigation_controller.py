"""Controller responsible for top-level page navigation."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QWidget

from app.navigation import NavigationRoute, PageDefinition
from views.main_window import MainWindow

LOGGER = logging.getLogger(__name__)


class NavigationController(QObject):
    """Bind navigation events from the main window to page switching."""

    def __init__(self, main_window: MainWindow) -> None:
        """Create a navigation controller for the given main window."""
        super().__init__()
        self._main_window = main_window
        self._main_window.navigation_requested.connect(self.show_page)

    def register_pages(self, pages: list[tuple[PageDefinition, QWidget]]) -> None:
        """Register top-level pages in the main window."""
        for definition, page in pages:
            self._main_window.register_page(
                route=definition.route,
                title=definition.title,
                page=page,
            )
            page_navigation = getattr(page, "navigation_requested", None)
            if page_navigation is not None:
                page_navigation.connect(self.show_page)
        LOGGER.info("Registered %s navigation pages.", len(pages))

    def select_initial_page(self, route: NavigationRoute) -> None:
        """Select the first page displayed after startup."""
        self._main_window.select_route(route)

    @Slot(str)
    def show_page(self, route_value: str) -> None:
        """Display the page for the selected route."""
        try:
            route = NavigationRoute(route_value)
        except ValueError:
            LOGGER.warning("Ignoring unknown navigation route: %s", route_value)
            return
        self._main_window.display_page(route)
