"""Application assembly for ArtInsight."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging

from PySide6.QtWidgets import QApplication

from app.config import Config, ConfigManager
from app.font_config import configure_application_font
from app.logging_config import initialize_logging
from app.navigation import NavigationRoute
from controllers.navigation_controller import NavigationController
from controllers.page_data_controller import PageDataController
from database.initializer import initialize_database
from database.session import DatabaseSessionManager
from views.main_window import MainWindow
from views.pages.registry import build_page_registry

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplicationContext:
    """Objects assembled during application startup."""

    qt_app: QApplication
    config: Config
    main_window: MainWindow
    navigation_controller: NavigationController
    session_manager: DatabaseSessionManager
    page_data_controller: PageDataController


def create_application(argv: Sequence[str]) -> ApplicationContext:
    """Create the Qt application, main window, pages, and controllers."""
    config = ConfigManager().load()
    initialize_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        file_name=config.logging.file_name,
    )
    LOGGER.info("Starting ArtInsight desktop application.")

    qt_app = QApplication(list(argv))
    qt_app.setApplicationName("ArtInsight")
    configure_application_font(
        qt_app,
        config.ui.font_family,
        config.ui.font_size,
    )
    _apply_theme(qt_app, config)

    session_manager = DatabaseSessionManager.from_url(config.database.url)
    initialize_database(session_manager.engine)
    page_data_controller = PageDataController(config, session_manager)

    main_window = MainWindow()
    navigation_controller = NavigationController(main_window)
    page_registry = build_page_registry(page_data_controller)
    navigation_controller.register_pages(page_registry)
    navigation_controller.select_initial_page(NavigationRoute.HOME)

    return ApplicationContext(
        qt_app=qt_app,
        config=config,
        main_window=main_window,
        navigation_controller=navigation_controller,
        session_manager=session_manager,
        page_data_controller=page_data_controller,
    )


def _apply_theme(qt_app: QApplication, config: Config) -> None:
    """Apply the global Qt stylesheet when it is available."""
    if not config.ui.theme_path.exists():
        LOGGER.warning("Theme file does not exist: %s", config.ui.theme_path)
        return
    qt_app.setStyleSheet(config.ui.theme_path.read_text(encoding="utf-8"))
