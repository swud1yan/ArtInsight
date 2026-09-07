"""Small UI helpers shared by feature pages."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QPushButton,
    QScrollArea,
    QTableView,
    QWidget,
)

from views.icons import load_icon

IMAGE_FILTER = "图片文件 (*.jpg *.jpeg *.png *.webp);;所有文件 (*)"


def create_button(
    text: str,
    icon_name: str | None = None,
    primary: bool = False,
    danger: bool = False,
) -> QPushButton:
    """Create a consistently styled button with an optional resource icon."""
    button = QPushButton(text)
    if icon_name:
        icon_color = "#FFFFFF" if primary or danger else None
        button.setIcon(load_icon(icon_name, color=icon_color))
    if primary:
        button.setObjectName("primaryButton")
    if danger:
        button.setObjectName("dangerButton")
    return button


def choose_image_file(parent: QWidget, start_dir: Path) -> Path | None:
    """Open an image file picker and return the selected path."""
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "选择图片",
        str(start_dir),
        IMAGE_FILTER,
    )
    return Path(file_path) if file_path else None


def choose_folder(parent: QWidget, start_dir: Path) -> Path | None:
    """Open a directory picker and return the selected folder."""
    folder_path = QFileDialog.getExistingDirectory(
        parent,
        "选择文件夹",
        str(start_dir),
    )
    return Path(folder_path) if folder_path else None


def create_panel() -> QFrame:
    """Create a neutral panel frame for grouped controls."""
    panel = QFrame()
    panel.setObjectName("panel")
    return panel


def create_scroll_area(widget: QWidget) -> QScrollArea:
    """Wrap a widget in a borderless resizable scroll area."""
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QFrame.Shape.NoFrame)
    scroll_area.setWidget(widget)
    return scroll_area


def configure_table(table: QTableView) -> None:
    """Apply common read-only table-view behavior."""
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
    table.setSortingEnabled(False)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
