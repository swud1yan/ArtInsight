"""Reusable Qt table models for ArtInsight pages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal

RowType = TypeVar("RowType")


@dataclass(frozen=True)
class TableColumn(Generic[RowType]):
    """Describe one read-only table column."""

    title: str
    value_getter: Callable[[RowType], object]
    alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft
    check_getter: Callable[[RowType], bool] | None = None


class ReadOnlyTableModel(QAbstractTableModel, Generic[RowType]):
    """Generic read-only table model backed by Python rows."""

    check_toggled = Signal(object, bool)

    def __init__(self, columns: Sequence[TableColumn[RowType]]) -> None:
        """Create a model with fixed column definitions."""
        super().__init__()
        self._columns = list(columns)
        self._rows: list[RowType] = []

    def set_rows(self, rows: Sequence[RowType]) -> None:
        """Replace all rows and notify attached views."""
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, row: int) -> RowType | None:
        """Return the row object at an index, if present."""
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def rows(self) -> list[RowType]:
        """Return a copy of all current rows."""
        return list(self._rows)

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        """Return row count for Qt views."""
        if parent is not None and parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        """Return column count for Qt views."""
        if parent is not None and parent.isValid():
            return 0
        return len(self._columns)

    def data(
        self,
        index: QModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object:
        """Return display, tooltip, and alignment data."""
        if not index.isValid():
            return None
        row = self.row_at(index.row())
        if row is None:
            return None
        column = self._columns[index.column()]
        if role == Qt.ItemDataRole.CheckStateRole and column.check_getter is not None:
            return (
                Qt.CheckState.Checked
                if column.check_getter(row)
                else Qt.CheckState.Unchecked
            )
        if role in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole}:
            if column.check_getter is not None and role == Qt.ItemDataRole.DisplayRole:
                return ""
            value = column.value_getter(row)
            return "" if value is None else str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(column.alignment | Qt.AlignmentFlag.AlignVCenter)
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Return item flags, enabling checkboxes for configured columns."""
        flags = super().flags(index)
        if not index.isValid():
            return flags
        column = self._columns[index.column()]
        if column.check_getter is not None:
            return flags | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
        return flags

    def setData(
        self,
        index: QModelIndex,
        value: object,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        """Emit checkbox changes for callers to persist."""
        if not index.isValid() or role != Qt.ItemDataRole.CheckStateRole:
            return False
        row = self.row_at(index.row())
        if row is None:
            return False
        column = self._columns[index.column()]
        if column.check_getter is None:
            return False
        enabled = value == Qt.CheckState.Checked.value or value == Qt.CheckState.Checked
        self.check_toggled.emit(row, bool(enabled))
        return True

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object:
        """Return horizontal header labels."""
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._columns):
                return self._columns[section].title
        return str(section + 1)
