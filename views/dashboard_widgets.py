"""Reusable widgets for the modern ArtInsight dashboard."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout


class DashboardCard(QFrame):
    """Compact card used by dashboard overview and workflow sections."""

    def __init__(self, title: str, subtitle: str = "") -> None:
        """Create a titled dashboard card."""
        super().__init__()
        self.setObjectName("dashboardCard")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(18, 16, 18, 16)
        self._layout.setSpacing(8)

        self._title_label = QLabel(title)
        self._title_label.setObjectName("dashboardCardTitle")
        self._title_label.setWordWrap(True)
        self._layout.addWidget(self._title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("dashboardCardSubtitle")
            subtitle_label.setWordWrap(True)
            self._layout.addWidget(subtitle_label)

    @property
    def content_layout(self) -> QVBoxLayout:
        """Return the inner layout for page-specific content."""
        return self._layout


class StatusStepWidget(QFrame):
    """One workflow status item with a pass/fail visual marker."""

    def __init__(self, title: str, done_text: str, pending_text: str) -> None:
        """Create a status widget."""
        super().__init__()
        self.setObjectName("statusStepPending")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(5)

        self._title_label = QLabel(title)
        self._title_label.setObjectName("statusStepTitle")
        self._message_label = QLabel(pending_text)
        self._message_label.setObjectName("statusStepMessage")
        self._message_label.setWordWrap(True)
        self._done_text = done_text
        self._pending_text = pending_text

        layout.addWidget(self._title_label)
        layout.addWidget(self._message_label)

    def set_done(self, is_done: bool) -> None:
        """Update the status marker and message."""
        self.setObjectName("statusStepDone" if is_done else "statusStepPending")
        self._message_label.setText(self._done_text if is_done else self._pending_text)
        self.style().unpolish(self)
        self.style().polish(self)


class OverviewMetricWidget(QFrame):
    """Metric card that avoids meaningless zero states on the dashboard."""

    def __init__(self, title: str) -> None:
        """Create a metric card."""
        super().__init__()
        self.setObjectName("overviewMetric")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 13)
        layout.setSpacing(5)

        self._value_label = QLabel("待完成")
        self._value_label.setObjectName("overviewMetricValue")
        self._title_label = QLabel(title)
        self._title_label.setObjectName("overviewMetricTitle")

        layout.addWidget(self._value_label)
        layout.addWidget(self._title_label)

    def set_display_value(self, value: str) -> None:
        """Set the visible metric value."""
        self._value_label.setText(value)


class EmptyActionPanel(QFrame):
    """Friendly empty-state panel with one call-to-action button."""

    def __init__(self, message: str, button_text: str) -> None:
        """Create an action-oriented empty state."""
        super().__init__()
        self.setObjectName("emptyActionPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)

        self._label = QLabel(message)
        self._label.setObjectName("emptyActionTitle")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)

        self.button = QPushButton(button_text)
        self.button.setObjectName("primaryButton")

        layout.addStretch(1)
        layout.addWidget(self._label)
        layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

    def set_message(self, message: str) -> None:
        """Update the empty-state message."""
        self._label.setText(message)

    def set_button_text(self, text: str) -> None:
        """Update the call-to-action button text."""
        self.button.setText(text)


class RecentExperimentRow(QFrame):
    """One key-value row for the recent experiment card."""

    def __init__(self, label: str) -> None:
        """Create a row with a static label and mutable value."""
        super().__init__()
        self.setObjectName("recentExperimentRow")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label_widget = QLabel(label)
        label_widget.setObjectName("recentExperimentLabel")
        self._value_label = QLabel("待完成")
        self._value_label.setObjectName("recentExperimentValue")
        self._value_label.setWordWrap(True)
        layout.addWidget(label_widget)
        layout.addWidget(self._value_label)

    def set_value(self, value: str) -> None:
        """Update the row value."""
        self._value_label.setText(value)
