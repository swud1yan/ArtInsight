"""Interactive PCA contribution and reconstruction comparison widget."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from controllers.gui_dtos import PCAExplanationData
from views.chart_widgets import MatplotlibChart
from views.dashboard_widgets import EmptyActionPanel
from views.pca_analysis_state import PCAAnalysisState
from views.ui_helpers import create_panel


class PCAInterpretationWidget(QWidget):
    """Show selected-K PCA charts and reconstruction metrics."""

    component_changed = Signal(int)

    def __init__(self, state: PCAAnalysisState) -> None:
        """Create the selected-K interpretation widget."""
        super().__init__()
        self._state = state
        self._k_value_label = QLabel("K = -")
        self._variance_label = QLabel("累计解释方差：-")
        self._mse_label = QLabel("重建误差：-")
        self._mse_label.setObjectName("hintLabel")
        self._k_slider = QSlider(Qt.Orientation.Horizontal)
        self._k_slider.setEnabled(False)
        self._k_slider.setMinimum(1)
        self._k_slider.valueChanged.connect(self._handle_k_changed)
        self._scree_chart = MatplotlibChart()
        self._cumulative_chart = MatplotlibChart()
        self._scatter_chart = MatplotlibChart()
        for chart in (self._scree_chart, self._cumulative_chart, self._scatter_chart):
            chart.setMinimumHeight(360)
        self._stack = QStackedWidget()
        self._empty_panel = EmptyActionPanel(
            "请先导入WikiArt数据集，完成特征提取与模型训练后再查看PCA解释结果。",
            "等待PCA结果",
        )
        self._empty_panel.button.setEnabled(False)
        self._setup_ui()
        self.draw_empty()

    @property
    def slider_enabled(self) -> bool:
        """Return whether the K slider is enabled."""
        return self._k_slider.isEnabled()

    def set_loading(self, loading: bool) -> None:
        """Disable slider while PCA data is loading."""
        self._k_slider.setEnabled((not loading) and self._state.has_data)

    def set_empty_action(
        self,
        message: str,
        button_text: str,
        callback: Callable[[], None],
    ) -> None:
        """Configure the empty-state action button."""
        self._empty_panel.set_message(message)
        self._empty_panel.set_button_text(button_text)
        self._empty_panel.button.setEnabled(True)
        self._empty_panel.button.clicked.connect(callback)

    def draw_empty(self) -> None:
        """Draw empty charts before a PCA model is loaded."""
        self._k_panel.setVisible(False)
        self._stack.setCurrentWidget(self._empty_panel)

    def load_data(self, data: PCAExplanationData) -> None:
        """Configure slider and charts after PCA data is loaded."""
        self._k_panel.setVisible(True)
        self._stack.setCurrentWidget(self._chart_panel)
        self._configure_slider(data.max_components)
        self.update_for_k(self._state.default_component_count())

    def update_for_k(self, component_count: int) -> None:
        """Update metrics and charts for one component count."""
        data = self._state.data
        if data is None:
            return
        metrics = self._state.metrics_for(component_count)
        self._k_value_label.setText(
            f"K = {metrics.component_count} / {data.max_components}",
        )
        self._variance_label.setText(metrics.retained_text)
        self._mse_label.setText(metrics.loss_text)
        self._scree_chart.draw_scree(
            data.eigenvalues,
            max_components=metrics.component_count,
        )
        self._cumulative_chart.draw_explained_variance(
            data.explained_variance_ratio,
            data.cumulative_variance_ratio,
            max_components=metrics.component_count,
        )
        self._scatter_chart.draw_pca_scatter(
            self._state.scatter_points(metrics.component_count),
            self._state.scatter_title(metrics.component_count),
        )

    def _setup_ui(self) -> None:
        """Build slider metrics and chart panels."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(18)
        self._k_panel = self._create_k_panel()
        root.addWidget(self._k_panel)
        self._chart_panel = self._create_chart_panel()
        self._stack.addWidget(self._empty_panel)
        self._stack.addWidget(self._chart_panel)
        root.addWidget(self._stack, stretch=1)

    def _create_k_panel(self) -> QWidget:
        """Create slider controls for interactive K comparison."""
        panel = create_panel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        metric_row = QHBoxLayout()
        metric_row.setSpacing(16)
        metric_row.addWidget(self._k_value_label)
        metric_row.addWidget(self._variance_label)
        metric_row.addStretch(1)
        layout.addLayout(metric_row)
        layout.addWidget(self._k_slider)
        layout.addWidget(self._mse_label)
        return panel

    def _create_chart_panel(self) -> QWidget:
        """Create the chart row for Scree, variance, and projection plots."""
        panel = create_panel()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        layout.addWidget(self._scree_chart)
        layout.addWidget(self._cumulative_chart)
        layout.addWidget(self._scatter_chart)
        return panel

    def _configure_slider(self, max_components: int) -> None:
        """Configure slider bounds without emitting stale updates."""
        self._k_slider.blockSignals(True)
        self._k_slider.setMaximum(max_components)
        self._k_slider.setValue(self._state.default_component_count())
        self._k_slider.blockSignals(False)
        self._k_slider.setEnabled(True)

    def _handle_k_changed(self, component_count: int) -> None:
        """Refresh interpretation charts for the selected K value."""
        if self._state.has_data:
            self.update_for_k(component_count)
            self.component_changed.emit(component_count)
