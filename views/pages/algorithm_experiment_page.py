"""Algorithm experiment center page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from controllers.gui_dtos import (
    PaperExperimentExportResult,
    PaperExperimentSummaryResult,
)
from controllers.page_data_controller import PageDataController
from services.experiments.experiment_models import (
    BASELINE_ALGORITHMS,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_TOP_K,
    DEFAULT_VARIANCE_THRESHOLDS,
    FEATURE_SET_METHODS,
    AlgorithmExperimentRequest,
    AlgorithmExperimentResult,
    ExperimentMetricRow,
    ExperimentType,
)
from services.style_translations import format_style_label
from views.dashboard_widgets import DashboardCard, OverviewMetricWidget
from views.experiment_charts import ExperimentChart
from views.pages.base_page import BasePage
from views.table_models import ReadOnlyTableModel, TableColumn
from views.ui_helpers import configure_table, create_button, create_panel
from views.widgets import button_row
from views.zoom_dialogs import show_enlarged_table


EXPERIMENT_TYPES: tuple[ExperimentType, ...] = (
    ExperimentType.BASELINE,
    ExperimentType.FEATURE_ABLATION,
    ExperimentType.PCA_SENSITIVITY,
    ExperimentType.RETRIEVAL_EVALUATION,
    ExperimentType.CLASSIFICATION_EVALUATION,
    ExperimentType.EFFICIENCY_BENCHMARK,
)


class AlgorithmExperimentPage(BasePage):
    """MATLAB-style algorithm experiment center."""

    page_title = "算法实验中心"
    page_subtitle = (
        "选择实验类型、算法与指标，运行可复现的分类、检索、降维和效率实验，"
        "并自动生成论文级统计图与结果讨论。"
    )

    def __init__(self, controller: PageDataController) -> None:
        """Create the experiment center page."""
        super().__init__(controller)
        self._result: AlgorithmExperimentResult | None = None
        self._last_request: AlgorithmExperimentRequest | None = None
        self._experiment_combo = QComboBox()
        self._algorithm_checks: dict[str, QCheckBox] = {}
        self._feature_checks: dict[str, QCheckBox] = {}
        self._top_k_checks: dict[int, QCheckBox] = {}
        self._variance_checks: dict[float, QCheckBox] = {}
        self._sample_limit = QSpinBox()
        self._neighbor_count = QSpinBox()
        self._seed_input = QSpinBox()
        self._component_input = QComboBox()
        self._run_button: QPushButton
        self._export_button: QPushButton
        self._summary_button: QPushButton
        self._progress = QProgressBar()
        self._log_view = QPlainTextEdit()
        self._summary_label = QLabel("尚未运行实验。")
        self._discussion = QPlainTextEdit()
        self._table_model = self._build_table_model()
        self._baseline_chart = ExperimentChart()
        self._feature_chart = ExperimentChart()
        self._pca_chart = ExperimentChart()
        self._confusion_chart = ExperimentChart()
        self._paper_chart = ExperimentChart()
        self._retrieval_list = QListWidget()
        self._figure_list = QListWidget()
        self._metric_best_accuracy = OverviewMetricWidget("Best Accuracy")
        self._metric_best_map = OverviewMetricWidget("Best mAP")
        self._metric_fastest = OverviewMetricWidget("Fastest")
        self._metric_memory = OverviewMetricWidget("Lowest Memory")
        self._setup_ui()
        self._draw_empty_charts()

    def _setup_ui(self) -> None:
        """Build the left configuration and right result panes."""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._create_config_panel())
        splitter.addWidget(self._create_result_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 900])
        self._root_layout.addWidget(splitter, stretch=1)

    def _create_config_panel(self) -> QFrame:
        """Create left-side experiment configuration."""
        panel = create_panel()
        panel.setMinimumWidth(350)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        layout.addWidget(self._experiment_type_group())
        layout.addWidget(self._algorithm_group())
        layout.addWidget(self._feature_group())
        layout.addWidget(self._pca_group())
        layout.addWidget(self._retrieval_group())
        layout.addWidget(self._runtime_group())

        self._run_button = create_button("开始实验", "play", primary=True)
        self._run_button.clicked.connect(self._start_experiment)
        self._export_button = create_button("导出论文结果", "save")
        self._export_button.setEnabled(False)
        self._export_button.clicked.connect(self._export_current_result)
        self._summary_button = create_button("汇总论文结果", "chart")
        self._summary_button.clicked.connect(self._summarize_paper_results)
        layout.addWidget(button_row(self._run_button, self._export_button))
        layout.addWidget(button_row(self._summary_button))

        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(150)
        self._log_view.setPlainText("Ready.")
        layout.addWidget(self._log_view)
        layout.addStretch(1)
        return panel

    def _experiment_type_group(self) -> QGroupBox:
        """Create experiment type selector."""
        group = QGroupBox("实验类型")
        layout = QVBoxLayout(group)
        for experiment_type in EXPERIMENT_TYPES:
            self._experiment_combo.addItem(experiment_type.value, experiment_type)
        layout.addWidget(self._experiment_combo)
        return group

    def _algorithm_group(self) -> QGroupBox:
        """Create baseline algorithm checkboxes."""
        group = QGroupBox("Baseline Comparison")
        layout = QVBoxLayout(group)
        for algorithm in BASELINE_ALGORITHMS:
            checkbox = QCheckBox(algorithm)
            checkbox.setChecked(algorithm != "CLIP Embedding + KNN")
            self._algorithm_checks[algorithm] = checkbox
            layout.addWidget(checkbox)
        return group

    def _feature_group(self) -> QGroupBox:
        """Create feature ablation checkboxes."""
        group = QGroupBox("Feature Ablation")
        layout = QGridLayout(group)
        for index, feature_name in enumerate(FEATURE_SET_METHODS):
            checkbox = QCheckBox(feature_name)
            checkbox.setChecked(feature_name in {"Color+Texture", "All Features"})
            self._feature_checks[feature_name] = checkbox
            layout.addWidget(checkbox, index // 2, index % 2)
        return group

    def _pca_group(self) -> QGroupBox:
        """Create PCA sensitivity controls."""
        group = QGroupBox("PCA Sensitivity")
        layout = QVBoxLayout(group)
        self._component_input.setEditable(True)
        self._component_input.addItems([",".join(str(v) for v in DEFAULT_PCA_COMPONENTS)])
        self._component_input.addItems(["2,5,10,20,30,50,100", "5,10,12,20,30"])
        layout.addWidget(QLabel("主成分数量"))
        layout.addWidget(self._component_input)
        variance_row = QHBoxLayout()
        for threshold in DEFAULT_VARIANCE_THRESHOLDS:
            checkbox = QCheckBox(f"{threshold:.0%}")
            checkbox.setChecked(True)
            self._variance_checks[threshold] = checkbox
            variance_row.addWidget(checkbox)
        layout.addWidget(QLabel("累计贡献率"))
        layout.addLayout(variance_row)
        return group

    def _retrieval_group(self) -> QGroupBox:
        """Create retrieval metric controls."""
        group = QGroupBox("Retrieval Evaluation")
        layout = QHBoxLayout(group)
        for top_k in DEFAULT_TOP_K:
            checkbox = QCheckBox(f"Top-{top_k}")
            checkbox.setChecked(True)
            self._top_k_checks[top_k] = checkbox
            layout.addWidget(checkbox)
        return group

    def _runtime_group(self) -> QGroupBox:
        """Create runtime/sample controls."""
        group = QGroupBox("实验规模")
        layout = QGridLayout(group)
        self._sample_limit.setRange(60, 5000)
        self._sample_limit.setSingleStep(60)
        self._sample_limit.setValue(600)
        self._neighbor_count.setRange(1, 50)
        self._neighbor_count.setValue(5)
        self._seed_input.setRange(0, 2_147_483_647)
        self._seed_input.setValue(42)
        layout.addWidget(QLabel("Random Seed"), 2, 0)
        layout.addWidget(self._seed_input, 2, 1)
        layout.addWidget(QLabel("采样上限"), 0, 0)
        layout.addWidget(self._sample_limit, 0, 1)
        layout.addWidget(QLabel("KNN邻居"), 1, 0)
        layout.addWidget(self._neighbor_count, 1, 1)
        return group

    def _create_result_panel(self) -> QWidget:
        """Create right-side tabbed experiment results."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._create_summary_panel())
        tabs = QTabWidget()
        tabs.setObjectName("analysisTabs")
        tabs.addTab(self._result_table_tab(), "实验结果")
        tabs.addTab(self._chart_tab(self._baseline_chart), "Baseline Comparison")
        tabs.addTab(self._chart_tab(self._feature_chart), "Feature Ablation")
        tabs.addTab(self._chart_tab(self._pca_chart), "PCA Sensitivity")
        tabs.addTab(self._chart_tab(self._confusion_chart), "Confusion Matrix")
        tabs.addTab(self._retrieval_tab(), "Retrieval Examples")
        tabs.addTab(self._paper_tab(), "论文图生成器")
        layout.addWidget(tabs, stretch=1)
        return container

    def _create_summary_panel(self) -> QFrame:
        """Create best-result and discussion summary."""
        panel = create_panel()
        layout = QGridLayout(panel)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)
        for index, metric in enumerate(
            (
                self._metric_best_accuracy,
                self._metric_best_map,
                self._metric_fastest,
                self._metric_memory,
            ),
        ):
            layout.addWidget(metric, 0, index)
        self._summary_label.setObjectName("hintLabel")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label, 1, 0, 1, 4)
        self._discussion.setReadOnly(True)
        self._discussion.setMaximumHeight(90)
        self._discussion.setPlainText("Discussion will be generated after the experiment.")
        layout.addWidget(self._discussion, 2, 0, 1, 4)
        return panel

    def _result_table_tab(self) -> QWidget:
        """Create the main metrics table tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        table = QTableView()
        configure_table(table)
        table.setModel(self._table_model)
        zoom_button = create_button("放大表格", "search")
        zoom_button.clicked.connect(
            lambda: show_enlarged_table(self, table, "算法实验结果"),
        )
        layout.addWidget(button_row(zoom_button))
        layout.addWidget(table)
        return tab

    def _chart_tab(self, chart: ExperimentChart) -> QWidget:
        """Wrap a chart in a tab widget."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(chart)
        return tab

    def _retrieval_tab(self) -> QWidget:
        """Create retrieval example tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._retrieval_list)
        return tab

    def _paper_tab(self) -> QWidget:
        """Create paper figure generator tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        card = DashboardCard("300dpi PNG Figures", "自动保存 Figure1 到 Figure7。")
        card.content_layout.addWidget(self._figure_list)
        layout.addWidget(card, stretch=1)
        layout.addWidget(self._paper_chart, stretch=2)
        return tab

    def _start_experiment(self) -> None:
        """Collect configuration and run experiment in the background."""
        request = self._build_request()
        self._last_request = request
        self._result = None
        self._set_busy(True)
        self._export_button.setEnabled(False)
        self._progress.setValue(8)
        self._log_view.setPlainText(
            "\n".join(
                [
                    "Loading Dataset...",
                    "Extract Feature...",
                    "Running PCA...",
                    "Training...",
                    "Testing...",
                    "Generating Report...",
                ],
            ),
        )
        self.run_task(
            "算法实验失败",
            self._controller.run_algorithm_experiment,
            self._handle_success,
            self._finish_task,
            request,
        )

    def _handle_success(self, result: object) -> None:
        """Render completed experiment results."""
        if not isinstance(result, AlgorithmExperimentResult):
            self.show_error("实验结果异常", "后端返回了无法识别的实验结果。")
            return
        self._result = result
        self._table_model.set_rows(result.rows)
        self._render_summary(result)
        self._render_charts(result)
        self._render_retrieval(result)
        self._render_figures(result)
        self._log_view.setPlainText("\n".join(result.logs))
        self._progress.setValue(100)
        self._export_button.setEnabled(True)

    def _finish_task(self) -> None:
        """Restore controls after background execution."""
        self._set_busy(False)

    def _export_current_result(self) -> None:
        """Export the current experiment result to paper-ready artifacts."""
        if self._result is None or self._last_request is None:
            self.show_info("尚无实验结果", "请先成功运行一次实验，然后再导出论文结果。")
            return
        self._set_busy(True)
        self._log_view.setPlainText("Exporting paper experiment artifacts...")
        self.run_task(
            "论文结果导出失败",
            self._controller.export_paper_experiment_result,
            self._handle_export_success,
            self._finish_task,
            self._result,
            self._last_request,
        )

    def _handle_export_success(self, result: object) -> None:
        """Render a completed paper export result."""
        if not isinstance(result, PaperExperimentExportResult):
            self.show_error("论文结果异常", "后端返回了无法识别的论文导出结果。")
            return
        self._log_view.setPlainText(
            "\n".join(
                [
                    result.message,
                    f"metrics.csv: {result.metrics_csv}",
                    f"summary.json: {result.summary_json}",
                    f"report.md: {result.report_md}",
                    f"figures: {result.figures_dir}",
                ],
            ),
        )
        self.show_info("导出完成", f"{result.message}\n\n{result.output_dir}")

    def _summarize_paper_results(self) -> None:
        """Aggregate all exported paper experiment metrics."""
        self._set_busy(True)
        self._log_view.setPlainText("Aggregating exported paper experiment metrics...")
        self.run_task(
            "论文结果汇总失败",
            self._controller.summarize_paper_experiments,
            self._handle_summary_success,
            self._finish_task,
        )

    def _handle_summary_success(self, result: object) -> None:
        """Render a completed paper summary result."""
        if not isinstance(result, PaperExperimentSummaryResult):
            self.show_error("论文汇总异常", "后端返回了无法识别的论文汇总结果。")
            return
        self._log_view.setPlainText(
            "\n".join(
                [
                    result.message,
                    f"aggregate_metrics.csv: {result.aggregate_csv}",
                    f"significance_tests.csv: {result.significance_csv}",
                    f"paper_experiment_summary.md: {result.report_md}",
                ],
            ),
        )
        self.show_info("汇总完成", f"{result.message}\n\n{result.output_dir}")

    def _set_busy(self, busy: bool) -> None:
        """Toggle controls while a long-running experiment is active."""
        self._run_button.setEnabled(not busy)
        self._export_button.setEnabled(
            not busy and self._result is not None and self._last_request is not None,
        )
        self._summary_button.setEnabled(not busy)
        self._experiment_combo.setEnabled(not busy)
        self._sample_limit.setEnabled(not busy)
        self._neighbor_count.setEnabled(not busy)
        self._seed_input.setEnabled(not busy)
        self._component_input.setEnabled(not busy)
        if busy:
            self._progress.setRange(0, 0)
        else:
            self._progress.setRange(0, 100)

    def _build_request(self) -> AlgorithmExperimentRequest:
        """Build a request DTO from UI controls."""
        experiment_type = self._experiment_combo.currentData()
        if not isinstance(experiment_type, ExperimentType):
            experiment_type = ExperimentType.BASELINE
        algorithms = tuple(
            name
            for name, checkbox in self._algorithm_checks.items()
            if checkbox.isChecked()
        )
        feature_sets = tuple(
            name
            for name, checkbox in self._feature_checks.items()
            if checkbox.isChecked()
        )
        top_k_values = tuple(
            value
            for value, checkbox in self._top_k_checks.items()
            if checkbox.isChecked()
        )
        variance_thresholds = tuple(
            value
            for value, checkbox in self._variance_checks.items()
            if checkbox.isChecked()
        )
        seed = self._seed_input.value()
        output_dir = (
            self._controller.config.paths.export_dir
            / "paper_experiments"
            / f"{experiment_type.name.lower()}_seed_{seed}"
            / "figures"
        )
        return AlgorithmExperimentRequest(
            experiment_type=experiment_type,
            algorithms=algorithms or ("PCA + KNN",),
            feature_sets=feature_sets or ("Color+Texture",),
            pca_components=self._parse_components(),
            variance_thresholds=variance_thresholds,
            top_k_values=top_k_values or (5,),
            n_neighbors=self._neighbor_count.value(),
            sample_limit=self._sample_limit.value(),
            random_state=seed,
            output_dir=output_dir,
        )

    def _parse_components(self) -> tuple[int, ...]:
        """Parse comma-separated component counts."""
        text = self._component_input.currentText()
        values: list[int] = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                value = int(part)
            except ValueError:
                continue
            if value > 0:
                values.append(value)
        return tuple(values) or DEFAULT_PCA_COMPONENTS

    def _render_summary(self, result: AlgorithmExperimentResult) -> None:
        """Render result summary and discussion."""
        summary = result.summary
        self._metric_best_accuracy.set_display_value(self._format_percent(summary.best_accuracy))
        self._metric_best_map.set_display_value(self._format_percent(summary.best_map))
        self._metric_fastest.set_display_value(self._format_ms(summary.fastest_time_ms))
        self._metric_memory.set_display_value(self._format_mb(summary.lowest_memory_mb))
        self._summary_label.setText(
            " | ".join(
                [
                    f"最佳算法: {summary.best_algorithm}",
                    f"最佳mAP: {summary.best_map_method}",
                    f"最快: {summary.fastest_method}",
                    f"最佳PCA维数: {summary.best_pca_dimension or 'N/A'}",
                ],
            ),
        )
        self._discussion.setPlainText(result.discussion)

    def _render_charts(self, result: AlgorithmExperimentResult) -> None:
        """Render all chart tabs."""
        self._baseline_chart.draw_metric_bars(
            result.baseline_rows or result.rows,
            "Baseline: Accuracy / Time / Memory / mAP",
            (
                ("Accuracy", "accuracy"),
                ("mAP", "map_score"),
                ("Memory(MB)", "memory_mb"),
            ),
        )
        self._feature_chart.draw_metric_bars(
            result.feature_rows or result.rows,
            "Feature Ablation: Accuracy",
            (("Accuracy", "accuracy"), ("Macro F1", "macro_f1")),
        )
        self._pca_chart.draw_pca_sensitivity(result.pca_rows)
        self._confusion_chart.draw_confusion_matrix(result.confusion_matrix)
        self._paper_chart.draw_dataset_distribution(result.dataset_info)

    def _render_retrieval(self, result: AlgorithmExperimentResult) -> None:
        """Render query and Top-5 retrieval examples."""
        self._retrieval_list.clear()
        if not result.retrieval_examples:
            self._retrieval_list.addItem("暂无检索样例。")
            return
        for example in result.retrieval_examples:
            self._retrieval_list.addItem(
                f"Query: {example.query_title} [{format_style_label(example.query_label)}]",
            )
            for rank, match in enumerate(example.matches, start=1):
                self._retrieval_list.addItem(
                    f"  Top{rank}: {match.title} [{format_style_label(match.label)}] "
                    f"Similarity={match.similarity_score:.3f}",
                )

    def _render_figures(self, result: AlgorithmExperimentResult) -> None:
        """Render generated paper figure paths."""
        self._figure_list.clear()
        for figure in result.paper_figures:
            path = Path(figure.path)
            self._figure_list.addItem(f"{figure.title}: {path}")

    def _draw_empty_charts(self) -> None:
        """Initialize all charts with empty states."""
        self._baseline_chart.draw_empty("Baseline Comparison")
        self._feature_chart.draw_empty("Feature Ablation")
        self._pca_chart.draw_empty("PCA Sensitivity")
        self._confusion_chart.draw_empty("Confusion Matrix")
        self._paper_chart.draw_empty("Paper Figure Generator")

    def _build_table_model(self) -> ReadOnlyTableModel[ExperimentMetricRow]:
        """Create the experiment metrics table model."""
        return ReadOnlyTableModel[ExperimentMetricRow](
            [
                TableColumn("Method", lambda row: row.method),
                TableColumn("Accuracy", lambda row: self._format_percent(row.accuracy), Qt.AlignRight),
                TableColumn("Macro F1", lambda row: self._format_percent(row.macro_f1), Qt.AlignRight),
                TableColumn("Precision", lambda row: self._format_percent(row.precision), Qt.AlignRight),
                TableColumn("Recall", lambda row: self._format_percent(row.recall), Qt.AlignRight),
                TableColumn("mAP", lambda row: self._format_percent(row.map_score), Qt.AlignRight),
                TableColumn("Time(ms)", lambda row: self._format_ms(row.total_time_ms), Qt.AlignRight),
                TableColumn("Memory(MB)", lambda row: self._format_mb(row.memory_mb), Qt.AlignRight),
                TableColumn("Dimension", lambda row: row.dimension or "", Qt.AlignRight),
                TableColumn("Notes", lambda row: row.notes),
            ],
        )

    def _format_percent(self, value: float | None) -> str:
        """Format a nullable ratio as percent."""
        return "N/A" if value is None else f"{value * 100:.2f}%"

    def _format_ms(self, value: float | None) -> str:
        """Format a nullable millisecond value."""
        return "N/A" if value is None else f"{value:.1f}"

    def _format_mb(self, value: float | None) -> str:
        """Format a nullable megabyte value."""
        return "N/A" if value is None else f"{value:.2f}"
