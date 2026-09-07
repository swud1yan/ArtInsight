"""High-quality ReportLab PDF rendering for ArtInsight reports."""

from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from services.report_chart_renderer import ReportChartRenderer
from services.report_models import ReportSnapshot
from services.report_pdf_styles import (
    build_report_styles,
    build_report_table_style,
    register_report_font,
)


class ReportPdfRenderer:
    """Render a report snapshot to a styled PDF document."""

    def __init__(self) -> None:
        """Create a PDF renderer and register fonts."""
        self._font_name = register_report_font()
        self._styles = build_report_styles(self._font_name)
        self._chart_renderer = ReportChartRenderer()

    def write(self, snapshot: ReportSnapshot, output_path: Path) -> None:
        """Write a styled multi-page PDF report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chart_dir = output_path.parent / f".{output_path.stem}_assets_{uuid4().hex}"
        chart_dir.mkdir(parents=True, exist_ok=True)
        try:
            story = self._build_story(snapshot, chart_dir)
            document = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=1.6 * cm,
                leftMargin=1.6 * cm,
                topMargin=1.5 * cm,
                bottomMargin=1.5 * cm,
                title=snapshot.report_name,
                author=snapshot.author,
            )
            document.build(story)
        finally:
            shutil.rmtree(chart_dir, ignore_errors=True)

    def _build_story(self, snapshot: ReportSnapshot, chart_dir: Path) -> list[object]:
        """Build the ordered ReportLab flowables for the report."""
        story: list[object] = []
        story.extend(self._cover_flowables(snapshot))
        story.append(PageBreak())
        story.extend(self._dataset_flowables(snapshot, chart_dir))
        story.append(PageBreak())
        story.extend(self._pca_flowables(snapshot, chart_dir))
        story.append(PageBreak())
        story.extend(self._conclusion_flowables(snapshot))
        return story

    def _cover_flowables(self, snapshot: ReportSnapshot) -> list[object]:
        """Return cover-page flowables."""
        return [
            Spacer(1, 2.5 * cm),
            Paragraph(snapshot.project_name, self._styles["cover_title"]),
            Spacer(1, 0.3 * cm),
            Paragraph(snapshot.subtitle, self._styles["cover_subtitle"]),
            Spacer(1, 1.2 * cm),
            Paragraph(f"报告名称：{snapshot.report_name}", self._styles["body"]),
            Paragraph(f"生成时间：{snapshot.created_at:%Y-%m-%d %H:%M}", self._styles["body"]),
            Paragraph(f"用户：{snapshot.author}", self._styles["body"]),
            Spacer(1, 1.2 * cm),
            Paragraph(
                "本报告由 ArtInsight 自动生成，汇总当前数据库、PCA模型与可视化分析结果。",
                self._styles["body"],
            ),
        ]

    def _dataset_flowables(
        self,
        snapshot: ReportSnapshot,
        chart_dir: Path,
    ) -> list[object]:
        """Return dataset summary flowables."""
        style_chart = self._chart_renderer.bar_chart(
            chart_dir / "style_distribution.png",
            "风格分布",
            snapshot.style_distribution,
        )
        artist_chart = self._chart_renderer.bar_chart(
            chart_dir / "artist_distribution.png",
            "作者作品数量 Top10",
            snapshot.artist_distribution,
        )
        return [
            Paragraph("一、数据集与样本概览", self._styles["heading"]),
            self._metrics_table(
                [
                    ("数据集数量", snapshot.dataset_count),
                    ("作品数量", snapshot.artwork_count),
                    ("风格数量", snapshot.style_count),
                    ("特征记录", snapshot.feature_count),
                ],
            ),
            Spacer(1, 0.35 * cm),
            self._dataset_table(snapshot),
            Spacer(1, 0.35 * cm),
            self._image_row([style_chart, artist_chart]),
        ]

    def _pca_flowables(self, snapshot: ReportSnapshot, chart_dir: Path) -> list[object]:
        """Return PCA result and chart flowables."""
        variance_chart = self._chart_renderer.variance_chart(
            chart_dir / "variance_ratio.png",
            snapshot.pca_summary.explained_variance_ratio,
        )
        cumulative_chart = self._chart_renderer.cumulative_chart(
            chart_dir / "cumulative_ratio.png",
            snapshot.pca_summary.cumulative_variance_ratio,
        )
        accuracy = snapshot.training_summary.cross_validation_accuracy
        accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "暂无"
        pca = snapshot.pca_summary
        flowables: list[object] = [
            Paragraph("二、PCA分析与模型配置", self._styles["heading"]),
            self._metrics_table(
                [
                    ("模型状态", "已训练" if pca.model_available else "未训练"),
                    ("特征类型", pca.feature_type),
                    ("样本数", pca.sample_count),
                    ("特征维度", pca.feature_count),
                    ("保留主成分 K", pca.n_components),
                    ("累计解释率", f"{pca.cumulative_at_k:.2%}"),
                    ("交叉验证准确率", accuracy_text),
                ],
            ),
            Spacer(1, 0.25 * cm),
            Paragraph(
                f"预处理：{', '.join(snapshot.training_summary.preprocessing_ops) or '暂无'}",
                self._styles["body"],
            ),
            Paragraph(
                f"特征方法：{', '.join(snapshot.training_summary.feature_methods) or '暂无'}",
                self._styles["body"],
            ),
            Spacer(1, 0.35 * cm),
            self._image_row([variance_chart, cumulative_chart]),
        ]
        if snapshot.example_image_paths:
            flowables.extend([Spacer(1, 0.35 * cm), self._example_image_table(snapshot)])
        return flowables

    def _conclusion_flowables(self, snapshot: ReportSnapshot) -> list[object]:
        """Return conclusion and citation flowables."""
        pca = snapshot.pca_summary
        accuracy = snapshot.training_summary.cross_validation_accuracy
        accuracy_text = f"分类交叉验证准确率为 {accuracy:.2%}。" if accuracy is not None else ""
        dataset_name = snapshot.datasets[0].name if snapshot.datasets else "当前数据集"
        return [
            Paragraph("三、自动分析结论", self._styles["heading"]),
            Paragraph(
                f"本报告使用 {dataset_name} 等数据集，共 {snapshot.artwork_count} "
                f"张作品图像，覆盖 {snapshot.style_count} 个艺术风格。",
                self._styles["body"],
            ),
            Paragraph(
                f"PCA 当前保留 K={pca.n_components} 个主成分，"
                f"累计解释率为 {pca.cumulative_at_k:.2%}。{accuracy_text}",
                self._styles["body"],
            ),
            Paragraph(
                "风格分布、作者分布与PCA贡献率图可用于答辩展示和后续误差分析。",
                self._styles["body"],
            ),
            Spacer(1, 0.5 * cm),
            Paragraph("数据来源与引用说明", self._styles["subheading"]),
            Paragraph(
                "报告内容由本地数据库和训练产物自动读取生成；如使用 WikiArt "
                "等公开数据集，请在答辩材料中补充对应数据集引用。",
                self._styles["body"],
            ),
        ]

    def _metrics_table(self, rows: list[tuple[str, object]]) -> Table:
        """Build a two-column metrics table."""
        table = Table(
            [["指标", "数值"], *[[label, str(value)] for label, value in rows]],
            colWidths=[5.2 * cm, 10.8 * cm],
            hAlign="LEFT",
        )
        table.setStyle(build_report_table_style(self._font_name))
        return table

    def _dataset_table(self, snapshot: ReportSnapshot) -> Table:
        """Build a compact dataset table."""
        rows = [
            [row.name, str(row.artwork_count), f"{row.imported_at:%Y-%m-%d}"]
            for row in snapshot.datasets[:8]
        ]
        if not rows:
            rows = [["暂无数据集", "-", "-"]]
        table = Table(
            [["数据集", "作品数", "导入日期"], *rows],
            colWidths=[8.0 * cm, 3.0 * cm, 5.0 * cm],
            hAlign="LEFT",
        )
        table.setStyle(build_report_table_style(self._font_name))
        return table

    def _image_row(self, image_paths: list[Path]) -> Table:
        """Create a side-by-side image row."""
        images = [
            Image(str(path), width=7.5 * cm, height=5.2 * cm)
            for path in image_paths
            if path.exists()
        ]
        table = Table([images], hAlign="LEFT")
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return table

    def _example_image_table(self, snapshot: ReportSnapshot) -> Table:
        """Build a thumbnail grid for artwork examples."""
        cells = []
        for path in snapshot.example_image_paths[:4]:
            cells.append(Image(str(path), width=3.2 * cm, height=3.2 * cm))
        while len(cells) < 4:
            cells.append(Paragraph("暂无图片", self._styles["small"]))
        table = Table([cells[:2], cells[2:]], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9E1E2")),
                ],
            ),
        )
        return table
