"""PCA-specific Matplotlib chart rendering helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from services.style_translations import format_style_label
from visualization.matplotlib_style import configure_matplotlib_fonts
from visualization.pca_chart_models import HoverAnnotation, PCAScatterPoint
from visualization.pca_chart_validation import (
    limit_components,
    validate_ratio_vector,
    validate_scatter_points,
    validate_vector,
)

configure_matplotlib_fonts()

COLOR_CYCLE: tuple[str, ...] = (
    "#2F6F73",
    "#C0563D",
    "#6A7FDB",
    "#8E6C3A",
    "#4E8F57",
    "#B35D9B",
    "#557A95",
    "#D08C2F",
)
MAX_INLINE_LEGEND_LABELS = 12


def draw_empty_chart(
    figure: Figure,
    title: str,
    message: str = "暂无数据",
) -> None:
    """Draw a centered empty-state message on a Matplotlib figure."""
    figure.clear()
    axis = figure.add_subplot(111)
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center")
    axis.set_axis_off()
    figure.tight_layout()


def draw_scree_plot(
    figure: Figure,
    eigenvalues: Sequence[float] | np.ndarray,
    *,
    max_components: int | None = 20,
    title: str = "Scree Plot",
) -> None:
    """Draw a Scree plot showing PCA eigenvalue magnitude."""
    values = limit_components(
        validate_vector(eigenvalues, "eigenvalues"),
        max_components,
    )
    figure.clear()
    axis = figure.add_subplot(111)
    components = np.arange(1, values.size + 1)
    axis.bar(components, values, color=COLOR_CYCLE[0])
    axis.set_title(title)
    axis.set_xlabel("主成分序号")
    axis.set_ylabel("特征值")
    axis.set_xticks(components)
    axis.grid(axis="y", alpha=0.24)
    figure.tight_layout()


def draw_explained_variance_plot(
    figure: Figure,
    explained_ratio: Sequence[float] | np.ndarray,
    cumulative_ratio: Sequence[float] | np.ndarray,
    *,
    max_components: int | None = 20,
    title: str = "解释方差与累计贡献率",
) -> None:
    """Draw explained variance bars with a cumulative variance curve."""
    explained = validate_ratio_vector(explained_ratio, "explained_ratio")
    cumulative = validate_ratio_vector(cumulative_ratio, "cumulative_ratio")
    if explained.size != cumulative.size:
        msg = "explained_ratio and cumulative_ratio must have the same length."
        raise ValueError(msg)
    explained = limit_components(explained, max_components)
    cumulative = limit_components(cumulative, max_components)

    figure.clear()
    axis = figure.add_subplot(111)
    components = np.arange(1, explained.size + 1)
    bars = axis.bar(
        components,
        explained,
        color=COLOR_CYCLE[0],
        alpha=0.82,
        label="单个贡献率",
    )
    axis.set_title(title)
    axis.set_xlabel("主成分序号")
    axis.set_ylabel("解释方差比")
    axis.set_xticks(components)
    axis.set_ylim(0.0, max(1.0, float(np.max(explained)) * 1.2))
    axis.grid(axis="y", alpha=0.22)

    twin_axis = axis.twinx()
    twin_axis.plot(
        components,
        cumulative,
        color=COLOR_CYCLE[1],
        marker="o",
        linewidth=2.0,
        label="累计贡献率",
    )
    twin_axis.set_ylabel("累计贡献率")
    twin_axis.set_ylim(0.0, 1.05)
    _annotate_bars(axis, bars)
    _merge_legends(axis, twin_axis)
    figure.tight_layout()


def draw_pca_scatter_2d(
    figure: Figure,
    points: Sequence[PCAScatterPoint],
    *,
    title: str = "PCA二维风格地图",
    interactive: bool = True,
    legend_mode: str = "compact",
) -> HoverAnnotation | None:
    """Draw an interactive 2-D PCA scatter plot grouped by style label."""
    validated_points = validate_scatter_points(points, required_dimension=2)
    figure.clear()
    axis = figure.add_subplot(111)
    if not validated_points:
        draw_empty_chart(figure, title, "暂无PCA地图数据")
        return None

    _draw_grouped_scatter_2d(axis, validated_points)
    axis.set_title(title)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    _place_scatter_legend(figure, axis, validated_points, legend_mode)
    axis.grid(alpha=0.22)
    _apply_scatter_layout(figure, validated_points, legend_mode)
    if not interactive:
        return None
    return _connect_hover_annotation(figure, axis, validated_points)


def draw_pca_scatter_3d(
    figure: Figure,
    points: Sequence[PCAScatterPoint],
    *,
    title: str = "PCA三维风格地图",
    legend_mode: str = "compact",
) -> None:
    """Draw a rotatable 3-D PCA scatter plot grouped by style label."""
    validated_points = validate_scatter_points(points, required_dimension=3)
    figure.clear()
    axis = figure.add_subplot(111, projection="3d")
    if not validated_points:
        axis.set_title(title)
        axis.text2D(0.5, 0.5, "暂无PCA三维数据", ha="center", va="center")
        axis.set_axis_off()
        figure.tight_layout()
        return

    labels = sorted({point.label for point in validated_points})
    for index, label in enumerate(labels):
        group = [point for point in validated_points if point.label == label]
        axis.scatter(
            [point.x_value for point in group],
            [point.y_value for point in group],
            [float(point.z_value) for point in group],
            label=format_style_label(label),
            color=COLOR_CYCLE[index % len(COLOR_CYCLE)],
            s=36,
            alpha=0.84,
            depthshade=True,
        )
    axis.set_title(title)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_zlabel("PC3")
    _place_scatter_legend(figure, axis, validated_points, legend_mode)
    _apply_scatter_layout(figure, validated_points, legend_mode)


def save_figure(figure: Figure, output_path: Path) -> Path:
    """Save a Matplotlib figure as an image and return the target path."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160, bbox_inches="tight")
    return target


def _draw_grouped_scatter_2d(axis: Axes, points: list[PCAScatterPoint]) -> None:
    """Draw grouped 2-D PCA points on an axis."""
    labels = sorted({point.label for point in points})
    for index, label in enumerate(labels):
        group = [point for point in points if point.label == label]
        axis.scatter(
            [point.x_value for point in group],
            [point.y_value for point in group],
            label=format_style_label(label),
            color=COLOR_CYCLE[index % len(COLOR_CYCLE)],
            s=36,
            alpha=0.84,
        )


def _place_scatter_legend(
    figure: Figure,
    axis: Axes,
    points: list[PCAScatterPoint],
    legend_mode: str,
) -> None:
    """Place a compact or full style legend for PCA scatter plots."""
    label_count = len({point.label for point in points})
    if legend_mode == "compact" and label_count > MAX_INLINE_LEGEND_LABELS:
        _draw_compact_legend_note(axis, label_count)
        return
    if label_count > MAX_INLINE_LEGEND_LABELS:
        _draw_full_legend_panel(figure, points)
        return
    axis.legend(loc="best", title="风格")


def _draw_compact_legend_note(axis: Axes, label_count: int) -> None:
    """Show a small non-blocking note instead of a huge inline legend."""
    axis.text(
        0.99,
        0.99,
        f"风格 {label_count} 类\n图例略，放大查看完整列表",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#b9c8ca",
            "alpha": 0.86,
        },
    )


def _apply_scatter_layout(
    figure: Figure,
    points: list[PCAScatterPoint],
    legend_mode: str,
) -> None:
    """Reserve right-side space for full long legends."""
    label_count = len({point.label for point in points})
    if legend_mode != "compact" and label_count > MAX_INLINE_LEGEND_LABELS:
        figure.subplots_adjust(left=0.08, right=0.60, top=0.9, bottom=0.12)
        axis = figure.axes[0]
        if hasattr(axis, "set_box_aspect"):
            axis.set_box_aspect(1)
        return
    figure.tight_layout()


def _draw_full_legend_panel(figure: Figure, points: list[PCAScatterPoint]) -> None:
    """Draw a full style legend in a separate right-side panel."""
    labels = sorted({point.label for point in points})
    _grow_figure_for_full_legend(figure, len(labels))
    legend_axis = figure.add_axes((0.66, 0.08, 0.31, 0.84))
    legend_axis.set_axis_off()
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=COLOR_CYCLE[index % len(COLOR_CYCLE)],
            markeredgecolor=COLOR_CYCLE[index % len(COLOR_CYCLE)],
            label=format_style_label(label),
        )
        for index, label in enumerate(labels)
    ]
    legend_axis.legend(
        handles=handles,
        loc="upper left",
        borderaxespad=0.0,
        title=f"风格 ({len(labels)})",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.92,
    )


def _grow_figure_for_full_legend(figure: Figure, label_count: int) -> None:
    """Increase figure height so long legends can be reached by scrolling."""
    width, height = figure.get_size_inches()
    target_height = min(18.0, max(height, 4.5 + label_count * 0.24))
    figure.set_size_inches(max(width, 13.5), target_height, forward=True)


def _connect_hover_annotation(
    figure: Figure,
    axis: Axes,
    points: list[PCAScatterPoint],
) -> HoverAnnotation:
    """Attach nearest-point hover annotation to a 2-D scatter figure."""
    annotation = axis.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "white", "alpha": 0.9},
    )
    annotation.set_visible(False)
    coordinates = np.asarray(
        [(point.x_value, point.y_value) for point in points],
        dtype=np.float64,
    )

    def on_move(event: Any) -> None:
        """Update the hover annotation for the nearest visible data point."""
        if event.inaxes != axis or event.xdata is None or event.ydata is None:
            annotation.set_visible(False)
            figure.canvas.draw_idle()
            return
        cursor = np.asarray([event.xdata, event.ydata], dtype=np.float64)
        distances = np.sum((coordinates - cursor) ** 2, axis=1)
        nearest = points[int(np.argmin(distances))]
        annotation.xy = (nearest.x_value, nearest.y_value)
        annotation.set_text(_format_hover_text(nearest))
        annotation.set_visible(True)
        figure.canvas.draw_idle()

    connection_id = figure.canvas.mpl_connect("motion_notify_event", on_move)
    return HoverAnnotation(connection_id=connection_id, annotation=annotation)


def _format_hover_text(point: PCAScatterPoint) -> str:
    """Format hover text for one PCA scatter point."""
    title = point.title or point.sample_id
    lines = [title, f"风格: {format_style_label(point.label)}"]
    if point.artist_name:
        lines.append(f"作者: {point.artist_name}")
    lines.append(f"ID: {point.sample_id}")
    return "\n".join(lines)


def _annotate_bars(axis: Axes, bars: Any) -> None:
    """Annotate contribution-ratio bars with compact percentage labels."""
    for bar in bars:
        height = float(bar.get_height())
        if height <= 0:
            continue
        axis.annotate(
            f"{height * 100:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _merge_legends(axis: Axes, twin_axis: Axes) -> None:
    """Merge legends from primary and secondary axes."""
    handles, labels = axis.get_legend_handles_labels()
    twin_handles, twin_labels = twin_axis.get_legend_handles_labels()
    axis.legend(
        handles + twin_handles,
        labels + twin_labels,
        loc="upper right",
    )
