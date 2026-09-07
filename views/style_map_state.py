"""Presentation state helpers for the PCA style map page."""

from dataclasses import dataclass, field

from controllers.gui_dtos import PCAMapPoint, UploadedMapPoint
from services.style_translations import format_style_label
from visualization.pca_chart_models import PCAScatterPoint


@dataclass
class StyleMapState:
    """Hold loaded PCA map points and derive view-specific values."""

    points: list[PCAMapPoint] = field(default_factory=list)
    active_style: str | None = None

    def set_points(self, points: list[PCAMapPoint]) -> None:
        """Replace map points and drop filters that are no longer valid."""
        self.points = list(points)
        if self.active_style not in self.style_labels:
            self.active_style = None

    @property
    def style_labels(self) -> list[str]:
        """Return sorted style labels in the loaded map."""
        return sorted({point.style_label for point in self.points})

    @property
    def style_counts(self) -> dict[str, int]:
        """Return artwork counts grouped by style."""
        counts: dict[str, int] = {}
        for point in self.points:
            counts[point.style_label] = counts.get(point.style_label, 0) + 1
        return counts

    @property
    def has_3d_points(self) -> bool:
        """Return whether any point carries a third PCA coordinate."""
        return any(point.z_value is not None for point in self.points)

    @property
    def visible_count(self) -> int:
        """Return point count visible after the current style filter."""
        return len(self.visible_points())

    def status_text(self) -> str:
        """Return compact status text for the current map state."""
        if not self.points:
            return "尚未加载PCA地图。"
        return (
            f"已加载 {len(self.points)} 个作品点，"
            f"当前显示 {self.visible_count} 个，"
            f"覆盖 {len(self.style_labels)} 个风格。"
        )

    def visible_points(self) -> list[PCAMapPoint]:
        """Return points visible after style filtering."""
        if self.active_style is None:
            return list(self.points)
        return [
            point
            for point in self.points
            if point.style_label == self.active_style
        ]

    def hidden_labels(self) -> set[str]:
        """Return labels that should be hidden for the current style filter."""
        if self.active_style is None:
            return set()
        return {
            point.style_label
            for point in self.points
            if point.style_label != self.active_style
        }

    def find_point(self, query: str) -> PCAMapPoint | None:
        """Find the first artwork point matching ID, title, artist, or style."""
        normalized = query.strip().lower()
        if not normalized:
            return None
        for point in self.points:
            searchable = " ".join(
                (
                    str(point.artwork_id),
                    point.title,
                    point.artist_name,
                    point.style_label,
                    format_style_label(point.style_label),
                ),
            ).lower()
            if normalized in searchable:
                return point
        return None


def map_point_to_scatter_point(point: PCAMapPoint) -> PCAScatterPoint:
    """Convert one database-backed map point to chart format."""
    return PCAScatterPoint(
        sample_id=str(point.artwork_id),
        label=point.style_label,
        x_value=point.x_value,
        y_value=point.y_value,
        z_value=point.z_value,
        title=point.title,
        artist_name=point.artist_name,
        image_path=point.image_path,
    )


def uploaded_to_scatter_point(point: UploadedMapPoint) -> PCAScatterPoint:
    """Convert an uploaded image projection to chart format."""
    return PCAScatterPoint(
        sample_id="uploaded",
        label=point.style_label,
        x_value=point.x_value,
        y_value=point.y_value,
        z_value=point.z_value,
        title=point.title,
        artist_name=point.artist_name,
        image_path=point.image_path,
    )
