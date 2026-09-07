"""View-state helpers for the interactive PCA analysis page."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controllers.gui_dtos import PCAExplanationData
from visualization.pca_chart_models import PCAScatterPoint


@dataclass(frozen=True)
class PCAComponentRow:
    """Table row for one PCA component."""

    index: int
    eigenvalue: float
    explained_ratio: float
    cumulative_ratio: float


@dataclass(frozen=True)
class PCAKMetrics:
    """Human-readable metrics for a selected component count."""

    component_count: int
    cumulative_ratio: float
    reconstruction_mse: float
    retained_text: str
    loss_text: str


class PCAAnalysisState:
    """Convert PCA explanation DTOs into table rows and selected-K views."""

    def __init__(self) -> None:
        """Create an empty PCA analysis state."""
        self._data: PCAExplanationData | None = None

    @property
    def data(self) -> PCAExplanationData | None:
        """Return the loaded PCA explanation data, if any."""
        return self._data

    @property
    def has_data(self) -> bool:
        """Return whether PCA explanation data has been loaded."""
        return self._data is not None

    def set_data(self, data: PCAExplanationData) -> None:
        """Replace the loaded PCA explanation data."""
        self._validate_data(data)
        self._data = data

    def default_component_count(self) -> int:
        """Return the best initial slider value for the loaded model."""
        data = self._require_data()
        return max(1, min(data.trained_components, data.max_components))

    def table_rows(self) -> list[PCAComponentRow]:
        """Return component rows for the PCA table."""
        data = self._require_data()
        return [
            PCAComponentRow(
                index=index + 1,
                eigenvalue=float(data.eigenvalues[index]),
                explained_ratio=float(data.explained_variance_ratio[index]),
                cumulative_ratio=float(data.cumulative_variance_ratio[index]),
            )
            for index in range(data.max_components)
        ]

    def summary_text(self) -> str:
        """Return a compact model summary."""
        data = self._require_data()
        return (
            f"当前模型训练时保留主成分数：{data.trained_components} / "
            f"{data.max_components}，样本数：{data.coordinates.shape[0]}。"
            "如累计解释方差偏低，请回到模型训练页调高 PCA 主成分后重新训练。"
        )

    def metrics_for(self, component_count: int) -> PCAKMetrics:
        """Return cumulative variance and reconstruction error for K."""
        data = self._require_data()
        k_value = self._clamp_component_count(component_count)
        cumulative = float(data.cumulative_variance_ratio[k_value - 1])
        mse = float(data.reconstruction_errors[k_value - 1])
        return PCAKMetrics(
            component_count=k_value,
            cumulative_ratio=cumulative,
            reconstruction_mse=mse,
            retained_text=f"累计解释方差：{cumulative:.2%}",
            loss_text=(
                f"K={k_value} 时，重建误差={mse:.6f}，"
                f"约保留 {cumulative:.2%} 的方差信息。"
            ),
        )

    def scatter_points(self, component_count: int) -> list[PCAScatterPoint]:
        """Return PCA scatter points for the selected K value."""
        data = self._require_data()
        k_value = self._clamp_component_count(component_count)
        coordinates = np.asarray(data.coordinates, dtype=np.float64)
        y_values = (
            coordinates[:, 1]
            if k_value >= 2 and coordinates.shape[1] >= 2
            else np.zeros(coordinates.shape[0], dtype=np.float64)
        )
        return [
            PCAScatterPoint(
                sample_id=str(index + 1),
                label=data.labels[index],
                x_value=float(coordinates[index, 0]),
                y_value=float(y_values[index]),
                title=data.titles[index],
            )
            for index in range(coordinates.shape[0])
        ]

    def scatter_title(self, component_count: int) -> str:
        """Return a chart title that explains the selected K projection."""
        k_value = self._clamp_component_count(component_count)
        if k_value == 1:
            return "PCA投影对比（K=1，PC2置零）"
        return f"PCA二维投影对比（当前K={k_value}）"

    def _clamp_component_count(self, component_count: int) -> int:
        """Clamp component count to the loaded PCA range."""
        data = self._require_data()
        return max(1, min(int(component_count), data.max_components))

    def _require_data(self) -> PCAExplanationData:
        """Return loaded data or raise a clear error."""
        if self._data is None:
            msg = "PCA explanation data has not been loaded."
            raise ValueError(msg)
        return self._data

    def _validate_data(self, data: PCAExplanationData) -> None:
        """Validate array and metadata shapes before rendering."""
        component_count = data.max_components
        if component_count < 1:
            msg = "PCA explanation data must contain at least one component."
            raise ValueError(msg)
        component_shapes = {
            data.explained_variance_ratio.shape[0],
            data.cumulative_variance_ratio.shape[0],
            data.reconstruction_errors.shape[0],
        }
        if component_shapes != {component_count}:
            msg = "PCA explanation vectors must have the same component count."
            raise ValueError(msg)
        if np.any(np.diff(data.cumulative_variance_ratio) < -1e-9):
            msg = "PCA cumulative variance must be monotonically increasing."
            raise ValueError(msg)
        coordinates = np.asarray(data.coordinates, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] < 1:
            msg = "PCA coordinates must be a 2-D matrix with at least PC1."
            raise ValueError(msg)
        sample_count = coordinates.shape[0]
        if len(data.labels) != sample_count or len(data.titles) != sample_count:
            msg = "PCA labels and titles must match coordinate rows."
            raise ValueError(msg)
