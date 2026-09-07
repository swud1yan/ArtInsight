"""DTOs for PCA visualization charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PCAScatterPoint:
    """One sample point displayed in a PCA scatter chart."""

    sample_id: str
    label: str
    x_value: float
    y_value: float
    z_value: float | None = None
    title: str = ""
    artist_name: str = ""
    image_path: str = ""


@dataclass(frozen=True)
class HoverAnnotation:
    """Connection metadata for a hover annotation event handler."""

    connection_id: int
    annotation: Any
