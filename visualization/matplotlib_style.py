"""Shared Matplotlib style configuration for ArtInsight charts."""

from __future__ import annotations

import matplotlib


def configure_matplotlib_fonts() -> None:
    """Configure fonts that can render Chinese labels in chart titles."""
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
