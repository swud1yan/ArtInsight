"""Dashboard and statistics queries for GUI pages."""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import func, select

from controllers.gui_dtos import AppSummary
from database.session import DatabaseSessionManager
from models import Artist, Artwork, Dataset, Feature, History, Style
from services.dataset_scope import enabled_artwork_ids_query
from services.ml.model_artifact import ModelArtifactStore

LOGGER = logging.getLogger(__name__)


class DashboardQuery:
    """Read dashboard counters and chart data for GUI pages."""

    def __init__(
        self,
        session_manager: DatabaseSessionManager,
        artifact_store: ModelArtifactStore,
    ) -> None:
        """Create dashboard query with database and model artifact access."""
        self._session_manager = session_manager
        self._artifact_store = artifact_store

    def get_summary(self) -> AppSummary:
        """Load dashboard counters and latest history action."""
        with self._session_manager.session_scope() as session:
            dataset_count = int(
                session.scalar(
                    select(func.count(Dataset.id)).where(Dataset.is_enabled.is_(True)),
                )
                or 0,
            )
            artwork_count = self._enabled_artwork_count(session)
            style_count = self._enabled_style_count(session)
            feature_count = self._enabled_feature_count(session)
            latest_dataset = session.scalar(
                select(Dataset)
                .where(Dataset.is_enabled.is_(True))
                .order_by(Dataset.date.desc())
                .limit(1),
            )
            latest_history = session.scalar(
                select(History).order_by(History.timestamp.desc()).limit(1),
            )
            latest_action = session.scalar(
                select(History.action).order_by(History.timestamp.desc()).limit(1),
            )
        pca_completed, component_count, cumulative_ratio = self._load_pca_status()
        accuracy = self._extract_accuracy(latest_history.details if latest_history else None)
        return AppSummary(
            dataset_count=dataset_count,
            artwork_count=artwork_count,
            style_count=style_count,
            feature_count=feature_count,
            model_available=self._artifact_store.model_path.exists(),
            latest_action=str(latest_action or "尚未记录实验操作"),
            pca_completed=pca_completed,
            pca_component_count=component_count,
            cumulative_variance_ratio=cumulative_ratio,
            latest_experiment_time=self._format_time(latest_history),
            latest_dataset_name=(
                latest_dataset.name if latest_dataset else "请先导入WikiArt数据集"
            ),
            latest_accuracy=accuracy,
        )

    def style_distribution(self) -> dict[str, int]:
        """Return artwork counts grouped by style."""
        with self._session_manager.session_scope() as session:
            statement = (
                select(Style.name, func.count(Artwork.id))
                .select_from(Artwork)
                .join(Style, Artwork.style_id == Style.id)
                .where(Artwork.id.in_(enabled_artwork_ids_query()))
                .group_by(Style.name)
                .order_by(Style.name)
            )
            return {str(name): int(count) for name, count in session.execute(statement)}

    def artist_distribution(self, limit: int = 12) -> dict[str, int]:
        """Return top artwork counts grouped by artist."""
        with self._session_manager.session_scope() as session:
            statement = (
                select(Artist.name, func.count(Artwork.id))
                .select_from(Artwork)
                .join(Artist, Artwork.artist_id == Artist.id)
                .where(Artwork.id.in_(enabled_artwork_ids_query()))
                .group_by(Artist.name)
                .order_by(func.count(Artwork.id).desc())
                .limit(limit)
            )
            return {str(name): int(count) for name, count in session.execute(statement)}

    def _enabled_artwork_count(self, session: Any) -> int:
        """Count artworks in enabled datasets."""
        return int(
            session.scalar(
                select(func.count(Artwork.id)).where(
                    Artwork.id.in_(enabled_artwork_ids_query()),
                ),
            )
            or 0,
        )

    def _enabled_style_count(self, session: Any) -> int:
        """Count distinct styles in enabled datasets."""
        return int(
            session.scalar(
                select(func.count(func.distinct(Artwork.style_id))).where(
                    Artwork.id.in_(enabled_artwork_ids_query()),
                ),
            )
            or 0,
        )

    def _enabled_feature_count(self, session: Any) -> int:
        """Count feature vectors attached to enabled datasets."""
        return int(
            session.scalar(
                select(func.count(Feature.id)).where(
                    Feature.artwork_id.in_(enabled_artwork_ids_query()),
                ),
            )
            or 0,
        )

    def _load_pca_status(self) -> tuple[bool, int, float | None]:
        """Return whether a trained PCA artifact exists and its main summary."""
        if not self._artifact_store.model_path.exists():
            return False, 0, None
        try:
            artifact = self._artifact_store.load()
            pca_result = artifact.pca_result
            component_count = int(artifact.metadata.n_components)
            if pca_result.cumulative_variance_ratio.size >= component_count:
                cumulative_ratio = float(
                    pca_result.cumulative_variance_ratio[component_count - 1],
                )
            else:
                cumulative_ratio = None
            return True, component_count, cumulative_ratio
        except (FileNotFoundError, TypeError, ValueError, AttributeError) as error:
            LOGGER.warning("Unable to read dashboard PCA status: %s", error)
            return False, 0, None

    def _extract_accuracy(self, details: str | None) -> float | None:
        """Read the latest cross-validation accuracy from a history JSON blob."""
        if not details:
            return None
        try:
            payload = json.loads(details)
        except json.JSONDecodeError:
            return None
        accuracy = payload.get("cross_validation_accuracy")
        return float(accuracy) if isinstance(accuracy, int | float) else None

    def _format_time(self, history: History | None) -> str:
        """Format the latest experiment timestamp for the dashboard."""
        if history is None:
            return "尚未开始"
        return history.timestamp.strftime("%Y-%m-%d %H:%M")
