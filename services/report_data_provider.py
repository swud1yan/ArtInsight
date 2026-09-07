"""Collect application state for generated analysis reports."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from database.session import DatabaseSessionManager
from models import Artist, Artwork, Dataset, Feature, History, Style
from services.dataset_scope import enabled_artwork_ids_query
from services.ml.model_artifact import ModelArtifactStore
from services.report_models import (
    DatasetReportRow,
    PCAReportSummary,
    ReportSnapshot,
    TrainingReportSummary,
)
from services.style_translations import format_style_distribution

PROJECT_NAME = "ArtInsight"
PROJECT_SUBTITLE = "基于PCA的艺术风格分析与相似作品检索系统"


class ReportDataProvider:
    """Assemble a report snapshot from database records and model artifacts."""

    def __init__(
        self,
        session_manager: DatabaseSessionManager,
        artifact_store: ModelArtifactStore,
    ) -> None:
        """Create a report data provider."""
        self._session_manager = session_manager
        self._artifact_store = artifact_store

    def build_snapshot(self, report_name: str, author: str = "ArtInsight用户") -> ReportSnapshot:
        """Collect the current application state for one report export."""
        with self._session_manager.session_scope() as session:
            datasets = self._dataset_rows(session)
            raw_style_distribution = self._style_distribution(session)
            style_distribution = format_style_distribution(raw_style_distribution)
            artist_distribution = self._artist_distribution(session)
            example_paths = self._example_image_paths(session)
            latest_history = self._latest_training_history(session)
            dataset_count = int(
                session.scalar(
                    select(func.count(Dataset.id)).where(Dataset.is_enabled.is_(True)),
                )
                or 0,
            )
            artwork_count = int(
                session.scalar(
                    select(func.count(Artwork.id)).where(
                        Artwork.id.in_(enabled_artwork_ids_query()),
                    ),
                )
                or 0,
            )
            style_count = len(style_distribution)
            feature_count = int(
                session.scalar(
                    select(func.count(Feature.id)).where(
                        Feature.artwork_id.in_(enabled_artwork_ids_query()),
                    ),
                )
                or 0,
            )

        return ReportSnapshot(
            project_name=PROJECT_NAME,
            subtitle=PROJECT_SUBTITLE,
            report_name=report_name,
            author=author,
            created_at=datetime.now().astimezone(),
            dataset_count=dataset_count,
            artwork_count=artwork_count,
            style_count=style_count,
            feature_count=feature_count,
            datasets=datasets,
            style_distribution=style_distribution,
            artist_distribution=artist_distribution,
            pca_summary=self._pca_summary(),
            training_summary=self._training_summary(latest_history),
            example_image_paths=example_paths,
        )

    def _dataset_rows(self, session: object) -> list[DatasetReportRow]:
        """Return imported datasets with artwork counts."""
        statement = (
            select(Dataset, func.count(Artwork.id))
            .outerjoin(Artwork, Artwork.dataset_id == Dataset.id)
            .where(Dataset.is_enabled.is_(True))
            .group_by(Dataset.id)
            .order_by(Dataset.date.desc())
        )
        return [
            DatasetReportRow(
                name=dataset.name,
                path=dataset.path,
                artwork_count=int(count),
                imported_at=dataset.date,
            )
            for dataset, count in session.execute(statement)
        ]

    def _style_distribution(self, session: object) -> dict[str, int]:
        """Return artwork counts grouped by style."""
        statement = (
            select(Style.name, func.count(Artwork.id))
            .select_from(Artwork)
            .join(Style, Artwork.style_id == Style.id)
            .where(Artwork.id.in_(enabled_artwork_ids_query()))
            .group_by(Style.name)
            .order_by(func.count(Artwork.id).desc())
        )
        return {str(name): int(count) for name, count in session.execute(statement)}

    def _artist_distribution(self, session: object, limit: int = 10) -> dict[str, int]:
        """Return top artist counts for report charts."""
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

    def _example_image_paths(self, session: object, limit: int = 6) -> list[Path]:
        """Return existing artwork image paths for report examples."""
        statement = (
            select(Artwork)
            .options(joinedload(Artwork.style), joinedload(Artwork.artist))
            .where(Artwork.id.in_(enabled_artwork_ids_query()))
            .order_by(Artwork.id.desc())
            .limit(limit)
        )
        paths = [Path(artwork.image_path) for artwork in session.scalars(statement)]
        return [path for path in paths if path.exists()]

    def _latest_training_history(self, session: object) -> dict[str, object]:
        """Return details from the latest model training history record."""
        details = session.scalar(
            select(History.details)
            .where(History.action == "style_model_trained")
            .order_by(History.timestamp.desc())
            .limit(1),
        )
        if not details:
            return {}
        try:
            parsed = json.loads(details)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _pca_summary(self) -> PCAReportSummary:
        """Return PCA model metadata if a trained artifact is available."""
        try:
            artifact = self._artifact_store.load()
        except (FileNotFoundError, TypeError, ValueError):
            return PCAReportSummary(
                model_available=False,
                feature_type="未训练",
                n_components=0,
                sample_count=0,
                feature_count=0,
                explained_variance_ratio=[],
                cumulative_variance_ratio=[],
            )
        result = artifact.pca_result
        return PCAReportSummary(
            model_available=True,
            feature_type=artifact.metadata.feature_type,
            n_components=artifact.metadata.n_components,
            sample_count=artifact.metadata.sample_count,
            feature_count=artifact.metadata.feature_count,
            explained_variance_ratio=[
                float(value) for value in result.explained_variance_ratio
            ],
            cumulative_variance_ratio=[
                float(value) for value in result.cumulative_variance_ratio
            ],
        )

    def _training_summary(self, history_details: dict[str, object]) -> TrainingReportSummary:
        """Return training parameters and validation accuracy from history."""
        accuracy = history_details.get("cross_validation_accuracy")
        feature_methods = history_details.get("feature_methods", [])
        preprocessing_ops = history_details.get("preprocessing_ops", [])
        return TrainingReportSummary(
            cross_validation_accuracy=(
                float(accuracy) if isinstance(accuracy, int | float) else None
            ),
            feature_methods=(
                [str(item) for item in feature_methods]
                if isinstance(feature_methods, list)
                else []
            ),
            preprocessing_ops=(
                [str(item) for item in preprocessing_ops]
                if isinstance(preprocessing_ops, list)
                else []
            ),
            latest_action="style_model_trained" if history_details else "暂无训练记录",
        )
