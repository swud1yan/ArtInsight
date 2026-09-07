"""PCA result adapter for GUI pages."""

from __future__ import annotations

from controllers.artwork_card_query import ArtworkCardQuery
from controllers.gui_dtos import (
    PCAExplanationData,
    PCALabStepData,
    PCAMapPoint,
    PCAViewData,
)
from database.session import DatabaseSessionManager
from services.ml.model_artifact import ModelArtifactStore
from services.ml.pca_explanation import PCAExplanationBuilder
from services.ml.training_data import TrainingDataLoader


class PCAViewAdapter:
    """Load trained PCA information as GUI DTOs."""

    def __init__(
        self,
        session_manager: DatabaseSessionManager,
        artifact_store: ModelArtifactStore,
        artwork_query: ArtworkCardQuery,
    ) -> None:
        """Create the PCA view adapter."""
        self._session_manager = session_manager
        self._artifact_store = artifact_store
        self._artwork_query = artwork_query
        self._explanation_builder = PCAExplanationBuilder()

    def get_pca_view_data(self) -> PCAViewData:
        """Load PCA eigenvalue and variance contribution data."""
        artifact = self._artifact_store.load()
        pca_result = artifact.pca_result
        return PCAViewData(
            eigenvalues=pca_result.eigenvalues.copy(),
            explained_variance_ratio=pca_result.explained_variance_ratio.copy(),
            cumulative_variance_ratio=pca_result.cumulative_variance_ratio.copy(),
            n_components=artifact.metadata.n_components,
        )

    def get_pca_explanation_data(self) -> PCAExplanationData:
        """Load PCA contribution, projection, and reconstruction summaries."""
        artifact = self._artifact_store.load()
        with self._session_manager.session_scope() as session:
            dataset = TrainingDataLoader(session).load(artifact.metadata.feature_type)
            cards = self._artwork_query.cards_by_id(session, dataset.artwork_ids)
        titles = [
            cards[artwork_id].title if artwork_id in cards else str(artwork_id)
            for artwork_id in dataset.artwork_ids
        ]
        explanation = self._explanation_builder.build(
            artifact=artifact,
            feature_matrix=dataset.feature_matrix,
            labels=dataset.style_labels,
            titles=titles,
        )
        return PCAExplanationData(
            eigenvalues=explanation.eigenvalues.copy(),
            explained_variance_ratio=explanation.explained_variance_ratio.copy(),
            cumulative_variance_ratio=explanation.cumulative_variance_ratio.copy(),
            reconstruction_errors=explanation.reconstruction_errors.copy(),
            coordinates=explanation.coordinates.copy(),
            labels=list(explanation.labels),
            titles=list(explanation.titles),
            trained_components=explanation.trained_components,
            lab_steps=[
                PCALabStepData(
                    title=step.title,
                    description=step.description,
                    column_headers=list(step.column_headers),
                    rows=[list(row) for row in step.rows],
                    note=step.note,
                )
                for step in explanation.lab_steps
            ],
        )

    def get_pca_map_points(self) -> list[PCAMapPoint]:
        """Return artworks projected to the first two PCA dimensions."""
        artifact = self._artifact_store.load()
        with self._session_manager.session_scope() as session:
            dataset = TrainingDataLoader(session).load(artifact.metadata.feature_type)
            scaled = artifact.scaler.transform(dataset.feature_matrix)
            coordinates = artifact.pca_model.transform(scaled)
            cards = self._artwork_query.cards_by_id(session, dataset.artwork_ids)
        if coordinates.shape[1] < 2:
            msg = "PCA模型至少需要包含两个主成分才能绘制二维地图。"
            raise ValueError(msg)
        return [
            PCAMapPoint(
                artwork_id=artwork_id,
                x_value=float(coordinates[index, 0]),
                y_value=float(coordinates[index, 1]),
                z_value=(
                    float(coordinates[index, 2])
                    if coordinates.shape[1] >= 3
                    else None
                ),
                style_label=dataset.style_labels[index],
                image_path=cards[artwork_id].image_path,
                title=cards[artwork_id].title,
                artist_name=cards[artwork_id].artist_name,
            )
            for index, artwork_id in enumerate(dataset.artwork_ids)
            if artwork_id in cards
        ]
