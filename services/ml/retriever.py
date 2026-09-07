"""Similar artwork retrieval service using PCA-space nearest neighbors."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from app.config import ConfigManager
from services.ml.image_pipeline import FeatureVectorizer, ImageFeatureVectorizer
from services.ml.model_artifact import ModelArtifactStore, default_model_path
from services.ml.prediction_pipeline import PredictionPipeline

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """One similar artwork result sorted by ascending distance."""

    artwork_id: int
    distance: float
    style_label: str


class SimilarArtworkRetriever:
    """Retrieve visually similar artworks with a trained PCA retrieval index."""

    def __init__(
        self,
        prediction_pipeline: PredictionPipeline,
        vectorizer: FeatureVectorizer | None = None,
    ) -> None:
        """Create a retriever from a shared prediction pipeline."""
        self._prediction_pipeline = prediction_pipeline
        self._vectorizer = vectorizer or ImageFeatureVectorizer()

    @classmethod
    def from_config(cls, model_path: Path | None = None) -> "SimilarArtworkRetriever":
        """Create a retriever using the default application model path."""
        app_config = ConfigManager().load()
        resolved_model_path = model_path or default_model_path(
            app_config.paths.model_dir,
        )
        return cls(PredictionPipeline(ModelArtifactStore(resolved_model_path)))

    def retrieve_similar(
        self,
        image: np.ndarray,
        top_n: int = 10,
        offset: int = 0,
    ) -> list[RetrievalResult]:
        """Retrieve similar artworks for an uploaded image.

        Args:
            image: Input image as a NumPy array.
            top_n: Maximum number of results to return.
            offset: Number of nearest results to skip for simple pagination.

        Returns:
            Ordered ``RetrievalResult`` objects.
        """
        artifact = self._prediction_pipeline.artifact
        feature_vector = self._vectorizer.vectorize(
            image,
            artifact.metadata.preprocessing_ops,
            artifact.metadata.feature_methods,
        )
        return self.retrieve_feature(feature_vector, top_n=top_n, offset=offset)

    def retrieve_feature(
        self,
        feature_vector: np.ndarray,
        top_n: int = 10,
        offset: int = 0,
    ) -> list[RetrievalResult]:
        """Retrieve similar artworks for a precomputed raw feature vector."""
        self._validate_pagination(top_n, offset)
        artifact = self._prediction_pipeline.artifact
        total_count = len(artifact.artwork_ids)
        if total_count == 0:
            return []

        projected = self._prediction_pipeline.project_feature(feature_vector)
        query_count = min(total_count, offset + top_n)
        distances, indices = artifact.retriever.kneighbors(
            projected,
            n_neighbors=query_count,
            return_distance=True,
        )
        results = self._build_results(distances[0], indices[0], offset, top_n)
        LOGGER.info(
            "Retrieved %s similar artworks with offset=%s.",
            len(results),
            offset,
        )
        return results

    def _build_results(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        offset: int,
        top_n: int,
    ) -> list[RetrievalResult]:
        """Convert neighbor arrays to retrieval DTOs."""
        artifact = self._prediction_pipeline.artifact
        selected = zip(
            distances[offset : offset + top_n],
            indices[offset : offset + top_n],
        )
        return [
            RetrievalResult(
                artwork_id=int(artifact.artwork_ids[int(index)]),
                distance=float(distance),
                style_label=str(artifact.labels[int(index)]),
            )
            for distance, index in selected
        ]

    def _validate_pagination(self, top_n: int, offset: int) -> None:
        """Validate retrieval pagination arguments."""
        if top_n <= 0:
            msg = "top_n must be positive."
            raise ValueError(msg)
        if offset < 0:
            msg = "offset must not be negative."
            raise ValueError(msg)


def retrieve_similar(image: np.ndarray, top_n: int = 10) -> list[tuple[int, float]]:
    """Convenience function returning ``(artwork_id, distance)`` tuples."""
    results = SimilarArtworkRetriever.from_config().retrieve_similar(image, top_n)
    return [(result.artwork_id, result.distance) for result in results]