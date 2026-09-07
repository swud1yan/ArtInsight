"""Style classification service for uploaded artwork images."""

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
class StyleSimilarityScore:
    """Similarity share for one style in PCA space."""

    style_label: str
    score: float
    distance: float


@dataclass(frozen=True)
class ClassificationResult:
    """Predicted style label and confidence value."""

    style_label: str
    confidence: float
    style_scores: list[StyleSimilarityScore]


class StyleClassifier:
    """Classify an input image with a trained PCA + KNN pipeline."""

    def __init__(
        self,
        prediction_pipeline: PredictionPipeline,
        vectorizer: FeatureVectorizer | None = None,
    ) -> None:
        """Create a classifier from a shared prediction pipeline."""
        self._prediction_pipeline = prediction_pipeline
        self._vectorizer = vectorizer or ImageFeatureVectorizer()

    @classmethod
    def from_config(cls, model_path: Path | None = None) -> "StyleClassifier":
        """Create a classifier using the default application model path."""
        app_config = ConfigManager().load()
        resolved_model_path = model_path or default_model_path(
            app_config.paths.model_dir,
        )
        return cls(PredictionPipeline(ModelArtifactStore(resolved_model_path)))

    def classify(self, image: np.ndarray) -> ClassificationResult:
        """Classify an uploaded image.

        Args:
            image: Input image as a NumPy array using OpenCV-compatible shape.

        Returns:
            ``ClassificationResult`` with the predicted style and probability.
            When the classifier cannot expose probabilities, confidence falls
            back to ``0.0``.
        """
        artifact = self._prediction_pipeline.artifact
        feature_vector = self._vectorizer.vectorize(
            image,
            artifact.metadata.preprocessing_ops,
            artifact.metadata.feature_methods,
        )
        projected = self._prediction_pipeline.project_feature(feature_vector)
        result = self._build_result(projected)
        LOGGER.info(
            "Classified artwork image as %s with confidence %.4f.",
            result.style_label,
            result.confidence,
        )
        return result

    def classify_feature(self, feature_vector: np.ndarray) -> ClassificationResult:
        """Classify a precomputed raw feature vector.

        This method is primarily useful for tests and non-GUI service callers
        that already own the feature extraction step.
        """
        projected = self._prediction_pipeline.project_feature(feature_vector)
        return self._build_result(projected)

    def _build_result(self, projected: np.ndarray) -> ClassificationResult:
        """Build a classification result from projected PCA coordinates."""
        artifact = self._prediction_pipeline.artifact
        prediction = artifact.classifier.predict(projected)[0]
        predicted_label = str(prediction)
        result = ClassificationResult(
            style_label=predicted_label,
            confidence=self._predict_confidence(projected, predicted_label),
            style_scores=self._style_similarity_scores(projected),
        )
        return result

    def _predict_confidence(self, projected: np.ndarray, label: str) -> float:
        """Return the probability for a predicted label when available."""
        classifier = self._prediction_pipeline.artifact.classifier
        if not hasattr(classifier, "predict_proba"):
            return 0.0
        probabilities = classifier.predict_proba(projected)[0]
        classes = [str(item) for item in classifier.classes_]
        if label not in classes:
            return 0.0
        confidence = float(probabilities[classes.index(label)])
        return max(0.0, min(1.0, confidence))

    def _style_similarity_scores(self, projected: np.ndarray) -> list[StyleSimilarityScore]:
        """Return normalized style shares from the nearest sample of each style."""
        artifact = self._prediction_pipeline.artifact
        embedding = getattr(artifact.retriever, "_fit_X", None)
        if embedding is None or len(artifact.labels) == 0:
            return []

        coordinates = np.asarray(embedding, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[0] == 0:
            return []

        sample_distances = np.linalg.norm(
            coordinates - np.asarray(projected[0], dtype=np.float64),
            axis=1,
        )
        nearest_distances = {
            str(label): float("inf") for label in artifact.metadata.style_labels
        }
        for label, distance in zip(artifact.labels, sample_distances):
            style_label = str(label)
            if style_label in nearest_distances:
                nearest_distances[style_label] = min(
                    nearest_distances[style_label],
                    float(distance),
                )

        finite_distances = np.asarray(
            [
                distance
                for distance in nearest_distances.values()
                if np.isfinite(distance)
            ],
            dtype=np.float64,
        )
        if finite_distances.size == 0:
            return []

        temperature = max(float(np.median(finite_distances)) / 4.0, 1e-6)
        raw_scores = {
            label: float(np.exp(-distance / temperature))
            if np.isfinite(distance)
            else 0.0
            for label, distance in nearest_distances.items()
        }
        total_score = sum(raw_scores.values())
        if total_score <= 0.0:
            uniform = 1.0 / len(raw_scores)
            return [
                StyleSimilarityScore(label, uniform, nearest_distances[label])
                for label in sorted(raw_scores)
            ]
        return sorted(
            [
                StyleSimilarityScore(
                    style_label=label,
                    score=max(0.0, min(1.0, score / total_score)),
                    distance=nearest_distances[label],
                )
                for label, score in raw_scores.items()
            ],
            key=lambda item: item.score,
            reverse=True,
        )


def classify(image: np.ndarray) -> tuple[str, float]:
    """Convenience function returning ``(style_label, confidence)``."""
    result = StyleClassifier.from_config().classify(image)
    return result.style_label, result.confidence
