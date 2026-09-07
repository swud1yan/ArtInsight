"""Image analysis adapter for GUI pages."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from controllers.artwork_card_query import ArtworkCardQuery
from controllers.gui_dtos import (
    ClassificationViewResult,
    FeaturePreview,
    SimilarArtworkViewResult,
    StyleScoreView,
    UploadedMapPoint,
)
from core.features import FeatureExtractor
from core.preprocessing import ImagePreprocessor
from services.ml import SimilarArtworkRetriever, StyleClassifier
from services.ml.image_pipeline import ImageFeatureVectorizer
from services.ml.prediction_pipeline import PredictionPipeline


class ImageAnalysisAdapter:
    """Adapt classification, retrieval, and feature preview to GUI DTOs."""

    def __init__(
        self,
        prediction_pipeline: PredictionPipeline,
        artwork_query: ArtworkCardQuery,
    ) -> None:
        """Create an image-analysis adapter."""
        self._prediction_pipeline = prediction_pipeline
        self._artwork_query = artwork_query
        self._vectorizer = ImageFeatureVectorizer()
        self._classifier: StyleClassifier | None = None
        self._retriever: SimilarArtworkRetriever | None = None

    def classify_image(self, image_path: Path) -> ClassificationViewResult:
        """Classify an image file and return same-style examples."""
        image = self._read_image(image_path)
        result = self._get_classifier().classify(image)
        examples = self._artwork_query.list_examples(result.style_label, limit=8)
        return ClassificationViewResult(
            style_label=result.style_label,
            confidence=result.confidence,
            examples=examples,
            style_scores=[
                StyleScoreView(
                    style_label=score.style_label,
                    score=score.score,
                    distance=score.distance,
                )
                for score in result.style_scores
            ],
        )

    def retrieve_similar_image(
        self,
        image_path: Path,
        top_n: int = 10,
    ) -> list[SimilarArtworkViewResult]:
        """Retrieve similar artworks for an uploaded image."""
        image = self._read_image(image_path)
        results = self._get_retriever().retrieve_similar(image, top_n=top_n)
        cards = self._artwork_query.get_cards([item.artwork_id for item in results])
        card_by_id = {card.artwork_id: card for card in cards}
        return [
            SimilarArtworkViewResult(
                artwork=card_by_id[item.artwork_id],
                distance=item.distance,
            )
            for item in results
            if item.artwork_id in card_by_id
        ]

    def extract_feature_preview(
        self,
        image_path: Path,
        preprocessing_ops: list[str],
        feature_methods: list[str],
    ) -> FeaturePreview:
        """Extract a feature vector preview for a single image file."""
        image = self._read_image(image_path)
        processed = ImagePreprocessor().process(image, preprocessing_ops)
        vector = FeatureExtractor().extract(processed, feature_methods)
        first_values = [round(float(value), 6) for value in vector[:12]]
        return FeaturePreview(
            length=int(vector.shape[0]),
            first_values=first_values,
            dtype_name=str(vector.dtype),
        )

    def project_uploaded_image(self, image_path: Path) -> UploadedMapPoint:
        """Project an uploaded image into PCA space for map highlighting."""
        image = self._read_image(image_path)
        artifact = self._prediction_pipeline.artifact
        feature_vector = self._vectorizer.vectorize(
            image,
            artifact.metadata.preprocessing_ops,
            artifact.metadata.feature_methods,
        )
        projected = self._prediction_pipeline.project_feature(feature_vector)[0]
        prediction = self._get_classifier().classify(image)
        z_value = float(projected[2]) if projected.shape[0] >= 3 else None
        return UploadedMapPoint(
            x_value=float(projected[0]),
            y_value=float(projected[1]),
            z_value=z_value,
            style_label=prediction.style_label,
            artist_name="上传图片",
            confidence=prediction.confidence,
            image_path=str(image_path),
            title=image_path.name,
        )

    def _get_classifier(self) -> StyleClassifier:
        """Return a cached style classifier."""
        if self._classifier is None:
            self._classifier = StyleClassifier(self._prediction_pipeline)
        return self._classifier

    def _get_retriever(self) -> SimilarArtworkRetriever:
        """Return a cached similar artwork retriever."""
        if self._retriever is None:
            self._retriever = SimilarArtworkRetriever(self._prediction_pipeline)
        return self._retriever

    def _read_image(self, image_path: Path) -> np.ndarray:
        """Read an image file with OpenCV and raise user-friendly errors."""
        if not image_path.exists():
            msg = f"图片不存在：{image_path}"
            raise FileNotFoundError(msg)
        image = cv2.imread(str(image_path))
        if image is None:
            msg = f"无法读取图片：{image_path}"
            raise ValueError(msg)
        return image
