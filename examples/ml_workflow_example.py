"""Example calls for training, classification, and similar artwork retrieval."""

from __future__ import annotations

import logging

import cv2

from services.ml import (
    ModelTrainer,
    SimilarArtworkRetriever,
    StyleClassifier,
    TrainingConfig,
)

LOGGER = logging.getLogger(__name__)


def run_example(image_path: str) -> None:
    """Train a model, classify one image, and retrieve similar artworks."""
    config = TrainingConfig(n_components=8, n_neighbors=5)
    training_result = ModelTrainer.from_config(config).train_model(config)
    LOGGER.info("Model saved to %s", training_result.model_path)
    LOGGER.info(
        "Cross-validation accuracy: %s",
        training_result.cross_validation_accuracy,
    )

    image = cv2.imread(image_path)
    if image is None:
        msg = f"Could not read image: {image_path}"
        raise FileNotFoundError(msg)

    classification = StyleClassifier.from_config(
        training_result.model_path,
    ).classify(image)
    LOGGER.info(
        "Predicted style=%s confidence=%.4f",
        classification.style_label,
        classification.confidence,
    )

    results = SimilarArtworkRetriever.from_config(
        training_result.model_path,
    ).retrieve_similar(image, top_n=10)
    LOGGER.info(
        "Top similar artwork IDs: %s",
        [(item.artwork_id, round(item.distance, 4)) for item in results],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    LOGGER.info("Call run_example('/path/to/query.jpg') after features exist in DB.")