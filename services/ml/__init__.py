"""Machine-learning services for training, classification, and retrieval."""

from services.ml.classifier import ClassificationResult, StyleClassifier, classify
from services.ml.model_artifact import ModelArtifactStore, StyleModelArtifact
from services.ml.retriever import (
    RetrievalResult,
    SimilarArtworkRetriever,
    retrieve_similar,
)
from services.ml.trainer import ModelTrainer, train_model
from services.ml.training_config import TrainingConfig, TrainingResult

__all__ = [
    "ClassificationResult",
    "ModelArtifactStore",
    "ModelTrainer",
    "RetrievalResult",
    "SimilarArtworkRetriever",
    "StyleClassifier",
    "StyleModelArtifact",
    "TrainingConfig",
    "TrainingResult",
    "classify",
    "retrieve_similar",
    "train_model",
]