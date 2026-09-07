"""Persisted machine-learning artifacts for classification and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import joblib
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from core.pca import PCAModel, PCAResult

DEFAULT_MODEL_FILE_NAME = "artinsight_style_model.joblib"


@dataclass(frozen=True)
class ModelMetadata:
    """Serializable training metadata."""

    feature_type: str
    feature_methods: list[str]
    preprocessing_ops: list[str]
    n_components: int
    sample_count: int
    feature_count: int
    style_labels: list[str]


@dataclass(frozen=True)
class StyleModelArtifact:
    """Complete trained artifact used by classifier and retriever."""

    scaler: StandardScaler
    pca_model: PCAModel
    classifier: KNeighborsClassifier
    retriever: NearestNeighbors
    artwork_ids: list[int]
    labels: list[str]
    metadata: ModelMetadata

    @property
    def pca_result(self) -> PCAResult:
        """Return the fitted manual PCA result."""
        return self.pca_model.result


class ModelArtifactStore:
    """Thread-safe joblib persistence for trained style model artifacts."""

    def __init__(self, model_path: Path) -> None:
        """Create a store for one model artifact path."""
        self._model_path = model_path
        self._lock = RLock()

    @property
    def model_path(self) -> Path:
        """Return the configured model path."""
        return self._model_path

    def save(self, artifact: StyleModelArtifact) -> None:
        """Persist a trained model artifact with joblib."""
        with self._lock:
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(artifact, self._model_path)

    def load(self) -> StyleModelArtifact:
        """Load a trained model artifact from disk."""
        with self._lock:
            if not self._model_path.exists():
                msg = f"Model artifact does not exist: {self._model_path}"
                raise FileNotFoundError(msg)
            artifact = joblib.load(self._model_path)
        if not isinstance(artifact, StyleModelArtifact):
            msg = "Loaded artifact has an unexpected type."
            raise TypeError(msg)
        return artifact


def default_model_path(model_dir: Path) -> Path:
    """Build the default style model path inside the configured model directory."""
    return model_dir / DEFAULT_MODEL_FILE_NAME


def artifact_to_dict(artifact: StyleModelArtifact) -> dict[str, Any]:
    """Return a compact dictionary summary for logs and history records."""
    metadata = artifact.metadata
    return {
        "feature_type": metadata.feature_type,
        "feature_methods": metadata.feature_methods,
        "preprocessing_ops": metadata.preprocessing_ops,
        "n_components": metadata.n_components,
        "sample_count": metadata.sample_count,
        "feature_count": metadata.feature_count,
        "style_count": len(set(metadata.style_labels)),
    }