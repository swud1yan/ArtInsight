"""Shared prediction-time projection pipeline."""

from __future__ import annotations

import numpy as np

from services.ml.model_artifact import ModelArtifactStore, StyleModelArtifact


class PredictionPipeline:
    """Load a trained artifact and project feature vectors into PCA space."""

    def __init__(self, artifact_store: ModelArtifactStore) -> None:
        """Create a pipeline backed by a model artifact store."""
        self._artifact_store = artifact_store
        self._artifact: StyleModelArtifact | None = None
        self._loaded_mtime_ns: int | None = None

    @property
    def artifact(self) -> StyleModelArtifact:
        """Return a cached trained artifact, loading it on first use."""
        model_path = self._artifact_store.model_path
        current_mtime = model_path.stat().st_mtime_ns if model_path.exists() else None
        if self._artifact is None or current_mtime != self._loaded_mtime_ns:
            self._artifact = self._artifact_store.load()
            self._loaded_mtime_ns = current_mtime
        return self._artifact

    def project_feature(self, feature_vector: np.ndarray) -> np.ndarray:
        """Standardize and project one feature vector into PCA coordinates.

        Args:
            feature_vector: One-dimensional raw feature vector.

        Returns:
            PCA coordinates with shape ``(1, n_components)``.
        """
        artifact = self.artifact
        feature = np.asarray(feature_vector, dtype=np.float32)
        if feature.ndim != 1:
            msg = "feature_vector must be one-dimensional."
            raise ValueError(msg)
        if feature.shape[0] != artifact.metadata.feature_count:
            msg = (
                "feature length does not match the trained model; "
                f"expected {artifact.metadata.feature_count}, got {feature.shape[0]}."
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(feature)):
            msg = "feature_vector must contain only finite values."
            raise ValueError(msg)

        scaled = artifact.scaler.transform(feature.reshape(1, -1))
        return artifact.pca_model.transform(scaled)
