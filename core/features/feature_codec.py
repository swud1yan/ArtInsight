"""Serialization helpers for NumPy feature vectors."""

from __future__ import annotations

from io import BytesIO

import numpy as np

from core.features.utils import FloatVector


def serialize_feature_vector(vector: np.ndarray) -> bytes:
    """Serialize a one-dimensional feature vector to portable NumPy bytes.

    Args:
        vector: Numeric feature vector with shape ``(n_features,)``.

    Returns:
        Bytes produced by ``np.save`` with ``allow_pickle=False`` semantics.
    """
    feature = _validate_feature_vector(vector)
    buffer = BytesIO()
    np.save(buffer, feature, allow_pickle=False)
    return buffer.getvalue()


def deserialize_feature_vector(payload: bytes) -> FloatVector:
    """Deserialize a feature vector stored in the database.

    The preferred format is ``np.save`` bytes from ``serialize_feature_vector``.
    As a compatibility fallback, raw ``float32`` bytes are accepted.

    Args:
        payload: Binary database payload.

    Returns:
        A one-dimensional ``np.float32`` feature vector.
    """
    if not payload:
        msg = "feature payload must not be empty."
        raise ValueError(msg)

    try:
        with BytesIO(payload) as buffer:
            vector = np.load(buffer, allow_pickle=False)
    except (OSError, ValueError):
        vector = np.frombuffer(payload, dtype=np.float32)
    return _validate_feature_vector(vector)


def _validate_feature_vector(vector: np.ndarray) -> FloatVector:
    """Validate and copy a feature vector as ``np.float32``."""
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        msg = "feature vector must be one-dimensional."
        raise ValueError(msg)
    if array.size == 0:
        msg = "feature vector must not be empty."
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "feature vector must contain only finite values."
        raise ValueError(msg)
    return array.copy()