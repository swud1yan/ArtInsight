"""Example workflow for the manual PCA implementation."""

from __future__ import annotations

import logging

import numpy as np

from core.pca import (
    PCAModel,
    PCAConfig,
    compute_covariance,
    compute_eigen,
    compute_mean,
    explained_variance,
    inverse_transform,
    mean_center,
    reconstruction_error,
    select_components,
    transform,
)

LOGGER = logging.getLogger(__name__)


def run_example() -> None:
    """Run PCA step by step on a deterministic synthetic matrix."""
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(120, 3))
    noise = rng.normal(scale=0.05, size=(120, 2))
    data = np.column_stack(
        [
            latent[:, 0] + 0.2 * latent[:, 1],
            latent[:, 1],
            latent[:, 2] - 0.1 * latent[:, 0],
            0.5 * latent[:, 0] + noise[:, 0],
            0.3 * latent[:, 2] + noise[:, 1],
        ],
    )

    mean = compute_mean(data)
    centered = mean_center(data)
    covariance = compute_covariance(centered)
    eigenvalues, eigenvectors = compute_eigen(covariance)
    ratio, cumulative = explained_variance(eigenvalues)
    projection = select_components(eigenvalues, eigenvectors, n_components=2)
    transformed = transform(centered, projection)
    reconstructed = inverse_transform(transformed, projection, mean)
    error = reconstruction_error(data, reconstructed)

    LOGGER.info("Covariance shape: %s", covariance.shape)
    LOGGER.info("First three eigenvalues: %s", np.round(eigenvalues[:3], 4).tolist())
    LOGGER.info("First three explained ratios: %s", np.round(ratio[:3], 4).tolist())
    LOGGER.info("Cumulative ratio for two components: %.4f", cumulative[1])
    LOGGER.info("Transformed shape: %s", transformed.shape)
    LOGGER.info("Reconstruction MSE with two components: %.6f", error)

    model = PCAModel(PCAConfig(n_components=2))
    result = model.fit(data)
    LOGGER.info("Model reconstruction MSE: %.6f", result.reconstruction_mse)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_example()
