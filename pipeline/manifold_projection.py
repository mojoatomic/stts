"""Stage 3: Manifold projection M — per paper Section 4.4.

Projects causally-weighted features to a lower-dimensional embedding space
that preserves trajectory similarity.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline.config import EMBEDDING_DIM, PROJECTION_METHOD


def fit_projection(
    weighted_features: np.ndarray,
    method: str = PROJECTION_METHOD,
    dim: int = EMBEDDING_DIM,
) -> tuple[object, StandardScaler]:
    """Fit dimensionality reduction on training data.

    Pre-scales features before projection so PCA/UMAP operates on
    standardized inputs.

    Args:
        weighted_features: (n_samples, feature_dim)
        method: "pca" or "umap"
        dim: target embedding dimension

    Returns:
        (projector, scaler) — fitted transformer objects
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(weighted_features)

    if method == "pca":
        projector = PCA(n_components=dim, random_state=42)
        projector.fit(scaled)
    elif method == "umap":
        import umap
        projector = umap.UMAP(
            n_components=dim,
            metric="euclidean",
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
        )
        projector.fit(scaled)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    return projector, scaler


def project(
    weighted_features: np.ndarray,
    projector: object,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply fitted projection to (possibly new) weighted features.

    Args:
        weighted_features: (n_samples, feature_dim)
        projector: fitted PCA or UMAP object
        scaler: fitted StandardScaler

    Returns:
        (n_samples, embedding_dim) array
    """
    scaled = scaler.transform(weighted_features)
    return projector.transform(scaled)
