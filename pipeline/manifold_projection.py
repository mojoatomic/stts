"""Stage 3: Manifold projection M — per paper Section 4.4.

Projects causally-weighted features to a lower-dimensional embedding space
that preserves trajectory similarity.

IMPORTANT: The StandardScaler here normalizes raw features BEFORE causal
weighting. It must NOT be applied after weighting, as that would undo
the causal weight amplification. The pipeline order is:
  1. F(T) — extract raw features
  2. StandardScaler — equalize sensor scales
  3. W — apply causal weights (amplifies upstream features)
  4. M — optional dimensionality reduction (PCA/UMAP), NO additional scaling
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline.config import EMBEDDING_DIM, PROJECTION_METHOD


def fit_scaler(raw_features: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on raw (pre-weighted) features.

    This scaler equalizes sensor scales before causal weighting is applied.
    It must be fitted on training data and applied to both train and test.
    """
    scaler = StandardScaler()
    scaler.fit(raw_features)
    return scaler


def fit_projection(
    weighted_features: np.ndarray,
    method: str = PROJECTION_METHOD,
    dim: int = EMBEDDING_DIM,
) -> object | None:
    """Fit optional dimensionality reduction on causally-weighted features.

    No additional scaling is applied — the causal weights must be preserved
    in the distance computation.

    Args:
        weighted_features: (n_samples, feature_dim) — already scaled and weighted
        method: "pca", "umap", or "none"
        dim: target embedding dimension (ignored if method="none")

    Returns:
        Fitted projector, or None if method="none"
    """
    if method == "none":
        return None

    if method == "pca":
        projector = PCA(n_components=dim, random_state=42)
        projector.fit(weighted_features)
        return projector

    if method == "umap":
        import umap
        projector = umap.UMAP(
            n_components=dim,
            metric="euclidean",
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
        )
        projector.fit(weighted_features)
        return projector

    raise ValueError(f"Unknown projection method: {method}")


def project(
    weighted_features: np.ndarray,
    projector: object | None,
) -> np.ndarray:
    """Apply fitted projection (or pass through if projector is None).

    Args:
        weighted_features: (n_samples, feature_dim)
        projector: fitted PCA/UMAP, or None for no projection

    Returns:
        (n_samples, embedding_dim) array
    """
    if projector is None:
        return weighted_features
    return projector.transform(weighted_features)
