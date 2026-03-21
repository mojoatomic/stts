"""Failure basin construction and STTS monitoring query.

Implements Definitions 4 (failure basin), 6 (monitoring query), and
7 (OOD detection) from the paper.
"""

from __future__ import annotations

import numpy as np
import faiss


def build_failure_basin(
    embeddings: np.ndarray,
    rul: np.ndarray,
    rul_threshold: int,
) -> np.ndarray:
    """Extract failure basin B_f: embeddings where RUL <= threshold.

    Args:
        embeddings: (n_samples, embedding_dim)
        rul: (n_samples,) remaining useful life
        rul_threshold: trajectories with RUL <= this enter B_f

    Returns:
        basin_embeddings: (n_basin, embedding_dim)
    """
    mask = rul <= rul_threshold
    return embeddings[mask].copy()


def build_index(points: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS L2 index over a set of embeddings.

    Args:
        points: (n_points, embedding_dim) float32 array

    Returns:
        FAISS index ready for search
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    dim = points.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(points)
    return index


def distance_to_basin(
    query: np.ndarray,
    index: faiss.IndexFlatL2,
    k: int = 5,
) -> np.ndarray:
    """Compute distance from each query point to the failure basin.

    Uses mean of k-nearest-neighbor distances for robustness.

    Args:
        query: (n_queries, embedding_dim)
        index: FAISS index built on B_f
        k: number of nearest neighbors

    Returns:
        distances: (n_queries,) mean k-NN distance to basin
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    k_actual = min(k, index.ntotal)
    dists, _ = index.search(query, k_actual)
    # FAISS returns squared L2 distances
    return np.sqrt(dists).mean(axis=1)


def distance_to_corpus(
    query: np.ndarray,
    corpus_index: faiss.IndexFlatL2,
    k: int = 5,
) -> np.ndarray:
    """OOD signal: distance to k-th nearest neighbor in the full corpus.

    Large values indicate the query is in a region the corpus has never seen.

    Args:
        query: (n_queries, embedding_dim)
        corpus_index: FAISS index built on the entire training corpus
        k: k-th neighbor distance to use

    Returns:
        ood_distances: (n_queries,) k-th neighbor distance
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    k_actual = min(k, corpus_index.ntotal)
    dists, _ = corpus_index.search(query, k_actual)
    # Return k-th neighbor distance (last column)
    return np.sqrt(dists[:, -1])


def monitoring_query(distance: float, epsilon: float) -> bool:
    """Definition 6: is the trajectory approaching the failure basin?

    Args:
        distance: distance to B_f
        epsilon: alert threshold

    Returns:
        True if approaching failure basin
    """
    return distance < epsilon
