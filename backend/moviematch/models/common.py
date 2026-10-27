"""Small helpers shared by the recommenders."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def top_k(scores: np.ndarray, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Indices of the ``k`` highest finite scores, best first.

    ``exclude`` holds item indices that must never be returned (for example,
    movies the user has already rated). ``scores`` is not modified.
    """
    if k <= 0 or scores.size == 0:
        return np.zeros(0, dtype=np.int64)
    working = scores.astype(np.float64, copy=True)
    if exclude is not None and len(exclude):
        working[exclude] = -np.inf
    finite = np.isfinite(working)
    available = int(finite.sum())
    if available == 0:
        return np.zeros(0, dtype=np.int64)
    k = min(k, available)
    candidates = np.argpartition(-working, k - 1)[:k]
    candidates = candidates[np.isfinite(working[candidates])]
    # Stable ordering: score descending, then item index ascending.
    order = np.lexsort((candidates, -working[candidates]))
    return candidates[order]


def user_row(matrix: sp.csr_matrix, user_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """(movie indices, ratings) for one user from a CSR matrix."""
    start, end = matrix.indptr[user_idx], matrix.indptr[user_idx + 1]
    return matrix.indices[start:end].copy(), matrix.data[start:end].copy()


def min_max(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] within the given set. A constant set maps to 1.0."""
    if values.size == 0:
        return values.astype(np.float64)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo <= 1e-12:
        return np.ones_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)
