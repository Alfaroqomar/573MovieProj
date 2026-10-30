"""ALS on explicit ratings (the proposal's Eq. 1).

Minimises the regularised squared error on OBSERVED ratings only:

    min_{U,V}  sum_{(u,i) in Omega} (r_ui - u_u . v_i)^2  +  lambda * (n_u |u_u|^2 + n_i |v_i|^2)

The regulariser is scaled by each user's / movie's number of ratings
(n_u, n_i), the "ALS-WR" weighting that Apache Spark MLlib's ALS uses, which
is the implementation the proposal names. Each half-step solves one small
k x k linear system per user (then per movie) in closed form.

This is a plain NumPy implementation so it runs anywhere without Java or
Spark. The predicted score u_u . v_i is a predicted star rating. Ranking all
unseen movies by predicted rating is the classic way to turn an explicit
model into a top-N list; it is known to favour obscure movies with few but
high ratings, which is one of the things the evaluation measures.

Serving re-solves the user's vector from their full rating history with the
movie vectors held fixed (one least-squares step, no retraining), exactly as
for the implicit ALS model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from moviematch.models.als import AlsCandidate
from moviematch.models.common import min_max, top_k

Log = Callable[[str], None]


def _solve_side(
    matrix: sp.csr_matrix, fixed: np.ndarray, out: np.ndarray, regularization: float
) -> None:
    """Solve every row's least-squares problem given the other side's factors."""
    k = fixed.shape[1]
    eye = np.eye(k)
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    for row in range(matrix.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        n = end - start
        if n == 0:
            out[row] = 0.0
            continue
        f = fixed[indices[start:end]].astype(np.float64)
        a = f.T @ f
        a += (regularization * n) * eye
        out[row] = np.linalg.solve(a, f.T @ data[start:end].astype(np.float64))


def rmse(matrix: sp.csr_matrix, user_factors: np.ndarray, item_factors: np.ndarray) -> float:
    coo = matrix.tocoo()
    total, count, chunk = 0.0, 0, 2_000_000
    for s in range(0, coo.nnz, chunk):
        r, c, v = coo.row[s : s + chunk], coo.col[s : s + chunk], coo.data[s : s + chunk]
        pred = np.einsum("ij,ij->i", user_factors[r], item_factors[c])
        total += float(((v - pred) ** 2).sum())
        count += len(v)
    return float(np.sqrt(total / max(count, 1)))


def fit_als_explicit(
    train: sp.csr_matrix,
    factors: int,
    regularization: float,
    iterations: int,
    random_seed: int,
    log: Log = print,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(random_seed)
    n_users, n_items = train.shape
    users = np.zeros((n_users, factors), dtype=np.float32)
    items = (rng.standard_normal((n_items, factors)) * 0.1).astype(np.float32)
    # Movies with no training ratings stay at zero, so they are never recommended.
    items[np.diff(train.tocsc().indptr) == 0] = 0.0
    by_item = train.T.tocsr()
    history: list[float] = []
    for it in range(iterations):
        started = time.perf_counter()
        _solve_side(train, items, users, regularization)
        _solve_side(by_item, users, items, regularization)
        err = rmse(train, users, items)
        history.append(err)
        log(f"    explicit ALS iteration {it + 1}/{iterations}: train RMSE {err:.4f} ({time.perf_counter() - started:.0f}s)")
    return users, items, history


def save_als_explicit(
    user_factors: np.ndarray, item_factors: np.ndarray, path: Path, meta_path: Path, meta: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, user_factors=user_factors, item_factors=item_factors)
    meta_path.write_text(json.dumps(meta, indent=2))


class ExplicitAlsModel:
    def __init__(self, user_factors: np.ndarray, item_factors: np.ndarray, regularization: float):
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.regularization = regularization
        self._eye = np.eye(item_factors.shape[1])
        self._untrained = ~item_factors.any(axis=1)

    @classmethod
    def load(cls, path: Path, meta_path: Path) -> ExplicitAlsModel:
        meta = json.loads(meta_path.read_text())
        with np.load(path) as data:
            return cls(data["user_factors"], data["item_factors"], meta["regularization"])

    @property
    def n_items(self) -> int:
        return int(self.item_factors.shape[0])

    def user_vector(self, history: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        if len(history) == 0:
            return np.zeros(self.item_factors.shape[1], dtype=np.float32)
        f = self.item_factors[history].astype(np.float64)
        a = f.T @ f + (self.regularization * len(history)) * self._eye
        return np.linalg.solve(a, f.T @ ratings.astype(np.float64)).astype(np.float32)

    def scores(self, history: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        """Predicted star rating for every movie."""
        return self.item_factors @ self.user_vector(history, ratings)

    def candidates(
        self, history: np.ndarray, ratings: np.ndarray, pool_size: int
    ) -> list[AlsCandidate]:
        scores = self.scores(history, ratings)
        # Movies without training ratings have zero vectors: keep them out.
        scores[self._untrained] = -np.inf
        picks = top_k(scores, pool_size, exclude=history)
        raw = scores[picks].astype(np.float64)
        normalized = min_max(raw)
        return [
            AlsCandidate(int(i), rank + 1, float(raw[rank]), float(normalized[rank]))
            for rank, i in enumerate(picks)
        ]
