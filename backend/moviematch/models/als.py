"""ALS matrix factorization (primary candidate generator).

Feedback treatment: IMPLICIT feedback derived from explicit ratings.
Every rated movie is an observed positive interaction (preference p_ui = 1)
with confidence

    c_ui = 1 + ALS_ALPHA * r_ui          (default ALS_ALPHA = 2.0)

so a 5-star rating counts more than a 1-star rating, while unrated movies
have preference 0 and confidence 1. The model is trained with the
``implicit`` library's conjugate-gradient ALS on the TRAINING split only.

Serving: item factors are fixed. The user's factor vector is re-solved from
the user's complete rating history (one least-squares step with the stored
item factors), so recommendations reflect everything the user has rated
while no model parameters are retrained. Already-rated movies are excluded.

Scores: the raw ALS score is the dot product x_u . y_i. The displayed
"ALS score" is min-max normalized over the user's candidate pool
(default 50 movies): the best candidate is 1.0, the weakest is 0.0. It is a
relative preference score, not a probability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from moviematch.models.common import min_max, top_k


@dataclass(frozen=True)
class AlsCandidate:
    movie_idx: int
    rank: int
    raw_score: float
    normalized_score: float


def confidence_matrix(ratings: sp.csr_matrix, alpha: float) -> sp.csr_matrix:
    confidence = ratings.copy().astype(np.float32)
    confidence.data = 1.0 + alpha * confidence.data
    return confidence


def fit_als(
    train: sp.csr_matrix,
    factors: int,
    regularization: float,
    alpha: float,
    iterations: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from implicit.cpu.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=1.0,  # confidence is already applied in confidence_matrix()
        iterations=iterations,
        random_state=random_seed,
    )
    model.fit(confidence_matrix(train, alpha), show_progress=False)
    return (
        np.ascontiguousarray(model.user_factors, dtype=np.float32),
        np.ascontiguousarray(model.item_factors, dtype=np.float32),
    )


def save_als(
    user_factors: np.ndarray, item_factors: np.ndarray, path: Path, meta_path: Path, meta: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, user_factors=user_factors, item_factors=item_factors)
    meta_path.write_text(json.dumps(meta, indent=2))


class AlsModel:
    def __init__(
        self,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        regularization: float,
        alpha: float,
    ):
        from implicit.cpu.als import AlternatingLeastSquares

        self.user_factors = user_factors
        self.item_factors = item_factors
        self.alpha = alpha
        self._solver = AlternatingLeastSquares(
            factors=item_factors.shape[1], regularization=regularization, alpha=1.0
        )
        self._solver.item_factors = item_factors
        self._solver.user_factors = user_factors

    @classmethod
    def load(cls, path: Path, meta_path: Path) -> AlsModel:
        meta = json.loads(meta_path.read_text())
        with np.load(path) as data:
            user_factors = data["user_factors"]
            item_factors = data["item_factors"]
        return cls(user_factors, item_factors, meta["regularization"], meta["alpha"])

    @property
    def n_items(self) -> int:
        return int(self.item_factors.shape[0])

    def user_vector(self, history: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        row = sp.csr_matrix(
            (1.0 + self.alpha * ratings.astype(np.float32), (np.zeros(len(history)), history)),
            shape=(1, self.n_items),
            dtype=np.float32,
        )
        return np.asarray(self._solver.recalculate_user(0, row), dtype=np.float32)

    def scores_for_vector(self, vector: np.ndarray) -> np.ndarray:
        return self.item_factors @ vector

    def candidates(
        self, history: np.ndarray, ratings: np.ndarray, pool_size: int
    ) -> list[AlsCandidate]:
        """Top ``pool_size`` unseen movies for a user, with normalized scores."""
        scores = self.scores_for_vector(self.user_vector(history, ratings))
        picks = top_k(scores, pool_size, exclude=history)
        raw = scores[picks].astype(np.float64)
        normalized = min_max(raw)
        return [
            AlsCandidate(int(i), rank + 1, float(raw[rank]), float(normalized[rank]))
            for rank, i in enumerate(picks)
        ]
