"""Item-based KNN.

Similarity: cosine similarity between movie rating vectors (columns of the
TRAINING user x movie rating matrix), computed with ``implicit``'s sparse,
multi-threaded all-pairs K-nearest-neighbours routine. Only the top
``KNN_NEIGHBORS`` neighbours of each movie are stored, so no dense
movie x movie matrix is ever built. Movies with fewer than
``KNN_MIN_ITEM_RATINGS`` training ratings get no neighbours (their cosine
similarities are dominated by one or two raters). The stored matrix is
symmetrised with max(S, S^T) and has a zero diagonal.

Scoring for a user with history H (KNN_SCORING):

* ``average`` (the proposal's Eq. 2, default):

      score(j) = sum_{i in H} S[i, j] * r_ui  /  sum_{i in H} |S[i, j]|

  i.e. a similarity-weighted average of the user's own ratings, a predicted
  rating on the 0.5-5 scale. Only history movies among j's stored neighbours
  contribute; a movie needs at least KNN_MIN_SUPPORT (default 2) of them,
  otherwise one 5-star neighbour would give a perfect 5. Ties (common, since
  many movies average to exactly the same value) are broken by the total
  similarity, so better-supported movies come first.

* ``sum``: score(j) = sum_{i in H+} r_ui * S[i, j], where H+ is the set of
  movies the user rated >= HIGH_RATING_THRESHOLD (all rated movies if none).

Movies the user already rated are excluded. Neither score is a probability.

Evidence for a candidate j is the list of history movies i in H+ with the
largest S[i, j].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from moviematch.models.common import top_k


@dataclass(frozen=True)
class Evidence:
    history_movie_idx: int
    similarity: float
    history_rating: float


def fit_item_knn(train: sp.csr_matrix, neighbors: int, min_item_ratings: int) -> sp.csr_matrix:
    from implicit.nearest_neighbours import CosineRecommender

    n_items = train.shape[1]
    counts = np.bincount(train.indices, minlength=n_items)
    supported = counts >= min_item_ratings
    matrix = train.copy().astype(np.float32)
    matrix.data[~supported[matrix.indices]] = 0.0
    matrix.eliminate_zeros()

    model = CosineRecommender(K=neighbors + 1)  # +1 because each movie is its own neighbour
    model.fit(matrix, show_progress=False)
    similarity = sp.csr_matrix(model.similarity, dtype=np.float32)
    if similarity.shape != (n_items, n_items):
        raise RuntimeError(f"Unexpected similarity shape {similarity.shape}")
    similarity.setdiag(0.0)
    similarity.eliminate_zeros()
    similarity = similarity.maximum(similarity.T).tocsr()
    similarity.sort_indices()
    return similarity.astype(np.float32)


def save_item_knn(similarity: sp.csr_matrix, path: Path, meta_path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, similarity, compressed=False)
    meta_path.write_text(json.dumps(meta, indent=2))


class ItemKnnModel:
    SCORINGS = ("average", "sum")

    def __init__(
        self,
        similarity: sp.csr_matrix,
        high_threshold: float,
        scoring: str = "sum",
        min_support: int = 2,
    ):
        if scoring not in self.SCORINGS:
            raise ValueError(f"Unknown KNN scoring {scoring}")
        self.similarity = similarity.tocsr()
        self.high_threshold = high_threshold
        self.scoring = scoring
        self.min_support = max(1, min_support)

    @classmethod
    def load(
        cls, path: Path, high_threshold: float, scoring: str = "sum", min_support: int = 2
    ) -> ItemKnnModel:
        return cls(sp.load_npz(path).tocsr().astype(np.float32), high_threshold, scoring, min_support)

    def with_scoring(self, scoring: str) -> ItemKnnModel:
        """Same similarity matrix, different scoring rule (no copy)."""
        return ItemKnnModel(self.similarity, self.high_threshold, scoring, self.min_support)

    def profile(self, history: np.ndarray, ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """History movies used for scoring/evidence, their weights, and
        whether they are all highly rated."""
        liked = ratings >= self.high_threshold
        if liked.any():
            return history[liked], ratings[liked], True
        return history, ratings, False

    def scores(self, history: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        """Score for every movie; -inf where there is not enough evidence."""
        n_items = self.similarity.shape[0]
        if self.scoring == "average":
            if len(history) == 0:
                return np.full(n_items, -np.inf)
            rows = self.similarity[history]
            weighted = np.asarray(rows.T @ ratings.astype(np.float64)).ravel()
            mass = np.asarray(rows.sum(axis=0)).ravel().astype(np.float64)
            indicator = rows.copy()
            indicator.data = np.ones_like(indicator.data)
            support = np.asarray(indicator.sum(axis=0)).ravel()
            out = np.full(n_items, -np.inf)
            ok = (support >= self.min_support) & (mass > 0)
            # Tie-break by similarity mass: the bonus is < 0.001 rating points.
            bonus = mass[ok] / (mass[ok].max() if ok.any() else 1.0) * 1e-3
            out[ok] = weighted[ok] / mass[ok] + bonus
            return out
        items, weights, _ = self.profile(history, ratings)
        if len(items) == 0:
            return np.full(n_items, -np.inf)
        rows = self.similarity[items]
        out = np.asarray(rows.T @ weights.astype(np.float64)).ravel()
        out[out <= 0] = -np.inf  # no similarity evidence at all
        return out

    def recommend(
        self, history: np.ndarray, ratings: np.ndarray, n: int
    ) -> list[tuple[int, float]]:
        scores = self.scores(history, ratings)
        picks = top_k(scores, n, exclude=history)
        return [(int(i), float(scores[i])) for i in picks]

    def evidence(
        self,
        history: np.ndarray,
        ratings: np.ndarray,
        candidates: list[int],
        per_candidate: int = 3,
    ) -> dict[int, list[Evidence]]:
        items, weights, _ = self.profile(history, ratings)
        result: dict[int, list[Evidence]] = {c: [] for c in candidates}
        if len(items) == 0 or not candidates:
            return result
        block = self.similarity[items][:, candidates].toarray()  # |H+| x |candidates|
        for col, candidate in enumerate(candidates):
            column = block[:, col]
            nonzero = np.flatnonzero(column > 0)
            if nonzero.size == 0:
                continue
            best = nonzero[np.lexsort((items[nonzero], -column[nonzero]))][:per_candidate]
            result[candidate] = [
                Evidence(int(items[b]), float(column[b]), float(weights[b])) for b in best
            ]
        return result
