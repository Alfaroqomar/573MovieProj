"""Item-based KNN.

Similarity: cosine similarity between movie rating vectors (columns of the
TRAINING user x movie rating matrix), computed with ``implicit``'s sparse,
multi-threaded all-pairs K-nearest-neighbours routine. Only the top
``KNN_NEIGHBORS`` neighbours of each movie are stored, so no dense
movie x movie matrix is ever built. Movies with fewer than
``KNN_MIN_ITEM_RATINGS`` training ratings get no neighbours (their cosine
similarities are dominated by one or two raters). The stored matrix is
symmetrised with max(S, S^T) and has a zero diagonal.

Scoring for a user with history H:

    score(j) = sum_{i in H+} r_ui * S[i, j]

where H+ is the set of movies the user rated >= HIGH_RATING_THRESHOLD
(default 4.0). If the user has no such ratings, all rated movies are used.
Movies the user already rated are excluded. The KNN score is this raw
weighted similarity sum; it is not a probability and is not bounded by 1.

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
    def __init__(self, similarity: sp.csr_matrix, high_threshold: float):
        self.similarity = similarity.tocsr()
        self.high_threshold = high_threshold

    @classmethod
    def load(cls, path: Path, high_threshold: float) -> ItemKnnModel:
        return cls(sp.load_npz(path).tocsr().astype(np.float32), high_threshold)

    def profile(self, history: np.ndarray, ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """History movies used for scoring/evidence, their weights, and
        whether they are all highly rated."""
        liked = ratings >= self.high_threshold
        if liked.any():
            return history[liked], ratings[liked], True
        return history, ratings, False

    def scores(self, history: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        items, weights, _ = self.profile(history, ratings)
        n_items = self.similarity.shape[0]
        if len(items) == 0:
            return np.zeros(n_items, dtype=np.float64)
        rows = self.similarity[items]
        return np.asarray(rows.T @ weights.astype(np.float64)).ravel()

    def recommend(
        self, history: np.ndarray, ratings: np.ndarray, n: int
    ) -> list[tuple[int, float]]:
        scores = self.scores(history, ratings)
        scores[scores <= 0] = -np.inf  # no similarity evidence at all
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
