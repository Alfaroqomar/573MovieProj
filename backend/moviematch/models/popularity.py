"""Popularity baselines.

Two rules, both computed on the TRAINING data and both non-personalised
(the same list for everyone, minus movies the user has already rated):

* ``count`` (the proposal's baseline, mode ``popularity``): movies ranked by
  number of training ratings; ties broken by mean rating, then movie index.
* ``weighted`` (mode ``popularity_weighted``): the Bayesian weighted rating
  below.

Weighted score (a Bayesian / IMDb-style weighted rating):

    WR(i) = v_i / (v_i + m) * R_i  +  m / (v_i + m) * C

    v_i  number of training ratings for movie i
    R_i  mean training rating for movie i
    C    mean of all training ratings
    m    minimum-votes threshold = the POPULARITY_MIN_VOTES_QUANTILE quantile
         (default 0.90) of v_i over movies with at least one training rating

Movies with v_i >= m are ranked first (widely rated and well rated). So that
very active users who have already rated all of them still get results, the
remaining rated movies follow, also ordered by WR. Ties are broken by v_i,
then movie index. Movies the selected user has already rated are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class PopularityItem:
    movie_idx: int
    weighted_rating: float
    rating_count: int
    mean_rating: float


def fit_popularity(train: sp.csr_matrix, min_votes_quantile: float) -> dict[str, np.ndarray]:
    n_items = train.shape[1]
    counts = np.bincount(train.indices, minlength=n_items).astype(np.int64)
    sums = np.bincount(train.indices, weights=train.data, minlength=n_items)
    means = np.divide(sums, counts, out=np.zeros(n_items), where=counts > 0)
    rated = counts > 0
    if not rated.any():
        raise ValueError("The training split has no ratings.")
    global_mean = float(train.data.mean())
    m = max(1.0, float(np.quantile(counts[rated], min_votes_quantile)))
    weighted = np.where(rated, counts / (counts + m) * means + m / (counts + m) * global_mean, 0.0)

    def ranked(items: np.ndarray) -> np.ndarray:
        return items[np.lexsort((items, -counts[items], -weighted[items]))]

    eligible = ranked(np.flatnonzero(counts >= m))
    remainder = ranked(np.flatnonzero(rated & (counts < m)))
    order = np.concatenate((eligible, remainder))
    rated_items = np.flatnonzero(rated)
    count_order = rated_items[np.lexsort((rated_items, -means[rated_items], -counts[rated_items]))]
    return {
        "order": order.astype(np.int32),
        "count_order": count_order.astype(np.int32),
        "weighted_rating": weighted.astype(np.float32),
        "counts": counts.astype(np.int32),
        "means": means.astype(np.float32),
        "global_mean": np.array(global_mean),
        "min_votes": np.array(m),
        "min_votes_quantile": np.array(min_votes_quantile),
        "eligible_count": np.array(len(eligible)),
    }


def save_popularity(arrays: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


class PopularityModel:
    RULES = ("count", "weighted")

    def __init__(self, arrays: dict[str, np.ndarray], rule: str = "weighted"):
        if rule not in self.RULES:
            raise ValueError(f"Unknown popularity rule {rule}")
        self.rule = rule
        self.order = arrays["count_order"] if rule == "count" else arrays["order"]
        self.weighted_rating = arrays["weighted_rating"]
        self.counts = arrays["counts"]
        self.means = arrays["means"]
        self.global_mean = float(arrays["global_mean"])
        self.min_votes = float(arrays["min_votes"])
        self.min_votes_quantile = float(arrays["min_votes_quantile"])
        self.eligible_count = int(arrays["eligible_count"])

    @classmethod
    def load(cls, path: Path, rule: str = "weighted") -> PopularityModel:
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        if rule == "count" and "count_order" not in arrays:
            raise FileNotFoundError("Popularity model predates the most-rated rule; retrain it.")
        return cls(arrays, rule)

    def ranking(self, exclude: np.ndarray, n: int) -> np.ndarray:
        """Movie indices only (fast path used by evaluation)."""
        if len(exclude) == 0:
            return self.order[:n].astype(np.int64)
        excluded = np.zeros(len(self.counts), dtype=bool)
        excluded[exclude] = True
        out = []
        for idx in self.order:
            if not excluded[idx]:
                out.append(int(idx))
                if len(out) >= n:
                    break
        return np.asarray(out, dtype=np.int64)

    def recommend(self, exclude: np.ndarray, n: int) -> list[PopularityItem]:
        excluded = set(int(i) for i in exclude)
        items: list[PopularityItem] = []
        for idx in self.order:
            i = int(idx)
            if i in excluded:
                continue
            items.append(
                PopularityItem(
                    movie_idx=i,
                    weighted_rating=float(self.weighted_rating[i]),
                    rating_count=int(self.counts[i]),
                    mean_rating=float(self.means[i]),
                )
            )
            if len(items) >= n:
                break
        return items

    def describe(self) -> dict[str, object]:
        if self.rule == "count":
            return {"formula": "rank by number of training ratings", "rule": "count"}
        return {
            "rule": "weighted",
            "formula": "WR = v/(v+m)*R + m/(v+m)*C",
            "globalMeanC": round(self.global_mean, 4),
            "minVotesM": round(self.min_votes, 1),
            "minVotesQuantile": self.min_votes_quantile,
            "eligibleMovies": self.eligible_count,
        }
