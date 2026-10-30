"""Stage 1 of the proposal: the merged ALS + Item-KNN shortlist.

For each user (proposal, section V.A.4):

1. ALS ranks every unseen movie; Item-KNN does the same.
2. Each model's scores are min-max normalised to 0-1 over its own top
   STAGE1_POOL_SIZE (default 50) movies for this user.
3. The union of ALS's top STAGE1_PER_MODEL (default 30) and Item-KNN's top 30
   is taken. A movie on both lists gets the average of its two normalised
   scores; a movie on one list keeps that list's score.
4. If the overlap leaves fewer than 50 movies, the list is topped up with
   ALS ranks 31, 32, ... (STAGE1_TOP_UP, on by default) so the next stage
   always receives a full pool.
5. Movies are ordered by merged score (ties: ALS rank, then KNN rank).

The top 10 of this list is evaluated as the "Stage 1 (merged)" model, and
the whole list is the candidate pool handed to the LLM reranker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moviematch.models.common import min_max, top_k


@dataclass(frozen=True)
class Stage1Candidate:
    movie_idx: int
    rank: int
    merged_score: float
    als_score: float | None
    knn_score: float | None
    als_rank: int | None
    knn_rank: int | None
    source: str  # "both" | "als" | "knn" | "als_top_up"


def _ranked(scores: np.ndarray, exclude: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    picks = top_k(scores, n, exclude=exclude)
    return picks, min_max(scores[picks].astype(np.float64))


def merge_candidates(
    als_scores: np.ndarray,
    knn_scores: np.ndarray,
    history: np.ndarray,
    per_model: int = 30,
    pool_size: int = 50,
    top_up: bool = True,
) -> list[Stage1Candidate]:
    """Merge two full score vectors into the Stage 1 shortlist."""
    als_items, als_norm = _ranked(als_scores, history, pool_size)
    knn_items, knn_norm = _ranked(knn_scores, history, pool_size)

    als_info = {int(m): (r + 1, float(als_norm[r])) for r, m in enumerate(als_items)}
    knn_info = {int(m): (r + 1, float(knn_norm[r])) for r, m in enumerate(knn_items)}

    chosen: dict[int, str] = {}
    for m in als_items[:per_model]:
        chosen[int(m)] = "als"
    for m in knn_items[:per_model]:
        chosen[int(m)] = "both" if int(m) in chosen else "knn"
    if top_up:
        for m in als_items[per_model:]:
            if len(chosen) >= pool_size:
                break
            chosen.setdefault(int(m), "als_top_up")

    rows = []
    for m, source in chosen.items():
        a = als_info.get(m)
        k = knn_info.get(m)
        use_a = a is not None and source in ("als", "both", "als_top_up")
        use_k = k is not None and source in ("knn", "both")
        parts = ([a[1]] if use_a else []) + ([k[1]] if use_k else [])
        merged = float(np.mean(parts)) if parts else 0.0
        rows.append(
            (
                -merged,
                a[0] if a else 10**9,
                k[0] if k else 10**9,
                m,
                Stage1Candidate(
                    movie_idx=m,
                    rank=0,
                    merged_score=merged,
                    als_score=a[1] if a else None,
                    knn_score=k[1] if k else None,
                    als_rank=a[0] if a else None,
                    knn_rank=k[0] if k else None,
                    source=source,
                ),
            )
        )
    rows.sort(key=lambda r: r[:4])
    out = []
    for rank, row in enumerate(rows[:pool_size], start=1):
        c = row[4]
        out.append(Stage1Candidate(**{**c.__dict__, "rank": rank}))
    return out
