"""Export Stage 1 shortlists for the LLM stage (handoff file).

Run with:  python -m moviematch.cli candidates [--split test] [--users 500]

Writes two files to ``artifacts/candidates/``:

* ``<split>_candidates.parquet`` — one row per (user, candidate):
  user_id, rank, movie_id, title, year, genres, tags, merged_score,
  als_score, knn_score, als_rank, knn_rank, source, knn_evidence (JSON),
  is_cold_start
* ``<split>_users.jsonl`` — one line per user: user_id, history size,
  is_cold_start, favourite genres, highly rated movies, the 5 most recently
  rated movies (before the evaluated period) and the user's relevant
  (held-out, >= 4.0) movie IDs for scoring the reranked lists.

Users are the same fixed random sample the evaluation uses (seeded), with
cold-start users included first so they are always represented.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from moviematch.config import Settings
from moviematch.data.paths import candidates_dir, model_paths, processed_paths
from moviematch.data.preprocess import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION, build_matrix
from moviematch.data.preprocess import normalize_tag
from moviematch.models.als import AlsModel
from moviematch.models.common import top_k, user_row
from moviematch.models.item_knn import ItemKnnModel
from moviematch.models.stage1 import merge_candidates

Log = Callable[[str], None]


def favorite_genres_from_history(
    history: np.ndarray, ratings: np.ndarray, movie_genres: np.ndarray, threshold: float, limit: int = 5
) -> list[str]:
    liked = history[ratings >= threshold]
    source = liked if len(liked) else history
    counts = Counter(
        genre
        for movie_idx in source
        for genre in movie_genres[int(movie_idx)].split("|")
        if genre and genre != "(no genres listed)"
    )
    return [genre for genre, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def historical_tags(path: Path, cutoff: str, limit: int) -> dict[int, list[str]]:
    frame = pd.read_csv(path, usecols=["userId", "movieId", "tag", "timestamp"])
    cutoff_ts = int(pd.Timestamp(cutoff, tz="UTC").timestamp())
    frame = frame[(frame["timestamp"] < cutoff_ts) & frame["tag"].notna()].copy()
    frame["tag"] = frame["tag"].astype(str).map(normalize_tag)
    frame = frame[frame["tag"].str.len() > 0].drop_duplicates(["userId", "movieId", "tag"])
    counts = frame.groupby(["movieId", "tag"]).size().rename("count").reset_index()
    counts = counts.sort_values(["movieId", "count", "tag"], ascending=[True, False, True])
    return counts.groupby("movieId")["tag"].apply(lambda values: values.head(limit).tolist()).to_dict()


def export_candidates(settings: Settings, split_name: str = "test", n_users: int = 500, log: Log = print) -> dict:
    pp, mp = processed_paths(settings), model_paths(settings)
    if not (mp.als.exists() and mp.knn.exists()):
        raise RuntimeError("Stage 1 needs the ALS and Item-KNN models. Run the training command first.")
    target = SPLIT_TEST if split_name == "test" else SPLIT_VALIDATION
    history_splits = [SPLIT_TRAIN] if target == SPLIT_VALIDATION else [SPLIT_TRAIN, SPLIT_VALIDATION]

    ratings = pd.read_parquet(pp.ratings)
    n_u, n_i = sp.load_npz(pp.train_matrix).shape
    hist = ratings[ratings["split"].isin(history_splits)]
    history = build_matrix(hist["user_idx"].to_numpy(), hist["movie_idx"].to_numpy(), hist["rating"].to_numpy(), n_u, n_i)
    recent = (
        hist.sort_values(["user_idx", "timestamp", "movie_idx"]).groupby("user_idx").tail(5)
    )
    del hist
    rel = ratings[(ratings["split"] == target) & (ratings["rating"] >= settings.high_rating_threshold)]
    relevant = rel.groupby("user_idx")["movie_idx"].apply(list)
    del ratings, rel

    counts = np.diff(history.indptr)
    eligible = relevant.index.to_numpy()
    eligible = eligible[counts[eligible] > 0]
    cold = eligible[(counts[eligible] <= settings.cold_start_max_ratings)]
    rng = np.random.default_rng(settings.eval_random_seed)
    chosen_cold = rng.choice(cold, size=min(len(cold), n_users // 5), replace=False) if len(cold) else np.zeros(0, int)
    rest = np.setdiff1d(eligible, chosen_cold)
    chosen_rest = rng.choice(rest, size=min(len(rest), n_users - len(chosen_cold)), replace=False)
    chosen = np.sort(np.concatenate([chosen_cold, chosen_rest]).astype(np.int64))

    movies = pd.read_parquet(pp.movies)
    titles = movies["title"].astype(str).to_numpy()
    years = movies["year"].to_numpy()
    genres = movies["genres"].astype(str).to_numpy()
    movie_ids = movies["movieId"].to_numpy()
    if settings.split_mode == "date" and settings.movielens_data_path is not None:
        cutoff = settings.split_validation_start if target == SPLIT_VALIDATION else settings.split_test_start
        tags_by_id = historical_tags(settings.movielens_data_path / "tags.csv", cutoff, settings.tags_per_movie)
        tags = {int(movie_id): values for movie_id, values in tags_by_id.items()}
    else:
        movie_tags = pd.read_parquet(pp.movie_tags).groupby("movie_idx")["tag"].apply(list).to_dict()
        tags = {int(movie_ids[idx]): values for idx, values in movie_tags.items()}
    users = pd.read_parquet(pp.users)
    user_ids = users["userId"].to_numpy()

    als = AlsModel.load(mp.als, mp.als_meta)
    knn = ItemKnnModel.load(mp.knn, settings.high_rating_threshold, settings.knn_scoring, settings.knn_min_support)

    rows, user_lines = [], []
    for n, u in enumerate(chosen):
        h, r = user_row(history, int(u))
        als_scores = als.scores_for_vector(als.user_vector(h, r))
        knn_scores = knn.scores(h, r)
        pool = merge_candidates(
            als_scores, knn_scores, h,
            settings.stage1_per_model, settings.stage1_pool_size, settings.stage1_top_up,
        )
        evidence = knn.evidence(h, r, [c.movie_idx for c in pool], per_candidate=2)
        is_cold = bool(counts[u] <= settings.cold_start_max_ratings)
        for c in pool:
            m = c.movie_idx
            rows.append({
                "user_id": int(user_ids[u]),
                "rank": c.rank,
                "movie_id": int(movie_ids[m]),
                "title": titles[m],
                "year": None if pd.isna(years[m]) else int(years[m]),
                "genres": genres[m].split("|") if genres[m] else [],
                "tags": tags.get(int(movie_ids[m]), [])[:6],
                "merged_score": round(c.merged_score, 4),
                "als_score": None if c.als_score is None else round(c.als_score, 4),
                "knn_score": None if c.knn_score is None else round(c.knn_score, 4),
                "als_rank": c.als_rank,
                "knn_rank": c.knn_rank,
                "source": c.source,
                "knn_evidence": json.dumps([
                    {"historyMovie": titles[e.history_movie_idx], "similarity": round(e.similarity, 3)}
                    for e in evidence.get(m, [])
                ]),
                "is_cold_start": is_cold,
            })
        liked = h[r >= settings.high_rating_threshold]
        liked_r = r[r >= settings.high_rating_threshold]
        order = np.argsort(-liked_r, kind="stable")[:10]
        rec = recent[recent["user_idx"] == u].sort_values("timestamp", ascending=False)
        user_lines.append({
            "user_id": int(user_ids[u]),
            "history_size": int(counts[u]),
            "als_top10": [int(movie_ids[m]) for m in top_k(als_scores, 10, exclude=h)],
            "knn_top10": [int(movie_ids[m]) for m in top_k(knn_scores, 10, exclude=h)],
            "is_cold_start": is_cold,
            "favorite_genres": favorite_genres_from_history(h, r, genres, settings.high_rating_threshold),
            "highly_rated": [{"movie_id": int(movie_ids[liked[i]]), "title": titles[liked[i]], "rating": float(liked_r[i])} for i in order],
            "recent_5": [{"movie_id": int(movie_ids[m]), "title": titles[m], "rating": float(x)} for m, x in zip(rec["movie_idx"], rec["rating"])],
            "relevant_movie_ids": [int(movie_ids[m]) for m in relevant[u]],
        })
        if (n + 1) % 100 == 0:
            log(f"  {n + 1} / {len(chosen)} users")

    out = candidates_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    cand_path = out / f"{split_name}_candidates.parquet"
    users_path = out / f"{split_name}_users.jsonl"
    frame = pd.DataFrame(rows)
    for column in ("year", "als_rank", "knn_rank"):
        frame[column] = frame[column].astype("Int64")
    frame.to_parquet(cand_path, index=False)
    with users_path.open("w") as handle:
        for line in user_lines:
            handle.write(json.dumps(line) + "\n")
    summary = {
        "users": len(chosen),
        "coldStartUsers": int(len(chosen_cold)),
        "rows": len(rows),
        "candidates": str(cand_path),
        "userFile": str(users_path),
    }
    log(f"Wrote {len(rows):,} candidates for {len(chosen)} users ({len(chosen_cold)} cold-start) to {out}")
    return summary
