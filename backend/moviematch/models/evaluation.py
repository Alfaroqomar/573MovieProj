"""Offline evaluation on the chronological validation or test split.

Run with:  python -m moviematch.cli evaluate [--split test|validation] [--users N]

Protocol (all numbers are computed, nothing is estimated):
* Models are the saved artifacts trained on the TRAINING split only.
* Model input (user history) is the user's training interactions.
* For --split test, movies in the user's training AND validation
  interactions are excluded from the ranked list; for --split validation,
  training interactions are excluded.
* Relevant movies are the user's interactions in the evaluated split rated
  >= HIGH_RATING_THRESHOLD (default 4.0). Users with no relevant movies are
  skipped.
* A random sample of EVAL_SAMPLE_USERS eligible users (fixed seed) is used.
* Metrics @K (default 10), averaged over evaluated users:
    Precision@K = hits / K
    Recall@K    = hits / |relevant|
    NDCG@K      = DCG / IDCG with binary gains, log2(rank + 1) discount
    HitRate@K   = 1 if hits > 0 else 0
    MRR@K       = 1 / rank of the first hit (0 if none)
* ALS + LLM reranking is not evaluated offline (it would require one LLM
  call per evaluated user).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.sparse as sp

from moviematch.config import Settings
from moviematch.data.paths import evaluation_results_path, model_paths, processed_paths
from moviematch.data.preprocess import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION
from moviematch.models.als import AlsModel
from moviematch.models.common import top_k, user_row
from moviematch.models.item_knn import ItemKnnModel
from moviematch.models.popularity import PopularityModel

Log = Callable[[str], None]
METRIC_NAMES = ("precision", "recall", "ndcg", "hitRate", "mrr")


def ranking_metrics(ranked: np.ndarray, relevant: set[int], k: int) -> dict[str, float]:
    ranked = ranked[:k]
    hits = np.array([int(i) in relevant for i in ranked], dtype=bool)
    n_hits = int(hits.sum())
    discounts = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
    dcg = float(discounts[hits].sum()) if len(ranked) else 0.0
    ideal = float((1.0 / np.log2(np.arange(2, min(len(relevant), k) + 2))).sum())
    first = int(np.argmax(hits)) + 1 if n_hits else 0
    return {
        "precision": n_hits / k,
        "recall": n_hits / len(relevant) if relevant else 0.0,
        "ndcg": dcg / ideal if ideal > 0 else 0.0,
        "hitRate": 1.0 if n_hits else 0.0,
        "mrr": 1.0 / first if first else 0.0,
    }


def evaluate(
    settings: Settings,
    split_name: str = "test",
    k: int = 10,
    sample_users: int | None = None,
    log: Log = print,
) -> dict[str, object]:
    if split_name not in {"test", "validation"}:
        raise ValueError("split must be 'test' or 'validation'")
    target = SPLIT_TEST if split_name == "test" else SPLIT_VALIDATION
    started = time.perf_counter()
    pp, mp = processed_paths(settings), model_paths(settings)

    ratings = pd.read_parquet(pp.ratings, columns=["user_idx", "movie_idx", "rating", "split"])
    train = sp.load_npz(pp.train_matrix).tocsr()
    n_users, n_items = train.shape
    split = ratings["split"].to_numpy()
    users_arr = ratings["user_idx"].to_numpy()
    items_arr = ratings["movie_idx"].to_numpy()

    seen_mask = split == SPLIT_TRAIN if target == SPLIT_VALIDATION else split != SPLIT_TEST
    seen = sp.csr_matrix(
        (np.ones(int(seen_mask.sum()), dtype=np.int8), (users_arr[seen_mask], items_arr[seen_mask])),
        shape=(n_users, n_items),
    )
    relevant_mask = (split == target) & (ratings["rating"].to_numpy() >= settings.high_rating_threshold)
    relevant_frame = pd.DataFrame(
        {"user": users_arr[relevant_mask], "item": items_arr[relevant_mask]}
    )
    del ratings, split, users_arr, items_arr
    relevant_by_user = relevant_frame.groupby("user")["item"].apply(lambda s: set(s.tolist()))
    eligible = relevant_by_user.index.to_numpy()

    n_sample = sample_users or settings.eval_sample_users
    rng = np.random.default_rng(settings.eval_random_seed)
    chosen = np.sort(rng.choice(eligible, size=min(n_sample, len(eligible)), replace=False))
    log(f"Evaluating {len(chosen):,} of {len(eligible):,} eligible users on the {split_name} split (K={k})")

    models: dict[str, object] = {}
    if mp.popularity.exists():
        models["popularity"] = PopularityModel.load(mp.popularity)
    if mp.knn.exists():
        models["item_knn"] = ItemKnnModel.load(mp.knn, settings.high_rating_threshold)
    if mp.als.exists() and mp.als_meta.exists():
        models["als"] = AlsModel.load(mp.als, mp.als_meta)
    if not models:
        raise RuntimeError("No trained models found. Run the training command first.")

    totals = {name: {m: 0.0 for m in METRIC_NAMES} for name in models}
    timings = {name: 0.0 for name in models}
    for position, user in enumerate(chosen):
        exclude, _ = user_row(seen, int(user))
        history, history_ratings = user_row(train, int(user))
        relevant = relevant_by_user[user]
        for name, model in models.items():
            t0 = time.perf_counter()
            if name == "popularity":
                ranked = np.array([it.movie_idx for it in model.recommend(exclude, k)])
            elif name == "item_knn":
                scores = model.scores(history, history_ratings)
                scores[scores <= 0] = -np.inf
                ranked = top_k(scores, k, exclude=exclude)
            else:
                scores = model.item_factors @ model.user_factors[int(user)]
                ranked = top_k(scores, k, exclude=exclude)
            timings[name] += time.perf_counter() - t0
            for metric, value in ranking_metrics(ranked, relevant, k).items():
                totals[name][metric] += value
        if (position + 1) % 1000 == 0:
            log(f"  {position + 1:,} users evaluated")

    n_eval = max(1, len(chosen))
    results = {
        "available": True,
        "split": split_name,
        "k": k,
        "usersEvaluated": int(len(chosen)),
        "eligibleUsers": int(len(eligible)),
        "relevanceThreshold": settings.high_rating_threshold,
        "sampleSeed": settings.eval_random_seed,
        "protocol": (
            "Models trained on the chronological training split; history = training "
            "interactions; previously seen movies excluded; relevant = held-out "
            f"{split_name} ratings >= {settings.high_rating_threshold}."
        ),
        "models": {
            name: {
                **{f"{metric}@{k}": round(totals[name][metric] / n_eval, 5) for metric in METRIC_NAMES},
                "meanLatencyMs": round(1000 * timings[name] / n_eval, 3),
            }
            for name in models
        },
        "notEvaluated": {
            "als_llm": "Not evaluated offline: reranking requires one LLM call per user."
        },
        "createdAt": datetime.now(tz=timezone.utc).isoformat(),
        "durationSeconds": round(time.perf_counter() - started, 1),
    }
    out = evaluation_results_path(settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log(f"Saved evaluation results to {out}")
    for name, metrics in results["models"].items():
        log("  " + name + ": " + ", ".join(f"{key}={value}" for key, value in metrics.items()))
    return results
