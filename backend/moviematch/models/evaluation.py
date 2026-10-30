"""Offline evaluation of every model on the same users with the same metrics.

Run with:  python -m moviematch.cli evaluate [--split test|validation] [--users N]

Protocol (every number is computed, nothing is estimated):

* Models are the saved artifacts (fitted on the training data, or on
  training + validation if trained with ``--with-validation``).
* History (what a model sees about a user) = everything before the evaluated
  period: training ratings for ``--split validation``; training + validation
  ratings for ``--split test``. History movies are never recommended.
* Relevant movies = the user's ratings in the evaluated period that are
  >= HIGH_RATING_THRESHOLD (default 4.0).
* Eligible users have at least one history rating and one relevant movie.
  A random sample of EVAL_SAMPLE_USERS eligible users (fixed seed) forms the
  main sample, used for "all users" and "regular users". Because cold-start
  users are rare, up to EVAL_COLD_START_EXTRA more eligible cold-start users
  are added for the cold-start group only.
* Cold-start user = 1 to COLD_START_MAX_RATINGS (default 10) history ratings.

Metrics (K = 10, averaged over users):
    Precision@K   hits / K
    Recall@K      hits / number of relevant movies
    NDCG@K        DCG / ideal DCG, binary gains, log2(rank + 1) discount
    HitRate@K     1 if any hit
    MRR@K         1 / rank of the first hit
    Recall@50     hits in the top 50 / relevant (for the Stage 1 models this is
                  the shortlist ceiling: the LLM can only pick from these 50)
    Serendipity@K relevant picks that are NOT in the most-rated top 50 for that
                  user / K (Ge et al., 2010: liked and not obvious)
    Novelty@K     mean of -log2(share of history users who rated the movie)
Catalog-level (over all users in the group, catalog = movies with at least
one history rating):
    Coverage@K    share of the catalog that appears in at least one top K
    Gini@K        inequality of how often each catalog movie is recommended
                  (0 = evenly spread, 1 = the same few movies for everyone)

Confidence intervals: 95% percentile bootstrap over users (EVAL_BOOTSTRAP
resamples), for NDCG@10 and Recall@10, plus paired differences in NDCG@10
against implicit ALS.

ALS + LLM reranking is evaluated separately through ``rerank_hook`` (see
``evaluate``), because it needs one LLM call per user.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.sparse as sp

from moviematch.config import Settings
from moviematch.data.paths import evaluation_results_path, model_paths, processed_paths
from moviematch.data.preprocess import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION, build_matrix
from moviematch.models.common import top_k, user_row

Log = Callable[[str], None]
K = 10
POOL = 50
ACCURACY = ("precision", "recall", "ndcg", "hitRate", "mrr")
MODEL_ORDER = (
    "popularity",
    "popularity_weighted",
    "item_knn",
    "item_knn_sum",
    "als",
    "als_explicit",
    "stage1",
    "neumf",
)
MODEL_LABELS = {
    "popularity": "Popularity (most-rated)",
    "popularity_weighted": "Popularity (weighted)",
    "item_knn": "Item-KNN (average, proposal)",
    "item_knn_sum": "Item-KNN (sum)",
    "als": "ALS (implicit)",
    "als_explicit": "ALS (explicit, proposal)",
    "stage1": "Stage 1 (ALS + KNN merged)",
    "neumf": "NeuMF",
}
BUCKETS = ((1, 10, "1-10"), (11, 50, "11-50"), (51, 200, "51-200"), (201, 10**9, "201+"))

# A reranker for the LLM teammate: (user_idx, stage1_pool_movie_idx) -> ranked movie_idx
RerankHook = Callable[[int, np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- #
# Metric helpers (pure functions, unit-tested)
# --------------------------------------------------------------------------- #

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


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of non-negative counts (0 = equal, ->1 = concentrated)."""
    values = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(values)
    total = values.sum()
    if n == 0 or total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * values).sum()) / (n * total) - (n + 1) / n)


def coverage(counts: np.ndarray) -> float:
    counts = np.asarray(counts)
    return float((counts > 0).sum() / len(counts)) if len(counts) else 0.0


def serendipity(ranked: np.ndarray, relevant: set[int], obvious: set[int], k: int) -> float:
    return sum(1 for i in ranked[:k] if int(i) in relevant and int(i) not in obvious) / k


def bootstrap_ci(values: np.ndarray, index: np.ndarray) -> list[float]:
    means = values[index].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 5), round(float(np.percentile(means, 97.5)), 5)]


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

@dataclass
class Rankers:
    names: list[str]
    rank: Callable[[str, int, np.ndarray, np.ndarray], np.ndarray]
    neumf_fallbacks: dict[str, int]
    skipped: dict[str, str]


def load_rankers(settings: Settings, log: Log) -> Rankers:
    from moviematch.models.als import AlsModel
    from moviematch.models.als_explicit import ExplicitAlsModel
    from moviematch.models.item_knn import ItemKnnModel
    from moviematch.models.popularity import PopularityModel
    from moviematch.models.stage1 import merge_candidates

    mp = model_paths(settings)
    loaded: dict[str, object] = {}
    skipped: dict[str, str] = {}
    if mp.popularity.exists():
        loaded["popularity"] = PopularityModel.load(mp.popularity, "count")
        loaded["popularity_weighted"] = PopularityModel.load(mp.popularity, "weighted")
    else:
        skipped["popularity"] = skipped["popularity_weighted"] = "not trained"
    if mp.knn.exists():
        base = ItemKnnModel.load(mp.knn, settings.high_rating_threshold, settings.knn_scoring, settings.knn_min_support)
        loaded["item_knn"] = base.with_scoring("average")
        loaded["item_knn_sum"] = base.with_scoring("sum")
    else:
        skipped["item_knn"] = skipped["item_knn_sum"] = "not trained"
    if mp.als.exists() and mp.als_meta.exists():
        loaded["als"] = AlsModel.load(mp.als, mp.als_meta)
    else:
        skipped["als"] = "not trained"
    if mp.als_explicit.exists() and mp.als_explicit_meta.exists():
        loaded["als_explicit"] = ExplicitAlsModel.load(mp.als_explicit, mp.als_explicit_meta)
    else:
        skipped["als_explicit"] = "not trained"
    if "als" in loaded and "item_knn" in loaded:
        loaded["stage1"] = True
    else:
        skipped["stage1"] = "needs ALS and Item-KNN"
    if mp.neumf.exists() and mp.neumf_meta.exists():
        try:
            from moviematch.models.neumf import NeuMFModel

            loaded["neumf"] = NeuMFModel.load(mp.neumf, mp.neumf_meta)
        except ImportError:
            skipped["neumf"] = "PyTorch not installed"
    else:
        skipped["neumf"] = "not trained (train --only neumf)"

    stage1_knn = loaded.get("item_knn") if settings.knn_scoring == "average" else loaded.get("item_knn_sum")
    fallbacks = {"neumf": 0}

    def rank(name: str, user: int, history: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        model = loaded[name]
        if name in ("popularity", "popularity_weighted"):
            return model.ranking(history, POOL)
        if name in ("item_knn", "item_knn_sum"):
            return top_k(model.scores(history, ratings), POOL, exclude=history)
        if name == "als":
            return top_k(model.scores_for_vector(model.user_vector(history, ratings)), POOL, exclude=history)
        if name == "als_explicit":
            return np.array([c.movie_idx for c in model.candidates(history, ratings, POOL)], dtype=np.int64)
        if name == "stage1":
            als = loaded["als"]
            pool = merge_candidates(
                als.scores_for_vector(als.user_vector(history, ratings)),
                stage1_knn.scores(history, ratings),
                history,
                settings.stage1_per_model,
                settings.stage1_pool_size,
                settings.stage1_top_up,
            )
            return np.array([c.movie_idx for c in pool], dtype=np.int64)
        if name == "neumf":
            if not model.knows(user):
                fallbacks["neumf"] += 1
                return loaded["popularity"].ranking(history, POOL) if "popularity" in loaded else np.zeros(0, np.int64)
            return top_k(model.scores(user), POOL, exclude=history)
        raise KeyError(name)

    names = [n for n in MODEL_ORDER if n in loaded]
    log("  models: " + ", ".join(names) + (f" (skipped: {', '.join(f'{k} - {v}' for k, v in skipped.items())})" if skipped else ""))
    return Rankers(names, rank, fallbacks, skipped)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(
    settings: Settings,
    split_name: str = "test",
    k: int = K,
    sample_users: int | None = None,
    log: Log = print,
    rerank_hook: RerankHook | None = None,
) -> dict[str, object]:
    if split_name not in {"test", "validation"}:
        raise ValueError("split must be 'test' or 'validation'")
    started = time.perf_counter()
    pp, mp = processed_paths(settings), model_paths(settings)
    target = SPLIT_TEST if split_name == "test" else SPLIT_VALIDATION
    history_splits = [SPLIT_TRAIN] if target == SPLIT_VALIDATION else [SPLIT_TRAIN, SPLIT_VALIDATION]

    ratings = pd.read_parquet(pp.ratings, columns=["user_idx", "movie_idx", "rating", "split"])
    n_users, n_items = sp.load_npz(pp.train_matrix).shape
    split = ratings["split"].to_numpy()
    u_arr = ratings["user_idx"].to_numpy()
    m_arr = ratings["movie_idx"].to_numpy()
    r_arr = ratings["rating"].to_numpy()
    del ratings
    hist_mask = np.isin(split, history_splits)
    history = build_matrix(u_arr[hist_mask], m_arr[hist_mask], r_arr[hist_mask], n_users, n_items)
    rel_mask = (split == target) & (r_arr >= settings.high_rating_threshold)
    relevant_frame = pd.DataFrame({"user": u_arr[rel_mask], "item": m_arr[rel_mask]})
    del split, u_arr, m_arr, r_arr, hist_mask, rel_mask
    relevant_by_user = relevant_frame.groupby("user")["item"].apply(lambda s: set(s.tolist()))

    history_counts = np.diff(history.indptr)
    item_history_counts = np.diff(history.tocsc().indptr)
    catalog = item_history_counts > 0
    users_with_history = int((history_counts > 0).sum())
    self_info = -np.log2(np.maximum(item_history_counts, 1) / max(users_with_history, 1))

    candidates = relevant_by_user.index.to_numpy()
    eligible = candidates[history_counts[candidates] > 0]
    cold_max = settings.cold_start_max_ratings
    is_cold = (history_counts >= 1) & (history_counts <= cold_max)
    eligible_cold = eligible[is_cold[eligible]]

    rng = np.random.default_rng(settings.eval_random_seed)
    n_main = min(sample_users or settings.eval_sample_users, len(eligible))
    main = np.sort(rng.choice(eligible, size=n_main, replace=False)) if n_main else np.zeros(0, np.int64)
    extra_pool = np.setdiff1d(eligible_cold, main)
    n_extra = min(settings.eval_cold_start_extra, len(extra_pool))
    extra = np.sort(rng.choice(extra_pool, size=n_extra, replace=False)) if n_extra else np.zeros(0, np.int64)
    users = np.concatenate([main, extra])
    in_main = np.concatenate([np.ones(len(main), bool), np.zeros(len(extra), bool)])
    log(
        f"Evaluating on the {split_name} split: {len(main):,} sampled users "
        f"(of {len(eligible):,} eligible) + {len(extra):,} extra cold-start users "
        f"({len(eligible_cold):,} eligible cold-start users in total)"
    )

    rankers = load_rankers(settings, log)
    if not rankers.names:
        raise RuntimeError("No trained models found. Run the training command first.")
    names = list(rankers.names)
    popularity_for_obvious = "popularity" in names
    if rerank_hook is not None and "stage1" in names:
        names.append("stage1_rerank")

    per_user = {name: {m: np.zeros(len(users)) for m in (*ACCURACY, "recall50", "serendipity", "novelty")} for name in names}
    rec_counts = {name: {"main": np.zeros(n_items, np.int64), "cold": np.zeros(n_items, np.int64), "regular": np.zeros(n_items, np.int64)} for name in names}
    timings = {name: 0.0 for name in names}
    unreachable = np.zeros(len(users))

    for pos, user in enumerate(users):
        user = int(user)
        hist, hist_r = user_row(history, user)
        relevant = relevant_by_user[user]
        unreachable[pos] = sum(1 for i in relevant if not catalog[i]) / len(relevant)
        obvious: set[int] = set()
        stage1_pool = None
        lists: dict[str, np.ndarray] = {}
        for name in rankers.names:
            t0 = time.perf_counter()
            lists[name] = rankers.rank(name, user, hist, hist_r)
            timings[name] += time.perf_counter() - t0
        if popularity_for_obvious:
            obvious = set(int(i) for i in lists["popularity"][:POOL])
        if rerank_hook is not None and "stage1" in lists:
            stage1_pool = lists["stage1"]
            t0 = time.perf_counter()
            lists["stage1_rerank"] = np.asarray(rerank_hook(user, stage1_pool), dtype=np.int64)
            timings["stage1_rerank"] += time.perf_counter() - t0
        cold = bool(is_cold[user])
        for name in names:
            ranked = lists[name]
            metrics = ranking_metrics(ranked, relevant, k)
            store = per_user[name]
            for m in ACCURACY:
                store[m][pos] = metrics[m]
            store["recall50"][pos] = sum(1 for i in ranked[:POOL] if int(i) in relevant) / len(relevant)
            store["serendipity"][pos] = serendipity(ranked, relevant, obvious, k) if obvious else 0.0
            top = ranked[:k]
            store["novelty"][pos] = float(self_info[top].mean()) if len(top) else 0.0
            if in_main[pos]:
                np.add.at(rec_counts[name]["main"], top, 1)
                np.add.at(rec_counts[name]["cold" if cold else "regular"], top, 1)
            elif cold:
                np.add.at(rec_counts[name]["cold"], top, 1)
        if (pos + 1) % 1000 == 0:
            log(f"  {pos + 1:,} / {len(users):,} users")

    cold_mask = is_cold[users]
    groups = {
        "all": in_main,
        "cold_start": cold_mask,
        "regular": in_main & ~cold_mask,
    }
    count_key = {"all": "main", "cold_start": "cold", "regular": "regular"}
    catalog_idx = np.flatnonzero(catalog)

    def summarise(name: str, mask: np.ndarray, group: str) -> dict[str, float]:
        store = per_user[name]
        if not mask.any():
            return {}
        out = {f"{m}@{k}": round(float(store[m][mask].mean()), 5) for m in ACCURACY}
        out["recall@50"] = round(float(store["recall50"][mask].mean()), 5)
        out[f"serendipity@{k}"] = round(float(store["serendipity"][mask].mean()), 5)
        out[f"novelty@{k}"] = round(float(store["novelty"][mask].mean()), 3)
        counts = rec_counts[name][count_key[group]][catalog_idx]
        out[f"coverage@{k}"] = round(coverage(counts), 5)
        out[f"gini@{k}"] = round(gini(counts), 5)
        return out

    results_groups = {
        group: {name: summarise(name, mask, group) for name in names} for group, mask in groups.items()
    }
    group_sizes = {group: int(mask.sum()) for group, mask in groups.items()}

    # Bootstrap confidence intervals.
    boot_rng = np.random.default_rng(settings.eval_random_seed + 1)
    confidence: dict[str, dict[str, dict[str, list[float]]]] = {}
    paired: dict[str, dict[str, float]] = {}
    for group, mask in groups.items():
        idx_users = np.flatnonzero(mask)
        if len(idx_users) < 2:
            continue
        index = idx_users[boot_rng.integers(0, len(idx_users), size=(settings.eval_bootstrap, len(idx_users)))]
        confidence[group] = {
            name: {
                f"ndcg@{k}": bootstrap_ci(per_user[name]["ndcg"], index),
                f"recall@{k}": bootstrap_ci(per_user[name]["recall"], index),
            }
            for name in names
        }
        if group == "all" and "als" in names:
            base = per_user["als"]["ndcg"]
            for name in names:
                if name == "als":
                    continue
                diff = per_user[name]["ndcg"] - base
                lo, hi = bootstrap_ci(diff, index)
                paired[name] = {"meanDifference": round(float(diff[mask].mean()), 5), "low": lo, "high": hi}

    buckets = {}
    hcount = history_counts[users]
    for lo, hi, label in BUCKETS:
        mask = (hcount >= lo) & (hcount <= hi) & (in_main | cold_mask)
        if not mask.any():
            continue
        buckets[label] = {
            "users": int(mask.sum()),
            "models": {
                name: {
                    f"ndcg@{k}": round(float(per_user[name]["ndcg"][mask].mean()), 5),
                    f"recall@{k}": round(float(per_user[name]["recall"][mask].mean()), 5),
                }
                for name in names
            },
        }

    n_eval = max(1, len(users))
    training_manifest = json.loads(mp.training_manifest.read_text()) if mp.training_manifest.exists() else {}
    results = {
        "available": True,
        "split": split_name,
        "k": k,
        "usersEvaluated": int(len(main)),
        "coldStartExtraUsers": int(len(extra)),
        "eligibleUsers": int(len(eligible)),
        "eligibleColdStartUsers": int(len(eligible_cold)),
        "groupSizes": group_sizes,
        "coldStartDefinition": f"1 to {cold_max} ratings before the evaluated period",
        "relevanceThreshold": settings.high_rating_threshold,
        "sampleSeed": settings.eval_random_seed,
        "protocol": (
            f"History = ratings before the {split_name} period ("
            + ("train" if target == SPLIT_VALIDATION else "train + validation")
            + "); history movies excluded from recommendations; relevant = "
            f"{split_name} ratings >= {settings.high_rating_threshold}. Same users for every model."
        ),
        "modelLabels": {n: MODEL_LABELS.get(n, n) for n in names},
        "models": results_groups["all"],
        "groups": results_groups,
        "confidence": confidence,
        "pairedNdcgVsAls": paired,
        "historyBuckets": buckets,
        "relevantUnreachable": {
            group: round(float(unreachable[mask].mean()), 5) if mask.any() else None
            for group, mask in groups.items()
        },
        "meanLatencyMs": {name: round(1000 * timings[name] / n_eval, 3) for name in names},
        "neumfFallbackUsers": rankers.neumf_fallbacks.get("neumf", 0) if "neumf" in names else None,
        "skippedModels": rankers.skipped,
        "trainedOn": {name: info.get("fitOn", "train") for name, info in training_manifest.get("models", {}).items()},
        "notEvaluated": {}
        if rerank_hook is not None
        else {"als_llm": "Not evaluated here: pass a rerank hook (LLM stage) to evaluate()."},
        "createdAt": datetime.now(tz=timezone.utc).isoformat(),
        "durationSeconds": round(time.perf_counter() - started, 1),
    }

    out = evaluation_results_path(settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    write_csv(results, out.parent)
    log(f"Saved evaluation results to {out}")
    for name in names:
        m = results_groups["all"][name]
        log(
            f"  {MODEL_LABELS.get(name, name):30s} NDCG@10 {m['ndcg@10']:.4f}  Recall@10 {m['recall@10']:.4f}  "
            f"Recall@50 {m['recall@50']:.4f}  Coverage {m['coverage@10']:.4f}  Gini {m['gini@10']:.3f}  "
            f"Serendipity {m['serendipity@10']:.4f}"
        )
    return results


def write_csv(results: dict, folder) -> None:
    """One CSV per user group, for report tables."""
    for group, models in results["groups"].items():
        rows = [(name, metrics) for name, metrics in models.items() if metrics]
        if not rows:
            continue
        columns = list(rows[0][1].keys())
        with (folder / f"results_{results['split']}_{group}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", *columns, "ndcg@10 95% CI", "recall@10 95% CI"])
            for name, metrics in rows:
                ci = results.get("confidence", {}).get(group, {}).get(name, {})
                writer.writerow(
                    [results["modelLabels"].get(name, name), *[metrics[c] for c in columns],
                     ci.get("ndcg@10", ""), ci.get("recall@10", "")]
                )
