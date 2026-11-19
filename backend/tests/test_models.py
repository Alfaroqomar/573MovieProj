from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp

from moviematch.data.paths import model_paths, processed_paths
from moviematch.models.als import AlsModel
from moviematch.models.candidates import favorite_genres_from_history, historical_tags
from moviematch.models.common import top_k, user_row
from moviematch.models.evaluation import coverage, evaluate, gini, ranking_metrics, serendipity
from moviematch.models.item_knn import ItemKnnModel
from moviematch.models.popularity import PopularityModel
from moviematch.models.stage1 import merge_candidates


def test_top_k_excludes_and_orders():
    scores = np.array([0.1, 0.9, 0.5, 0.9, -np.inf])
    assert top_k(scores, 3).tolist() == [1, 3, 2]
    assert top_k(scores, 3, exclude=np.array([1])).tolist() == [3, 2, 0]
    assert top_k(scores, 10).tolist() == [1, 3, 2, 0]


def test_candidate_profile_uses_only_history():
    genres = np.array(["Drama", "Sci-Fi", "Sci-Fi", "Horror"])
    history = np.array([0, 1, 2])
    ratings = np.array([5.0, 3.0, 2.0])
    assert favorite_genres_from_history(history, ratings, genres, 4.0) == ["Drama"]
    assert favorite_genres_from_history(history, ratings, genres, 6.0) == ["Sci-Fi", "Drama"]


def test_historical_tags_exclude_later_tags(tmp_path):
    path = tmp_path / "tags.csv"
    path.write_text(
        "userId,movieId,tag,timestamp\n"
        "1,10,Thoughtful,1577836799\n"
        "2,10,Future Plot,1577836800\n"
    )
    assert historical_tags(path, "2020-01-01", 6) == {10: ["thoughtful"]}


def _user(prepared, user_idx=5):
    full = sp.load_npz(processed_paths(prepared).full_matrix).tocsr()
    return user_row(full, user_idx)


def test_popularity_excludes_rated(prepared):
    model = PopularityModel.load(model_paths(prepared).popularity)
    history, _ = _user(prepared)
    recs = model.recommend(history, 10)
    assert recs and not set(r.movie_idx for r in recs) & set(history.tolist())
    assert recs[0].rating_count >= model.min_votes
    # Users who rated every eligible movie still receive the remaining rated movies.
    everything_eligible = model.order[: model.eligible_count]
    assert model.recommend(everything_eligible, 3)
    eligible = [r.weighted_rating for r in recs if r.rating_count >= model.min_votes]
    assert eligible == sorted(eligible, reverse=True)


def test_item_knn_excludes_rated_and_explains(prepared):
    model = ItemKnnModel.load(model_paths(prepared).knn, prepared.high_rating_threshold)
    assert (model.similarity.diagonal() == 0).all()
    history, ratings = _user(prepared)
    recs = model.recommend(history, ratings, 10)
    ids = [i for i, _ in recs]
    assert ids and not set(ids) & set(history.tolist())
    evidence = model.evidence(history, ratings, ids)
    liked = set(history[ratings >= prepared.high_rating_threshold].tolist())
    for items in evidence.values():
        for e in items:
            assert e.history_movie_idx in liked and e.similarity > 0


def test_als_candidates(prepared):
    mp = model_paths(prepared)
    model = AlsModel.load(mp.als, mp.als_meta)
    history, ratings = _user(prepared)
    pool = model.candidates(history, ratings, 50)
    assert len(pool) == 50
    assert not {c.movie_idx for c in pool} & set(history.tolist())
    assert [c.rank for c in pool] == list(range(1, 51))
    assert pool[0].normalized_score == 1.0 and pool[-1].normalized_score == 0.0
    assert all(0.0 <= c.normalized_score <= 1.0 for c in pool)
    raw = [c.raw_score for c in pool]
    assert raw == sorted(raw, reverse=True)


def test_ranking_metrics():
    metrics = ranking_metrics(np.array([5, 1, 9]), {1, 2}, k=3)
    assert metrics["precision"] == 1 / 3
    assert metrics["recall"] == 0.5
    assert metrics["hitRate"] == 1.0
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg"] < 1


def test_evaluation_writes_real_results(prepared):
    results = evaluate(prepared, log=lambda _: None)
    assert results["available"] and results["usersEvaluated"] > 0
    assert {"popularity", "popularity_weighted", "item_knn", "item_knn_sum", "als", "als_explicit", "stage1"} <= set(results["models"])
    for metrics in results["models"].values():
        assert 0.0 <= metrics["precision@10"] <= 1.0
        assert 0.0 <= metrics["coverage@10"] <= 1.0
        assert 0.0 <= metrics["gini@10"] <= 1.0
        assert metrics["recall@50"] >= metrics["recall@10"]
        assert 0.0 <= metrics["serendipity@10"] <= metrics["precision@10"] + 1e-9
    assert set(results["groups"]) == {"all", "cold_start", "regular"}
    validation = evaluate(prepared, split_name="validation", log=lambda _: None)
    assert validation["groupSizes"]["cold_start"] > 0
    assert validation["groups"]["cold_start"]["als"]["ndcg@10"] >= 0
    lo, hi = results["confidence"]["all"]["als"]["ndcg@10"]
    assert lo <= results["models"]["als"]["ndcg@10"] <= hi
    assert (prepared.evaluation_dir / "results_test_all.csv").exists()
    saved = json.loads((prepared.evaluation_dir / "results.json").read_text())
    assert saved["split"] == "validation"  # the last run wins


# --------------------------------------------------------------------------- #
# Proposal-aligned models and metrics
# --------------------------------------------------------------------------- #



def test_gini_coverage_serendipity():
    assert gini(np.array([5, 5, 5, 5])) == 0.0
    assert gini(np.array([0, 0, 0, 20])) == 0.75  # (n-1)/n for one movie taking everything
    assert gini(np.zeros(4)) == 0.0
    assert coverage(np.array([0, 1, 0, 3])) == 0.5
    ranked = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert serendipity(ranked, relevant={1, 2, 3}, obvious={1}, k=10) == 0.2


def test_popularity_count_rule(prepared):
    model = PopularityModel.load(model_paths(prepared).popularity, "count")
    counts = model.counts[model.order]
    assert (np.diff(counts) <= 0).all()  # most-rated first
    history, _ = _user(prepared)
    recs = model.ranking(history, 10)
    assert len(recs) == 10 and not set(recs.tolist()) & set(history.tolist())


def test_knn_average_matches_formula():
    # Movie 2's neighbours: movie 0 (sim 0.5) and movie 1 (sim 1.0).
    sim = sp.csr_matrix(np.array([
        [0.0, 0.0, 0.5, 0.2],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 1.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.0],
    ], dtype=np.float32))
    model = ItemKnnModel(sim, 4.0, scoring="average", min_support=2)
    scores = model.scores(np.array([0, 1]), np.array([4.0, 1.0], dtype=np.float32))
    assert abs(scores[2] - (0.5 * 4 + 1.0 * 1) / 1.5) < 1e-2  # weighted average = 2.0 (+ tiny tie-break)
    assert scores[3] == -np.inf  # only one neighbour in history: below min support


def test_explicit_als(prepared):
    from moviematch.models.als_explicit import ExplicitAlsModel

    mp = model_paths(prepared)
    meta = json.loads(mp.als_explicit_meta.read_text())
    assert meta["trainRmse"][-1] < meta["trainRmse"][0]
    model = ExplicitAlsModel.load(mp.als_explicit, mp.als_explicit_meta)
    history, ratings = _user(prepared)
    pool = model.candidates(history, ratings, 20)
    assert len(pool) == 20 and not {c.movie_idx for c in pool} & set(history.tolist())


def test_stage1_merge_rules():
    als = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.0])
    knn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4.0])
    knn[knn == 0] = -np.inf
    pool = merge_candidates(als, knn, history=np.array([0]), per_model=3, pool_size=6, top_up=True)
    ids = [c.movie_idx for c in pool]
    assert 0 not in ids and len(pool) == 6
    sources = {c.movie_idx: c.source for c in pool}
    assert sources[1] == "als" and sources[9] == "knn" and sources[10] == "knn"
    assert [c.rank for c in pool] == list(range(1, 7))
    assert sum(1 for c in pool if c.source == "als_top_up") == 6 - 5
    merged = [c.merged_score for c in pool]
    assert merged == sorted(merged, reverse=True)
    both = merge_candidates(np.array([3.0, 2, 1]), np.array([1.0, 3, 2]), np.array([], int), 2, 3, False)
    assert {c.source for c in both} >= {"both"}


def test_neumf_trained(prepared):
    pytest.importorskip("torch")
    from moviematch.models.neumf import NeuMFModel

    mp = model_paths(prepared)
    model = NeuMFModel.load(mp.neumf, mp.neumf_meta)
    history, _ = _user(prepared)
    scores = model.scores(5)
    picks = top_k(scores, 10, exclude=history)
    assert len(picks) == 10 and not set(picks.tolist()) & set(history.tolist())
