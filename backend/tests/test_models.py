from __future__ import annotations

import json

import numpy as np
import scipy.sparse as sp

from moviematch.data.paths import model_paths, processed_paths
from moviematch.models.als import AlsModel
from moviematch.models.common import top_k, user_row
from moviematch.models.evaluation import evaluate, ranking_metrics
from moviematch.models.item_knn import ItemKnnModel
from moviematch.models.popularity import PopularityModel


def test_top_k_excludes_and_orders():
    scores = np.array([0.1, 0.9, 0.5, 0.9, -np.inf])
    assert top_k(scores, 3).tolist() == [1, 3, 2]
    assert top_k(scores, 3, exclude=np.array([1])).tolist() == [3, 2, 0]
    assert top_k(scores, 10).tolist() == [1, 3, 2, 0]


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
    assert set(results["models"]) == {"popularity", "item_knn", "als"}
    for metrics in results["models"].values():
        assert 0.0 <= metrics["precision@10"] <= 1.0
    saved = json.loads((prepared.evaluation_dir / "results.json").read_text())
    assert saved["models"] == results["models"]
