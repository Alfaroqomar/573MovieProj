from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from moviematch.data.paths import processed_paths
from moviematch.data.preprocess import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    chronological_split,
    date_split,
    date_to_timestamp,
    split_title,
)


def test_split_title():
    assert split_title("Matrix, The (1999)") == ("The Matrix", 1999)
    assert split_title("Toy Story (1995)") == ("Toy Story", 1995)
    assert split_title("Untitled") == ("Untitled", None)
    assert split_title("Cosmos (1980– )") == ("Cosmos", 1980)


def test_chronological_split_fractions():
    user_idx = np.repeat(np.arange(3), [10, 20, 1])
    split = chronological_split(user_idx, 0.8, 0.1)
    assert split[:10].tolist() == [0] * 8 + [1] + [2]
    assert (split[10:30] == 0).sum() == 16 and (split[10:30] == 1).sum() == 2 and (split[10:30] == 2).sum() == 2
    assert split[30] == SPLIT_TRAIN  # a single interaction always goes to training


def test_preprocess_cleans_and_splits(prepared):
    pp = processed_paths(prepared)
    manifest = json.loads(pp.manifest.read_text())
    cleaning = manifest["cleaning"]["ratings"]
    assert cleaning["rows_dropped_invalid"] >= 3  # bad rating, unknown movie, missing rating
    assert cleaning["rows_dropped_duplicate"] >= 1
    assert manifest["config"]["splitMode"] == "date"
    assert manifest["config"]["validationStart"] == "2020-01-01"

    ratings = pd.read_parquet(pp.ratings)
    assert ratings["rating"].between(0.5, 5.0).all()
    assert not ratings.duplicated(["user_idx", "movie_idx"]).any()

    # Date split: one calendar cutoff for everyone.
    v, t = date_to_timestamp("2020-01-01"), date_to_timestamp("2022-01-01")
    assert (ratings.loc[ratings["split"] == SPLIT_TRAIN, "timestamp"] < v).all()
    val = ratings.loc[ratings["split"] == SPLIT_VALIDATION, "timestamp"]
    assert ((val >= v) & (val < t)).all()
    assert (ratings.loc[ratings["split"] == SPLIT_TEST, "timestamp"] >= t).all()

    # Cold-start flag: 1-10 training ratings.
    users = pd.read_parquet(pp.users)
    cold = users[users["is_cold_start"]]
    assert len(cold) > 0
    assert cold["train_count"].between(1, 10).all()
    assert not users.loc[~users["is_cold_start"], "train_count"].between(1, 10).any()
    stats = json.loads(pp.statistics.read_text())
    assert stats["splitDetails"]["coldStartUsers"] == len(cold)

    # Five most recent movies per user, newest first.
    recent = pd.read_parquet(pp.user_recent_movies)
    assert recent.groupby("user_idx").size().max() == 5
    first_user = recent[recent["user_idx"] == 0].sort_values("rank")
    assert first_user["timestamp"].is_monotonic_decreasing

    # Chronology: within each user, every train timestamp <= validation <= test.
    grouped = ratings.groupby(["user_idx", "split"])["timestamp"].agg(["min", "max"]).unstack()
    train_max = grouped[("max", SPLIT_TRAIN)]
    val_min, val_max = grouped[("min", SPLIT_VALIDATION)], grouped[("max", SPLIT_VALIDATION)]
    test_min = grouped[("min", SPLIT_TEST)]
    assert ((train_max <= val_min) | val_min.isna()).all()
    assert ((val_max <= test_min) | val_max.isna() | test_min.isna()).all()

    # The training matrix contains only training interactions.
    train = sp.load_npz(pp.train_matrix)
    assert train.nnz == int((ratings["split"] == SPLIT_TRAIN).sum())

    movies = pd.read_parquet(pp.movies)
    assert "The Synthetic Picture 1" in set(movies["title"])
    assert movies.loc[movies["movieId"] == 8, "genres"].item() == ""  # "(no genres listed)"

    tags = pd.read_parquet(pp.movie_tags)
    first = tags[tags["movie_idx"] == 0].set_index("tag")["count"]
    assert first["mind-bending"] == 2  # normalized and counted per distinct user
    assert "slow, quiet" in first.index


def test_date_split_boundaries():
    v, t = date_to_timestamp("2020-01-01"), date_to_timestamp("2022-01-01")
    stamps = np.array([v - 1, v, t - 1, t, t + 5])
    assert date_split(stamps, "2020-01-01", "2022-01-01").tolist() == [
        SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_VALIDATION, SPLIT_TEST, SPLIT_TEST
    ]


def test_per_user_mode_still_works(tmp_path):
    import dataclasses

    from moviematch.data.preprocess import preprocess
    from tests.conftest import make_settings, write_synthetic_movielens

    data = write_synthetic_movielens(tmp_path / "ml-32m", n_users=40, n_movies=60)
    settings = dataclasses.replace(make_settings(tmp_path, data), split_mode="per_user")
    manifest = preprocess(settings, log=lambda _: None)
    assert manifest["config"]["splitMode"] == "per_user"
    assert manifest["config"]["trainFraction"] == 0.8
