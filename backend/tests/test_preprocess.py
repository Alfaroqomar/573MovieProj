from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from moviematch.data.paths import processed_paths
from moviematch.data.preprocess import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION, chronological_split, split_title


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
    assert manifest["config"]["trainFraction"] == 0.8

    ratings = pd.read_parquet(pp.ratings)
    assert ratings["rating"].between(0.5, 5.0).all()
    assert not ratings.duplicated(["user_idx", "movie_idx"]).any()

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
