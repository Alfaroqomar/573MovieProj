"""Preprocess the raw MovieLens 32M CSV files into compact local artifacts.

Run with:  python -m moviematch.cli preprocess

Steps (matching the project specification):
 1. Verify that the required MovieLens files exist.
 2. Validate the expected columns.
 3. Load movies, ratings, tags, and links.
 4. Remove invalid rows (missing values, out-of-range ratings, unknown movies).
 5. Deduplicate ratings, movies, tags, and links.
 6. Map user and movie IDs to contiguous internal indices.
 7. Sort interactions by user and timestamp.
 8. Create a chronological per-user train / validation / test split.
 9. Save processed tables (Parquet) and sparse matrices (NPZ).
10. Record the dataset version and preprocessing configuration.
11. Aggregate MovieLens tags per movie.
12. Pre-compute the user summaries shown in the UI.
"""

from __future__ import annotations

import json
import platform
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import scipy
import scipy.sparse as sp

from moviematch.config import Settings
from moviematch.data.paths import ARTIFACT_FORMAT_VERSION, processed_paths
from moviematch.data.raw import (
    DATASET_ID,
    DATASET_NAME,
    EXPECTED_COUNTS,
    REQUIRED_COLUMNS,
    check_raw_dataset,
    csv_read_options,
)

Log = Callable[[str], None]

NO_GENRES = "(no genres listed)"
SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST = 0, 1, 2

_YEAR_RE = re.compile(r"\s*\((\d{4})(?:\s*[–-]\s*\d{0,4}\s*)?\)\s*$")
_ARTICLE_RE = re.compile(r"^(?P<body>.+), (?P<article>The|A|An)$")


class DatasetError(RuntimeError):
    """Raised when the raw dataset is missing or malformed."""


@dataclass
class CleaningReport:
    rows_read: int
    unparseable_rows: int
    rows_dropped_invalid: int
    rows_dropped_duplicate: int
    rows_kept: int

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# Title helpers
# --------------------------------------------------------------------------- #

def split_title(raw_title: str) -> tuple[str, int | None]:
    """Split "Matrix, The (1999)" into ("The Matrix", 1999).

    Only a trailing ", The" / ", A" / ", An" is moved to the front; titles
    with alternate names in parentheses are otherwise left untouched.
    """
    title = raw_title.strip()
    year: int | None = None
    match = _YEAR_RE.search(title)
    if match:
        year = int(match.group(1))
        title = title[: match.start()].rstrip()
    article = _ARTICLE_RE.match(title)
    if article:
        title = f"{article.group('article')} {article.group('body')}"
    return title, year


# --------------------------------------------------------------------------- #
# CSV readers
# --------------------------------------------------------------------------- #

def _read_csv(path: Path, column_types: dict[str, pa.DataType]) -> tuple[pd.DataFrame, int]:
    """Read a CSV with pyarrow, skipping structurally broken rows.

    Falls back to a tolerant pandas reader if a value cannot be converted to
    the requested type; unconvertible values become NaN and are dropped by
    the caller's validation step.
    """
    skipped = 0

    def on_invalid(_row: pacsv.InvalidRow) -> str:
        nonlocal skipped
        skipped += 1
        return "skip"

    try:
        table = pacsv.read_csv(
            path,
            read_options=csv_read_options(),
            parse_options=pacsv.ParseOptions(invalid_row_handler=on_invalid),
            convert_options=pacsv.ConvertOptions(
                column_types=column_types,
                include_columns=list(column_types),
                strings_can_be_null=True,
            ),
        )
        return table.to_pandas(), skipped
    except pa.ArrowInvalid:
        frame = pd.read_csv(
            path, dtype=str, usecols=list(column_types), on_bad_lines="skip", keep_default_na=False
        )
        for column, dtype in column_types.items():
            if pa.types.is_integer(dtype) or pa.types.is_floating(dtype):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            else:
                frame[column] = frame[column].replace("", np.nan)
        return frame, skipped


def load_movies(path: Path, log: Log) -> tuple[pd.DataFrame, CleaningReport]:
    frame, skipped = _read_csv(
        path, {"movieId": pa.int64(), "title": pa.string(), "genres": pa.string()}
    )
    rows_read = len(frame) + skipped
    valid = frame["movieId"].notna() & (frame["movieId"] > 0) & frame["title"].notna()
    invalid = int((~valid).sum())
    frame = frame.loc[valid].copy()
    before = len(frame)
    frame = frame.drop_duplicates("movieId", keep="first")
    duplicates = before - len(frame)

    frame["movieId"] = frame["movieId"].astype(np.int64)
    titles, years = zip(*(split_title(str(t)) for t in frame["title"])) if len(frame) else ((), ())
    frame["display_title"] = list(titles)
    frame["year"] = pd.array(list(years), dtype="Int32")
    frame["genres"] = [
        "|".join(g for g in str(value).split("|") if g and g != NO_GENRES)
        if isinstance(value, str)
        else ""
        for value in frame["genres"]
    ]
    frame = frame.sort_values("movieId").reset_index(drop=True)
    report = CleaningReport(rows_read, skipped, invalid, duplicates, len(frame))
    log(f"  movies.csv: {report.rows_kept:,} movies kept ({invalid + skipped:,} invalid, {duplicates:,} duplicate)")
    return frame, report


def load_links(path: Path, movie_ids: np.ndarray, log: Log) -> tuple[pd.DataFrame, CleaningReport]:
    frame, skipped = _read_csv(
        path, {"movieId": pa.int64(), "imdbId": pa.string(), "tmdbId": pa.string()}
    )
    rows_read = len(frame) + skipped
    valid = frame["movieId"].notna() & frame["movieId"].isin(movie_ids)
    invalid = int((~valid).sum())
    frame = frame.loc[valid].copy()
    before = len(frame)
    frame = frame.drop_duplicates("movieId", keep="first")
    duplicates = before - len(frame)
    imdb = pd.to_numeric(frame["imdbId"], errors="coerce")
    frame["imdbId"] = [f"tt{int(v):07d}" if pd.notna(v) else None for v in imdb]
    frame["tmdbId"] = pd.to_numeric(frame["tmdbId"], errors="coerce").astype("Int64")
    frame["movieId"] = frame["movieId"].astype(np.int64)
    report = CleaningReport(rows_read, skipped, invalid, duplicates, len(frame))
    log(f"  links.csv: {report.rows_kept:,} links kept")
    return frame[["movieId", "imdbId", "tmdbId"]], report


def load_ratings(
    path: Path, movie_ids: np.ndarray, log: Log
) -> tuple[pd.DataFrame, CleaningReport]:
    frame, skipped = _read_csv(
        path,
        {
            "userId": pa.int64(),
            "movieId": pa.int64(),
            "rating": pa.float64(),
            "timestamp": pa.int64(),
        },
    )
    rows_read = len(frame) + skipped
    log(f"  ratings.csv: read {rows_read:,} rows")

    user = frame["userId"].to_numpy(dtype=np.float64, na_value=np.nan)
    movie = frame["movieId"].to_numpy(dtype=np.float64, na_value=np.nan)
    rating = frame["rating"].to_numpy(dtype=np.float64, na_value=np.nan)
    stamp = frame["timestamp"].to_numpy(dtype=np.float64, na_value=np.nan)
    del frame

    valid = (
        np.isfinite(user)
        & np.isfinite(movie)
        & np.isfinite(rating)
        & np.isfinite(stamp)
        & (user > 0)
        & (movie > 0)
        & (rating >= 0.5)
        & (rating <= 5.0)
        & np.isclose(rating * 2, np.round(rating * 2))
        & (stamp >= 0)
    )
    valid &= np.isin(movie, movie_ids)
    invalid = int((~valid).sum())

    ratings = pd.DataFrame(
        {
            "userId": user[valid].astype(np.int64),
            "movieId": movie[valid].astype(np.int64),
            "rating": rating[valid].astype(np.float32),
            "timestamp": stamp[valid].astype(np.int64),
        }
    )
    del user, movie, rating, stamp, valid

    # Keep the most recent rating when a user rated the same movie twice.
    before = len(ratings)
    order = np.lexsort(
        (ratings["timestamp"].to_numpy(), ratings["movieId"].to_numpy(), ratings["userId"].to_numpy())
    )
    ratings = ratings.iloc[order].reset_index(drop=True)
    ratings = ratings.drop_duplicates(["userId", "movieId"], keep="last").reset_index(drop=True)
    duplicates = before - len(ratings)

    report = CleaningReport(rows_read, skipped, invalid, duplicates, len(ratings))
    log(
        f"  ratings.csv: {report.rows_kept:,} ratings kept "
        f"({invalid + skipped:,} invalid, {duplicates:,} duplicate)"
    )
    return ratings, report


def normalize_tag(tag: str) -> str:
    return " ".join(tag.strip().lower().split())


def load_tags(path: Path, movie_ids: np.ndarray, log: Log) -> tuple[pd.DataFrame, CleaningReport]:
    frame, skipped = _read_csv(
        path,
        {
            "userId": pa.int64(),
            "movieId": pa.int64(),
            "tag": pa.string(),
            "timestamp": pa.int64(),
        },
    )
    rows_read = len(frame) + skipped
    valid = (
        frame["userId"].notna()
        & frame["movieId"].notna()
        & frame["tag"].notna()
        & frame["movieId"].isin(movie_ids)
    )
    frame = frame.loc[valid, ["userId", "movieId", "tag"]].copy()
    frame["tag"] = frame["tag"].astype(str).map(normalize_tag)
    nonempty = frame["tag"].str.len() > 0
    invalid = int((~valid).sum()) + int((~nonempty).sum())
    frame = frame.loc[nonempty]
    before = len(frame)
    frame = frame.drop_duplicates(["userId", "movieId", "tag"])
    duplicates = before - len(frame)
    frame["userId"] = frame["userId"].astype(np.int64)
    frame["movieId"] = frame["movieId"].astype(np.int64)
    report = CleaningReport(rows_read, skipped, invalid, duplicates, len(frame))
    log(f"  tags.csv: {report.rows_kept:,} tag applications kept")
    return frame.reset_index(drop=True), report


# --------------------------------------------------------------------------- #
# Core transformations
# --------------------------------------------------------------------------- #

def chronological_split(
    user_idx: np.ndarray, train_fraction: float, validation_fraction: float
) -> np.ndarray:
    """Assign each interaction to train / validation / test.

    ``user_idx`` must already be grouped by user and sorted by time within
    each user. For a user with ``n`` interactions, the earliest
    ``floor(n * train_fraction)`` (at least one) go to training, the next
    interactions up to ``floor(n * (train_fraction + validation_fraction))``
    go to validation, and the rest go to test.
    """
    if len(user_idx) == 0:
        return np.zeros(0, dtype=np.int8)
    n_users = int(user_idx.max()) + 1
    counts = np.bincount(user_idx, minlength=n_users)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    position = np.arange(len(user_idx)) - starts[user_idx]
    n = counts[user_idx]
    train_end = np.maximum(1, np.floor(n * train_fraction + 1e-9).astype(np.int64))
    val_end = np.maximum(
        train_end, np.floor(n * (train_fraction + validation_fraction) + 1e-9).astype(np.int64)
    )
    split = np.full(len(user_idx), SPLIT_TEST, dtype=np.int8)
    split[position < val_end] = SPLIT_VALIDATION
    split[position < train_end] = SPLIT_TRAIN
    return split


def build_matrix(
    user_idx: np.ndarray, movie_idx: np.ndarray, values: np.ndarray, n_users: int, n_movies: int
) -> sp.csr_matrix:
    matrix = sp.csr_matrix(
        (values.astype(np.float32), (user_idx, movie_idx)), shape=(n_users, n_movies)
    )
    matrix.sort_indices()
    return matrix


def genre_matrix(genres: list[str], vocabulary: list[str]) -> sp.csr_matrix:
    lookup = {g: i for i, g in enumerate(vocabulary)}
    rows, cols = [], []
    for movie_idx, value in enumerate(genres):
        for genre in value.split("|") if value else []:
            if genre in lookup:
                rows.append(movie_idx)
                cols.append(lookup[genre])
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(genres), len(vocabulary)))


def aggregate_tags(tags: pd.DataFrame, movie_index: pd.Series, per_movie: int) -> pd.DataFrame:
    """Count distinct users per (movie, tag) and keep the most common tags."""
    if tags.empty:
        return pd.DataFrame({"movie_idx": [], "tag": [], "count": []})
    counts = (
        tags.groupby(["movieId", "tag"], sort=False).size().rename("count").reset_index()
    )
    counts["movie_idx"] = counts["movieId"].map(movie_index).astype(np.int32)
    counts = counts.sort_values(["movie_idx", "count", "tag"], ascending=[True, False, True])
    counts = counts.groupby("movie_idx", sort=False).head(per_movie)
    return counts[["movie_idx", "tag", "count"]].reset_index(drop=True).astype(
        {"count": np.int32}
    )


def user_summaries(
    user_ids: np.ndarray,
    user_idx: np.ndarray,
    movie_idx: np.ndarray,
    rating: np.ndarray,
    timestamp: np.ndarray,
    split: np.ndarray,
    full: sp.csr_matrix,
    genres_by_movie: sp.csr_matrix,
    genre_vocabulary: list[str],
    high_threshold: float,
    top_genres: int = 3,
    top_movies: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_users = len(user_ids)
    counts = np.bincount(user_idx, minlength=n_users)
    train_counts = np.bincount(user_idx[split == SPLIT_TRAIN], minlength=n_users)
    sums = np.bincount(user_idx, weights=rating, minlength=n_users)
    means = np.divide(sums, counts, out=np.zeros(n_users), where=counts > 0)
    # Interactions are grouped by user and sorted by time, so each user's
    # first and last timestamps sit at the group boundaries.
    ends = np.cumsum(counts)
    starts = ends - counts
    first_ts = timestamp[np.minimum(starts, len(timestamp) - 1)].astype(np.int64)
    last_ts = timestamp[np.maximum(ends - 1, 0)].astype(np.int64)

    # Favourite genres: genres of movies the user rated highly; users with no
    # high ratings fall back to genres of everything they rated.
    liked = full.copy()
    liked.data = (liked.data >= high_threshold).astype(np.float32)
    liked.eliminate_zeros()
    liked_genres = np.asarray((liked @ genres_by_movie).todense())
    rated = full.copy()
    rated.data = np.ones_like(rated.data)
    all_genres = np.asarray((rated @ genres_by_movie).todense())
    has_liked = liked_genres.sum(axis=1) > 0
    genre_counts = np.where(has_liked[:, None], liked_genres, all_genres)
    order = np.argsort(-genre_counts, axis=1, kind="stable")[:, :top_genres]
    favourite = []
    for u in range(n_users):
        picks = [genre_vocabulary[g] for g in order[u] if genre_counts[u, g] > 0]
        favourite.append("|".join(picks))

    users = pd.DataFrame(
        {
            "user_idx": np.arange(n_users, dtype=np.int32),
            "userId": user_ids.astype(np.int64),
            "rating_count": counts.astype(np.int32),
            "train_count": train_counts.astype(np.int32),
            "mean_rating": means.astype(np.float32),
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "favorite_genres": favourite,
        }
    )

    # Highly rated movies: rating >= threshold, highest rating first, then
    # most recent first.
    high = rating >= high_threshold
    hu, hm, hr, ht = user_idx[high], movie_idx[high], rating[high], timestamp[high]
    order = np.lexsort((-ht, -hr, hu))
    hu, hm, hr = hu[order], hm[order], hr[order]
    if len(hu):
        starts = np.concatenate(([0], np.flatnonzero(np.diff(hu)) + 1))
        group_start = np.repeat(starts, np.diff(np.concatenate((starts, [len(hu)]))))
        rank = np.arange(len(hu)) - group_start
    else:
        rank = np.zeros(0, dtype=np.int64)
    keep = rank < top_movies
    top = pd.DataFrame(
        {
            "user_idx": hu[keep].astype(np.int32),
            "rank": (rank[keep] + 1).astype(np.int16),
            "movie_idx": hm[keep].astype(np.int32),
            "rating": hr[keep].astype(np.float32),
        }
    )
    return users, top


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


def _file_info(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "name": path.name,
        "bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def preprocess(settings: Settings, log: Log = print) -> dict[str, object]:
    started = time.perf_counter()
    data_path = settings.movielens_data_path

    # 1–2. Files exist and have the expected columns.
    check = check_raw_dataset(data_path)
    if not check.ok:
        raise DatasetError(check.problem() or "MovieLens dataset is not usable.")
    assert data_path is not None
    log(f"Reading {DATASET_NAME} from {data_path}")

    # 3–5. Load, validate, deduplicate.
    movies, movies_report = load_movies(data_path / "movies.csv", log)
    movie_ids = movies["movieId"].to_numpy()
    links, links_report = load_links(data_path / "links.csv", movie_ids, log)
    ratings, ratings_report = load_ratings(data_path / "ratings.csv", movie_ids, log)
    tags, tags_report = load_tags(data_path / "tags.csv", movie_ids, log)
    if ratings.empty:
        raise DatasetError("ratings.csv contained no valid ratings.")

    # 6. Internal indices. Every movie in movies.csv gets an index so that
    #    tags and metadata are available even for unrated movies.
    movie_index = pd.Series(np.arange(len(movies), dtype=np.int32), index=movies["movieId"])
    user_ids = np.unique(ratings["userId"].to_numpy())
    user_idx = np.searchsorted(user_ids, ratings["userId"].to_numpy()).astype(np.int32)
    movie_idx = np.searchsorted(movie_ids, ratings["movieId"].to_numpy()).astype(np.int32)
    rating = ratings["rating"].to_numpy(dtype=np.float32)
    timestamp = ratings["timestamp"].to_numpy(dtype=np.int64)
    del ratings
    n_users, n_movies = len(user_ids), len(movie_ids)

    # 7. Sort by user, then time (movie index breaks exact-timestamp ties).
    order = np.lexsort((movie_idx, timestamp, user_idx))
    user_idx, movie_idx, rating, timestamp = (
        user_idx[order], movie_idx[order], rating[order], timestamp[order]
    )
    del order

    # 8. Chronological per-user split.
    split = chronological_split(
        user_idx, settings.train_fraction, settings.validation_fraction
    )
    split_counts = {
        "train": int((split == SPLIT_TRAIN).sum()),
        "validation": int((split == SPLIT_VALIDATION).sum()),
        "test": int((split == SPLIT_TEST).sum()),
    }
    log(
        "  split: "
        + ", ".join(f"{name} {count:,}" for name, count in split_counts.items())
    )

    # 11. Tag aggregation.
    movie_tags = aggregate_tags(tags, movie_index, settings.tags_per_movie)

    # Movie table with display fields and whole-dataset rating stats.
    counts_all = np.bincount(movie_idx, minlength=n_movies)
    sums_all = np.bincount(movie_idx, weights=rating, minlength=n_movies)
    movies_out = movies.merge(links, on="movieId", how="left")
    movies_out = movies_out.sort_values("movieId").reset_index(drop=True)
    movies_out.insert(0, "movie_idx", np.arange(n_movies, dtype=np.int32))
    movies_out["rating_count"] = counts_all.astype(np.int32)
    movies_out["mean_rating"] = np.divide(
        sums_all, counts_all, out=np.zeros(n_movies), where=counts_all > 0
    ).astype(np.float32)
    movies_out = movies_out.rename(columns={"title": "raw_title", "display_title": "title"})
    movies_out = movies_out[
        [
            "movie_idx", "movieId", "title", "raw_title", "year", "genres",
            "imdbId", "tmdbId", "rating_count", "mean_rating",
        ]
    ]

    genre_vocabulary = sorted(
        {g for value in movies_out["genres"] for g in (value.split("|") if value else [])}
    )

    # Sparse user x movie matrices (explicit ratings as values).
    train_mask = split == SPLIT_TRAIN
    train_matrix = build_matrix(
        user_idx[train_mask], movie_idx[train_mask], rating[train_mask], n_users, n_movies
    )
    full_matrix = build_matrix(user_idx, movie_idx, rating, n_users, n_movies)

    # 12. User summaries.
    users, top_movies = user_summaries(
        user_ids, user_idx, movie_idx, rating, timestamp, split, full_matrix,
        genre_matrix(movies_out["genres"].tolist(), genre_vocabulary),
        genre_vocabulary, settings.high_rating_threshold,
    )

    # 9. Persist.
    paths = processed_paths(settings)
    paths.root.mkdir(parents=True, exist_ok=True)
    log(f"Writing processed artifacts to {paths.root}")
    movies_out.to_parquet(paths.movies, index=False)
    movie_tags.to_parquet(paths.movie_tags, index=False)
    pd.DataFrame(
        {
            "user_idx": user_idx,
            "movie_idx": movie_idx,
            "rating": rating,
            "timestamp": timestamp,
            "split": split,
        }
    ).to_parquet(paths.ratings, index=False)
    users.to_parquet(paths.users, index=False)
    top_movies.to_parquet(paths.user_top_movies, index=False)
    sp.save_npz(paths.train_matrix, train_matrix, compressed=False)
    sp.save_npz(paths.full_matrix, full_matrix, compressed=False)

    # Dataset statistics computed from the loaded data.
    most_rated = np.argsort(-counts_all, kind="stable")[:24]
    genre_counts = {
        g: int(sum(1 for value in movies_out["genres"] if g in value.split("|")))
        for g in genre_vocabulary
    }
    statistics = {
        "dataset": DATASET_NAME,
        "ratings": int(len(rating)),
        "users": int(n_users),
        "movies": int(n_movies),
        "ratedMovies": int((counts_all > 0).sum()),
        "tagApplications": int(len(tags)),
        "uniqueTags": int(tags["tag"].nunique()) if len(tags) else 0,
        "links": int(len(links)),
        "meanRating": float(rating.mean()),
        "ratingDistribution": {
            f"{value:.1f}": int(count)
            for value, count in zip(*np.unique(rating, return_counts=True))
        },
        "firstRatingAt": _iso(int(timestamp.min())),
        "lastRatingAt": _iso(int(timestamp.max())),
        "medianRatingsPerUser": float(np.median(users["rating_count"])),
        "split": split_counts,
        "genres": genre_counts,
        "mostRatedMovieIdx": [int(i) for i in most_rated if counts_all[i] > 0],
    }
    paths.statistics.write_text(json.dumps(statistics, indent=2))

    # 10. Dataset version and preprocessing configuration.
    observed = {
        "ratings": ratings_report.rows_read,
        "tags": tags_report.rows_read,
        "movies": movies_report.rows_read,
        "users": int(n_users),
    }
    manifest = {
        "artifactFormatVersion": ARTIFACT_FORMAT_VERSION,
        "dataset": {
            "name": DATASET_NAME,
            "id": DATASET_ID,
            "sourceFiles": [_file_info(data_path / name) for name in REQUIRED_COLUMNS],
            "expectedCounts": EXPECTED_COUNTS,
            "observedCounts": observed,
            "matchesPublishedMl32mCounts": observed == EXPECTED_COUNTS,
        },
        "cleaning": {
            "movies": movies_report.as_dict(),
            "links": links_report.as_dict(),
            "ratings": ratings_report.as_dict(),
            "tags": tags_report.as_dict(),
        },
        "config": {
            "split": "chronological per user",
            "trainFraction": settings.train_fraction,
            "validationFraction": settings.validation_fraction,
            "testFraction": round(1 - settings.train_fraction - settings.validation_fraction, 6),
            "tieBreak": "timestamp, then movieId",
            "duplicateRatings": "keep most recent",
            "highRatingThreshold": settings.high_rating_threshold,
            "tagsPerMovie": settings.tags_per_movie,
            "genreVocabulary": genre_vocabulary,
        },
        "shape": {"users": int(n_users), "movies": int(n_movies)},
        "createdAt": datetime.now(tz=timezone.utc).isoformat(),
        "durationSeconds": round(time.perf_counter() - started, 1),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyarrow": pa.__version__,
        },
    }
    if not manifest["dataset"]["matchesPublishedMl32mCounts"]:
        log(
            "  warning: row counts differ from the published ml-32m figures "
            f"({observed} vs {EXPECTED_COUNTS}). Continuing with the files provided."
        )
    paths.manifest.write_text(json.dumps(manifest, indent=2))
    log(f"Preprocessing finished in {manifest['durationSeconds']}s")
    return manifest
