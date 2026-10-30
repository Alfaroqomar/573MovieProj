"""Test fixtures.

The tests need *some* MovieLens-format data to exercise the pipeline, and the
real 239 MB dataset is not available in CI. ``synthetic_movielens`` writes a
small, clearly synthetic dataset in the official ml-32m file format into a
temporary directory. It exists only inside the test suite; the application
itself never generates, bundles, or falls back to sample data.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from moviematch.config import LlmSettings, Settings, load_settings

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Horror", "Animation", "Thriller"]


def write_synthetic_movielens(root: Path, n_users: int = 300, n_movies: int = 400, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    movie_genre = rng.integers(0, len(GENRES), size=n_movies)

    lines = ["movieId,title,genres"]
    for m in range(n_movies):
        movie_id = m + 1
        second = GENRES[(movie_genre[m] + 3) % len(GENRES)]
        title = f"Synthetic Title {movie_id} ({1950 + m % 70})"
        if m % 50 == 0:
            title = f'"Synthetic Picture {movie_id}, The ({1980 + m % 30})"'
        genres = "(no genres listed)" if m == 7 else f"{GENRES[movie_genre[m]]}|{second}"
        lines.append(f"{movie_id},{title},{genres}")
    (root / "movies.csv").write_text("\n".join(lines) + "\n")

    lines = ["movieId,imdbId,tmdbId"]
    for m in range(n_movies):
        tmdb = "" if m % 9 == 0 else str(10000 + m)
        lines.append(f"{m + 1},{100000 + m:07d},{tmdb}")
    (root / "links.csv").write_text("\n".join(lines) + "\n")

    lines = ["userId,movieId,rating,timestamp"]
    t_2015 = 1_420_070_400  # 2015-01-01
    year = 365 * 86_400
    for u in range(n_users):
        favourite = u % len(GENRES)
        # Most users have plenty of ratings; every 15th user is "cold" (few
        # ratings before 2020, the rest after), so the date split produces a
        # cold-start group.
        cold = u % 15 == 0
        count = int(rng.integers(20, 30)) if cold else int(rng.integers(20, 60))
        weights = np.where(movie_genre == favourite, 8.0, 1.0)
        weights /= weights.sum()
        movies = rng.choice(n_movies, size=count, replace=False, p=weights)
        if cold:
            times = np.sort(np.concatenate([
                rng.integers(t_2015 + 4 * year, t_2015 + 5 * year - 86_400, size=5),
                rng.integers(t_2015 + 5 * year + 86_400, t_2015 + 8 * year, size=count - 5),
            ]))
        else:
            start = t_2015 + int(rng.integers(0, 3 * year))
            times = np.sort(rng.integers(start, t_2015 + 8 * year + 200 * 86_400, size=count))
        for m, t in zip(movies, times):
            rating = 4.5 if movie_genre[m] == favourite else float(rng.choice([2.0, 3.0, 3.5]))
            lines.append(f"{u + 1},{m + 1},{rating},{t}")
    # Rows the preprocessor must clean up.
    lines.append("1,1,7.0,1500000001")  # rating out of range
    lines.append("2,99999,4.0,1500000002")  # unknown movie
    lines.append("3,2,,1500000003")  # missing rating
    lines.append(lines[1])  # exact duplicate of the first rating
    (root / "ratings.csv").write_text("\n".join(lines) + "\n")

    t0 = t_2015
    lines = ["userId,movieId,tag,timestamp"]
    for m in range(0, n_movies, 3):
        lines.append(f"1,{m + 1},Mind-Bending ,{t0}")
        lines.append(f"2,{m + 1},mind-bending,{t0}")
        lines.append(f"3,{m + 1},\"slow, quiet\",{t0}")
    lines.append(f"4,1,,{t0}")  # empty tag
    (root / "tags.csv").write_text("\n".join(lines) + "\n")
    return root


def make_settings(tmp: Path, data: Path | None, **llm: object) -> Settings:
    base = load_settings()
    llm_settings = LlmSettings(
        enabled=bool(llm.get("enabled", False)),
        base_url=str(llm.get("base_url", "")),
        api_key=str(llm.get("api_key", "")),
        model=str(llm.get("model", "")),
        timeout_seconds=float(llm.get("timeout_seconds", 5)),
        json_mode=True,
        temperature=0.0,
    )
    return dataclasses.replace(
        base,
        movielens_data_path=data,
        artifacts_dir=tmp / "artifacts",
        llm=llm_settings,
        poster_url_template="",
        knn_min_item_ratings=2,
        knn_neighbors=20,
        als_factors=16,
        als_iterations=8,
        als_explicit_factors=8,
        als_explicit_iterations=4,
        neumf_gmf_dim=8,
        neumf_mlp_layers=(16, 8),
        neumf_epochs=2,
        neumf_batch_size=1024,
        eval_sample_users=200,
        eval_cold_start_extra=50,
        eval_bootstrap=50,
    )


@pytest.fixture(scope="session")
def prepared(tmp_path_factory) -> Settings:
    """Synthetic dataset preprocessed and trained once for the whole session."""
    from moviematch.data.preprocess import preprocess
    from moviematch.models.training import train_models

    tmp = tmp_path_factory.mktemp("prepared")
    data = write_synthetic_movielens(tmp / "ml-32m")
    settings = make_settings(tmp, data)
    preprocess(settings, log=lambda _: None)
    train_models(settings, log=lambda _: None)
    try:
        import torch  # noqa: F401

        train_models(settings, only=["neumf"], log=lambda _: None)
    except ImportError:
        pass
    return settings


class FakeLlm:
    """Records the messages it receives and replays scripted responses."""

    def __init__(self, responses: list[object]):
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(messages)
        return str(response)


def candidate_ids_from(messages: list[dict[str, str]]) -> list[int]:
    payload = json.loads(messages[1]["content"])
    return [c["movieId"] for c in payload["candidates"]]


def valid_response_for(top_n: int, reverse: bool = True):
    def build(messages: list[dict[str, str]]) -> str:
        ids = candidate_ids_from(messages)
        picks = list(reversed(ids))[:top_n] if reverse else ids[:top_n]
        return json.dumps({
            "interpretedRequest": {
                "preferredGenres": ["Sci-Fi", "Science Fiction"],
                "preferredTags": ["thoughtful"],
                "similarTo": [],
                "avoid": ["heavy action"],
            },
            "rankedMovies": [
                {"movieId": mid, "rank": i + 1, "llmRelevanceScore": 0.9 - i * 0.01,
                 "explanation": "Matches your interest in the supplied genres."}
                for i, mid in enumerate(picks)
            ],
        })

    return build
