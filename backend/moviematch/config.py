"""Runtime configuration.

Every setting comes from environment variables (optionally loaded from a
``.env`` file at the repository root). Nothing here hard-codes a filesystem
path, provider, model name, base URL, or credential.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# backend/moviematch/config.py -> repository root is two levels above backend/
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None else value.strip()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return Path(value.strip()).expanduser()


@dataclass(frozen=True)
class LlmSettings:
    enabled: bool
    base_url: str
    api_key: str = field(repr=False)  # never printed
    model: str
    timeout_seconds: float
    json_mode: bool
    temperature: float

    @property
    def configured(self) -> bool:
        """True only when the LLM is enabled and has the minimum it needs.

        An API key is optional because some OpenAI-compatible servers
        (for example a local Ollama instance) do not require one.
        """
        return self.enabled and bool(self.base_url) and bool(self.model)

    @property
    def missing(self) -> list[str]:
        missing = []
        if not self.enabled:
            missing.append("LLM_ENABLED")
        if not self.base_url:
            missing.append("LLM_BASE_URL")
        if not self.model:
            missing.append("LLM_MODEL")
        return missing


@dataclass(frozen=True)
class Settings:
    movielens_data_path: Path | None
    artifacts_dir: Path
    llm: LlmSettings
    poster_url_template: str
    cors_origins: list[str]

    # Data preparation
    train_fraction: float
    validation_fraction: float
    high_rating_threshold: float
    tags_per_movie: int

    # Popularity
    popularity_min_votes_quantile: float

    # Item-KNN
    knn_neighbors: int
    knn_min_item_ratings: int

    # ALS
    als_factors: int
    als_regularization: float
    als_alpha: float
    als_iterations: int
    als_random_seed: int

    # Evaluation
    eval_sample_users: int
    eval_random_seed: int

    @property
    def processed_dir(self) -> Path:
        return self.artifacts_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def evaluation_dir(self) -> Path:
        return self.artifacts_dir / "evaluation"


def load_settings() -> Settings:
    load_dotenv(REPO_ROOT / ".env", override=False)

    artifacts = _env_path("MOVIEMATCH_ARTIFACTS_DIR") or (REPO_ROOT / "artifacts")
    cors = _env_str("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")

    return Settings(
        movielens_data_path=_env_path("MOVIELENS_DATA_PATH"),
        artifacts_dir=artifacts,
        llm=LlmSettings(
            enabled=_env_bool("LLM_ENABLED", False),
            base_url=_env_str("LLM_BASE_URL"),
            api_key=_env_str("LLM_API_KEY"),
            model=_env_str("LLM_MODEL"),
            timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 60.0),
            json_mode=_env_bool("LLM_JSON_MODE", True),
            temperature=_env_float("LLM_TEMPERATURE", 0.2),
        ),
        poster_url_template=_env_str("POSTER_URL_TEMPLATE"),
        cors_origins=[o.strip() for o in cors.split(",") if o.strip()],
        train_fraction=_env_float("SPLIT_TRAIN_FRACTION", 0.8),
        validation_fraction=_env_float("SPLIT_VALIDATION_FRACTION", 0.1),
        high_rating_threshold=_env_float("HIGH_RATING_THRESHOLD", 4.0),
        tags_per_movie=_env_int("TAGS_PER_MOVIE", 10),
        popularity_min_votes_quantile=_env_float("POPULARITY_MIN_VOTES_QUANTILE", 0.90),
        knn_neighbors=_env_int("KNN_NEIGHBORS", 50),
        knn_min_item_ratings=_env_int("KNN_MIN_ITEM_RATINGS", 20),
        als_factors=_env_int("ALS_FACTORS", 64),
        als_regularization=_env_float("ALS_REGULARIZATION", 0.05),
        als_alpha=_env_float("ALS_ALPHA", 2.0),
        als_iterations=_env_int("ALS_ITERATIONS", 15),
        als_random_seed=_env_int("ALS_RANDOM_SEED", 42),
        eval_sample_users=_env_int("EVAL_SAMPLE_USERS", 10000),
        eval_random_seed=_env_int("EVAL_RANDOM_SEED", 42),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
