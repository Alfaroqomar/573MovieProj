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

    # Proposal-aligned settings (all have defaults so older code paths keep working)
    split_mode: str = "date"                      # "date" (proposal) or "per_user"
    split_validation_start: str = "2020-01-01"    # date mode: first day of validation
    split_test_start: str = "2022-01-01"          # date mode: first day of test
    cold_start_max_ratings: int = 10              # proposal: <= 10 training ratings
    knn_scoring: str = "sum"                      # "sum" (chosen on validation) or "average" (proposal Eq. 2)
    knn_min_support: int = 2
    als_explicit_factors: int = 64
    als_explicit_regularization: float = 0.05
    als_explicit_iterations: int = 10
    stage1_per_model: int = 30                    # proposal: top 30 from each model
    stage1_pool_size: int = 50                    # proposal: up to 50 candidates
    stage1_top_up: bool = True                    # fill to 50 from ALS ranks 31+
    neumf_gmf_dim: int = 32
    neumf_mlp_layers: tuple[int, ...] = (64, 32, 16)
    neumf_negatives: int = 4
    neumf_epochs: int = 10
    neumf_batch_size: int = 8192
    neumf_learning_rate: float = 0.001
    neumf_patience: int = 2
    neumf_max_train_ratings: int = 0              # 0 = use every training rating
    neumf_device: str = "auto"                    # auto | cpu | mps | cuda
    eval_cold_start_extra: int = 5000             # extra cold-start users for that group
    eval_bootstrap: int = 1000

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
        split_mode=_env_str("SPLIT_MODE", "date").lower(),
        split_validation_start=_env_str("SPLIT_VALIDATION_START", "2020-01-01"),
        split_test_start=_env_str("SPLIT_TEST_START", "2022-01-01"),
        cold_start_max_ratings=_env_int("COLD_START_MAX_RATINGS", 10),
        knn_scoring=_env_str("KNN_SCORING", "sum").lower(),
        knn_min_support=_env_int("KNN_MIN_SUPPORT", 2),
        als_explicit_factors=_env_int("ALS_EXPLICIT_FACTORS", 64),
        als_explicit_regularization=_env_float("ALS_EXPLICIT_REGULARIZATION", 0.05),
        als_explicit_iterations=_env_int("ALS_EXPLICIT_ITERATIONS", 10),
        stage1_per_model=_env_int("STAGE1_PER_MODEL", 30),
        stage1_pool_size=_env_int("STAGE1_POOL_SIZE", 50),
        stage1_top_up=_env_bool("STAGE1_TOP_UP", True),
        neumf_gmf_dim=_env_int("NEUMF_GMF_DIM", 32),
        neumf_mlp_layers=tuple(
            int(x) for x in _env_str("NEUMF_MLP_LAYERS", "64,32,16").split(",") if x.strip()
        ),
        neumf_negatives=_env_int("NEUMF_NEGATIVES", 4),
        neumf_epochs=_env_int("NEUMF_EPOCHS", 10),
        neumf_batch_size=_env_int("NEUMF_BATCH_SIZE", 8192),
        neumf_learning_rate=_env_float("NEUMF_LEARNING_RATE", 0.001),
        neumf_patience=_env_int("NEUMF_PATIENCE", 2),
        neumf_max_train_ratings=_env_int("NEUMF_MAX_TRAIN_RATINGS", 0),
        neumf_device=_env_str("NEUMF_DEVICE", "auto").lower(),
        eval_cold_start_extra=_env_int("EVAL_COLD_START_EXTRA", 5000),
        eval_bootstrap=_env_int("EVAL_BOOTSTRAP", 1000),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
