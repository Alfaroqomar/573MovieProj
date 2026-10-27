"""Names of every artifact the pipeline writes, in one place."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moviematch.config import Settings

ARTIFACT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ProcessedPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def statistics(self) -> Path:
        return self.root / "statistics.json"

    @property
    def movies(self) -> Path:
        return self.root / "movies.parquet"

    @property
    def movie_tags(self) -> Path:
        return self.root / "movie_tags.parquet"

    @property
    def ratings(self) -> Path:
        return self.root / "ratings.parquet"

    @property
    def users(self) -> Path:
        return self.root / "users.parquet"

    @property
    def user_top_movies(self) -> Path:
        return self.root / "user_top_movies.parquet"

    @property
    def train_matrix(self) -> Path:
        return self.root / "train_matrix.npz"

    @property
    def full_matrix(self) -> Path:
        return self.root / "full_matrix.npz"

    def all_required(self) -> list[Path]:
        return [
            self.manifest,
            self.statistics,
            self.movies,
            self.movie_tags,
            self.ratings,
            self.users,
            self.user_top_movies,
            self.train_matrix,
            self.full_matrix,
        ]


@dataclass(frozen=True)
class ModelPaths:
    root: Path

    @property
    def popularity(self) -> Path:
        return self.root / "popularity.npz"

    @property
    def knn(self) -> Path:
        return self.root / "item_knn_similarity.npz"

    @property
    def knn_meta(self) -> Path:
        return self.root / "item_knn.json"

    @property
    def als(self) -> Path:
        return self.root / "als_factors.npz"

    @property
    def als_meta(self) -> Path:
        return self.root / "als.json"

    @property
    def training_manifest(self) -> Path:
        return self.root / "training_manifest.json"


def processed_paths(settings: Settings) -> ProcessedPaths:
    return ProcessedPaths(settings.processed_dir)


def model_paths(settings: Settings) -> ModelPaths:
    return ModelPaths(settings.models_dir)


def evaluation_results_path(settings: Settings) -> Path:
    return settings.evaluation_dir / "results.json"
