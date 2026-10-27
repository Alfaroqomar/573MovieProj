"""Locating and validating the raw MovieLens 32M files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.csv as pacsv

DATASET_NAME = "MovieLens 32M"
DATASET_ID = "ml-32m"
DATASET_PAGE = "https://grouplens.org/datasets/movielens/32m/"
DATASET_README = "https://files.grouplens.org/datasets/movielens/ml-32m-README.html"
DATASET_ZIP = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"

# Published figures for ml-32m (from the GroupLens README). Used only to warn
# when the files on disk look like a different MovieLens release.
EXPECTED_COUNTS = {
    "ratings": 32_000_204,
    "tags": 2_000_072,
    "movies": 87_585,
    "users": 200_948,
}

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "ratings.csv": ["userId", "movieId", "rating", "timestamp"],
    "movies.csv": ["movieId", "title", "genres"],
    "tags.csv": ["userId", "movieId", "tag", "timestamp"],
    "links.csv": ["movieId", "imdbId", "tmdbId"],
}


@dataclass
class RawDatasetCheck:
    configured: bool
    path: Path | None
    path_exists: bool = False
    missing_files: list[str] = field(default_factory=list)
    column_errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (
            self.configured
            and self.path_exists
            and not self.missing_files
            and not self.column_errors
        )

    def problem(self) -> str | None:
        if not self.configured:
            return (
                "MOVIELENS_DATA_PATH is not set. Download MovieLens 32M "
                f"({DATASET_ZIP}), unzip it, and set MOVIELENS_DATA_PATH to the ml-32m folder."
            )
        if not self.path_exists:
            return (
                "MovieLens 32M was not found at MOVIELENS_DATA_PATH. "
                "Check the path and that the ZIP has been extracted."
            )
        if self.missing_files:
            return "MovieLens 32M is incomplete. Missing: " + ", ".join(self.missing_files)
        if self.column_errors:
            details = "; ".join(f"{k}: {v}" for k, v in self.column_errors.items())
            return f"MovieLens files have unexpected columns ({details})."
        return None


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        first = handle.readline()
    # Header lines in MovieLens are plain comma-separated names.
    return [c.strip().strip('"') for c in first.rstrip("\r\n").split(",")]


def check_raw_dataset(path: Path | None) -> RawDatasetCheck:
    check = RawDatasetCheck(configured=path is not None, path=path)
    if path is None:
        return check
    check.path_exists = path.is_dir()
    if not check.path_exists:
        return check
    for name, expected in REQUIRED_COLUMNS.items():
        file_path = path / name
        if not file_path.is_file():
            check.missing_files.append(name)
            continue
        try:
            header = read_header(file_path)
        except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover - defensive
            check.column_errors[name] = f"could not read header ({type(exc).__name__})"
            continue
        if header != expected:
            check.column_errors[name] = f"expected {expected}, found {header}"
    return check


def csv_read_options() -> pacsv.ReadOptions:
    return pacsv.ReadOptions(block_size=64 << 20)
