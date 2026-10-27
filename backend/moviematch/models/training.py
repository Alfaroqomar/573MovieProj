"""Train and save all models from the processed TRAINING split.

Run with:  python -m moviematch.cli train [--only popularity|knn|als]

Only ``train_matrix.npz`` (the chronological training split) is read here.
Validation and test interactions are never passed to any model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np
import scipy.sparse as sp

from moviematch.config import Settings
from moviematch.data.paths import model_paths, processed_paths
from moviematch.models.als import fit_als, save_als
from moviematch.models.item_knn import fit_item_knn, save_item_knn
from moviematch.models.popularity import fit_popularity, save_popularity

Log = Callable[[str], None]
MODEL_NAMES = ("popularity", "knn", "als")


class NotPreprocessedError(RuntimeError):
    pass


def _load_train(settings: Settings) -> tuple[sp.csr_matrix, dict]:
    paths = processed_paths(settings)
    missing = [p.name for p in paths.all_required() if not p.exists()]
    if missing:
        raise NotPreprocessedError(
            "Processed dataset artifacts are missing ("
            + ", ".join(missing)
            + "). Run the preprocessing command first."
        )
    manifest = json.loads(paths.manifest.read_text())
    train = sp.load_npz(paths.train_matrix).tocsr().astype(np.float32)
    return train, manifest


def train_models(
    settings: Settings, only: list[str] | None = None, log: Log = print
) -> dict[str, object]:
    selected = list(only) if only else list(MODEL_NAMES)
    unknown = sorted(set(selected) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")

    train, manifest = _load_train(settings)
    paths = model_paths(settings)
    paths.root.mkdir(parents=True, exist_ok=True)
    log(
        f"Training on {train.nnz:,} training interactions "
        f"({train.shape[0]:,} users x {train.shape[1]:,} movies)"
    )
    summary: dict[str, object] = {}
    source = {
        "dataset": manifest["dataset"]["id"],
        "processedAt": manifest["createdAt"],
        "trainingInteractions": int(train.nnz),
        "split": "train",
    }

    if "popularity" in selected:
        started = time.perf_counter()
        arrays = fit_popularity(train, settings.popularity_min_votes_quantile)
        save_popularity(arrays, paths.popularity)
        seconds = round(time.perf_counter() - started, 1)
        summary["popularity"] = {
            "eligibleMovies": int(arrays["eligible_count"]),
            "minVotes": float(arrays["min_votes"]),
            "seconds": seconds,
        }
        log(f"  popularity: {int(arrays['eligible_count']):,} eligible movies, m={float(arrays['min_votes']):.0f} ({seconds}s)")

    if "knn" in selected:
        started = time.perf_counter()
        similarity = fit_item_knn(
            train, settings.knn_neighbors, settings.knn_min_item_ratings
        )
        seconds = round(time.perf_counter() - started, 1)
        meta = {
            **source,
            "similarity": "cosine on training rating vectors",
            "neighbors": settings.knn_neighbors,
            "minItemRatings": settings.knn_min_item_ratings,
            "symmetrized": "max(S, S^T)",
            "storedPairs": int(similarity.nnz),
            "highRatingThreshold": settings.high_rating_threshold,
            "trainedAt": datetime.now(tz=timezone.utc).isoformat(),
            "seconds": seconds,
        }
        save_item_knn(similarity, paths.knn, paths.knn_meta, meta)
        summary["knn"] = {"storedPairs": int(similarity.nnz), "seconds": seconds}
        log(f"  item-knn: {similarity.nnz:,} stored similarities ({seconds}s)")

    if "als" in selected:
        started = time.perf_counter()
        user_factors, item_factors = fit_als(
            train,
            settings.als_factors,
            settings.als_regularization,
            settings.als_alpha,
            settings.als_iterations,
            settings.als_random_seed,
        )
        seconds = round(time.perf_counter() - started, 1)
        meta = {
            **source,
            "library": "implicit (CPU, conjugate gradient)",
            "feedback": "implicit; confidence c = 1 + alpha * rating",
            "factors": settings.als_factors,
            "regularization": settings.als_regularization,
            "alpha": settings.als_alpha,
            "iterations": settings.als_iterations,
            "randomSeed": settings.als_random_seed,
            "trainedAt": datetime.now(tz=timezone.utc).isoformat(),
            "seconds": seconds,
        }
        save_als(user_factors, item_factors, paths.als, paths.als_meta, meta)
        summary["als"] = {"factors": settings.als_factors, "seconds": seconds}
        log(f"  als: {settings.als_factors} factors, {settings.als_iterations} iterations ({seconds}s)")

    previous = {}
    if paths.training_manifest.exists():
        previous = json.loads(paths.training_manifest.read_text()).get("models", {})
    previous.update(summary)
    paths.training_manifest.write_text(
        json.dumps(
            {
                "source": source,
                "models": previous,
                "updatedAt": datetime.now(tz=timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    return summary
