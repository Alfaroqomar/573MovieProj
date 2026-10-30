"""Train and save the models.

Run with:
    python -m moviematch.cli train                       # popularity, knn, als, als_explicit
    python -m moviematch.cli train --only neumf          # NeuMF (needs PyTorch; slow)
    python -m moviematch.cli train --with-validation     # fit on train + validation (final test runs)

By default only the TRAINING split is used, so validation and test
interactions never reach any model. ``--with-validation`` adds the
validation period for the final test-set evaluation, as is standard once
settings have been tuned on validation; test interactions are never used.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.sparse as sp

from moviematch.config import Settings
from moviematch.data.paths import model_paths, processed_paths
from moviematch.data.preprocess import SPLIT_TRAIN, SPLIT_VALIDATION, build_matrix
from moviematch.models.als import fit_als, save_als
from moviematch.models.als_explicit import fit_als_explicit, save_als_explicit
from moviematch.models.item_knn import fit_item_knn, save_item_knn
from moviematch.models.popularity import fit_popularity, save_popularity

Log = Callable[[str], None]
MODEL_NAMES = ("popularity", "knn", "als", "als_explicit", "neumf")
DEFAULT_MODELS = ("popularity", "knn", "als", "als_explicit")


class NotPreprocessedError(RuntimeError):
    pass


class MissingDependencyError(RuntimeError):
    pass


def _check_processed(settings: Settings) -> dict:
    paths = processed_paths(settings)
    missing = [p.name for p in paths.all_required() if not p.exists()]
    if missing:
        raise NotPreprocessedError(
            "Processed dataset artifacts are missing ("
            + ", ".join(missing)
            + "). Run the preprocessing command first."
        )
    return json.loads(paths.manifest.read_text())


def load_fit_matrix(settings: Settings, with_validation: bool) -> sp.csr_matrix:
    """User x movie ratings the models are fitted on."""
    paths = processed_paths(settings)
    if not with_validation:
        return sp.load_npz(paths.train_matrix).tocsr().astype(np.float32)
    shape = sp.load_npz(paths.train_matrix).shape
    ratings = pd.read_parquet(paths.ratings, columns=["user_idx", "movie_idx", "rating", "split"])
    mask = ratings["split"].isin([SPLIT_TRAIN, SPLIT_VALIDATION]).to_numpy()
    return build_matrix(
        ratings["user_idx"].to_numpy()[mask],
        ratings["movie_idx"].to_numpy()[mask],
        ratings["rating"].to_numpy()[mask],
        shape[0],
        shape[1],
    )


def validation_relevant(settings: Settings, max_users: int = 5000) -> dict[int, set[int]]:
    """Validation movies rated >= threshold, for users with training history."""
    paths = processed_paths(settings)
    ratings = pd.read_parquet(paths.ratings, columns=["user_idx", "movie_idx", "rating", "split"])
    train_users = set(ratings.loc[ratings["split"] == SPLIT_TRAIN, "user_idx"].unique().tolist())
    val = ratings[(ratings["split"] == SPLIT_VALIDATION) & (ratings["rating"] >= settings.high_rating_threshold)]
    val = val[val["user_idx"].isin(train_users)]
    grouped = val.groupby("user_idx")["movie_idx"].apply(lambda s: set(s.tolist()))
    if len(grouped) > max_users:
        grouped = grouped.sample(max_users, random_state=settings.eval_random_seed)
    return {int(u): items for u, items in grouped.items()}


def train_models(
    settings: Settings,
    only: list[str] | None = None,
    with_validation: bool = False,
    log: Log = print,
) -> dict[str, object]:
    selected = list(only) if only else list(DEFAULT_MODELS)
    unknown = sorted(set(selected) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    if "neumf" in selected:
        from moviematch.models.neumf import torch_available

        if not torch_available():
            raise MissingDependencyError(
                "NeuMF needs PyTorch. Install it with `pip install torch`, then run `train --only neumf`."
            )

    manifest = _check_processed(settings)
    train = load_fit_matrix(settings, with_validation)
    fit_on = "train+validation" if with_validation else "train"
    paths = model_paths(settings)
    paths.root.mkdir(parents=True, exist_ok=True)
    log(
        f"Training on {train.nnz:,} {fit_on} interactions "
        f"({train.shape[0]:,} users x {train.shape[1]:,} movies)"
    )
    summary: dict[str, object] = {}
    source = {
        "dataset": manifest["dataset"]["id"],
        "processedAt": manifest["createdAt"],
        "splitMode": manifest["config"].get("splitMode", "per_user"),
        "trainingInteractions": int(train.nnz),
        "fitOn": fit_on,
    }

    if "popularity" in selected:
        started = time.perf_counter()
        arrays = fit_popularity(train, settings.popularity_min_votes_quantile)
        save_popularity(arrays, paths.popularity)
        seconds = round(time.perf_counter() - started, 1)
        summary["popularity"] = {
            "rules": ["count (most-rated)", "weighted"],
            "eligibleMovies": int(arrays["eligible_count"]),
            "minVotes": float(arrays["min_votes"]),
            "seconds": seconds,
            "fitOn": fit_on,
        }
        log(f"  popularity: most-rated and weighted lists, m={float(arrays['min_votes']):.0f} ({seconds}s)")

    if "knn" in selected:
        started = time.perf_counter()
        similarity = fit_item_knn(train, settings.knn_neighbors, settings.knn_min_item_ratings)
        seconds = round(time.perf_counter() - started, 1)
        meta = {
            **source,
            "similarity": "cosine on rating vectors",
            "neighbors": settings.knn_neighbors,
            "minItemRatings": settings.knn_min_item_ratings,
            "symmetrized": "max(S, S^T)",
            "storedPairs": int(similarity.nnz),
            "scoring": settings.knn_scoring,
            "minSupport": settings.knn_min_support,
            "highRatingThreshold": settings.high_rating_threshold,
            "trainedAt": datetime.now(tz=timezone.utc).isoformat(),
            "seconds": seconds,
        }
        save_item_knn(similarity, paths.knn, paths.knn_meta, meta)
        summary["knn"] = {"storedPairs": int(similarity.nnz), "seconds": seconds, "fitOn": fit_on}
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
        summary["als"] = {"factors": settings.als_factors, "seconds": seconds, "fitOn": fit_on}
        log(f"  als (implicit): {settings.als_factors} factors, {settings.als_iterations} iterations ({seconds}s)")

    if "als_explicit" in selected:
        started = time.perf_counter()
        log("  als (explicit ratings): training")
        user_factors, item_factors, rmse_history = fit_als_explicit(
            train,
            settings.als_explicit_factors,
            settings.als_explicit_regularization,
            settings.als_explicit_iterations,
            settings.als_random_seed,
            log=log,
        )
        seconds = round(time.perf_counter() - started, 1)
        meta = {
            **source,
            "library": "NumPy closed-form ALS (ALS-WR regularisation, as in Spark MLlib)",
            "feedback": "explicit; minimises squared error on observed ratings (proposal Eq. 1)",
            "factors": settings.als_explicit_factors,
            "regularization": settings.als_explicit_regularization,
            "iterations": settings.als_explicit_iterations,
            "randomSeed": settings.als_random_seed,
            "trainRmse": rmse_history,
            "trainedAt": datetime.now(tz=timezone.utc).isoformat(),
            "seconds": seconds,
        }
        save_als_explicit(user_factors, item_factors, paths.als_explicit, paths.als_explicit_meta, meta)
        summary["als_explicit"] = {
            "factors": settings.als_explicit_factors,
            "trainRmse": rmse_history[-1] if rmse_history else None,
            "seconds": seconds,
            "fitOn": fit_on,
        }
        log(f"  als (explicit): done ({seconds}s)")

    if "neumf" in selected:
        from moviematch.models.neumf import fit_neumf, save_neumf

        started = time.perf_counter()
        # With validation in the training data there is nothing held out to stop on.
        relevant = {} if with_validation else validation_relevant(settings)
        state, info = fit_neumf(
            train,
            relevant,
            gmf_dim=settings.neumf_gmf_dim,
            layers=settings.neumf_mlp_layers,
            negatives=settings.neumf_negatives,
            epochs=settings.neumf_epochs,
            batch_size=settings.neumf_batch_size,
            learning_rate=settings.neumf_learning_rate,
            patience=settings.neumf_patience,
            max_train_ratings=settings.neumf_max_train_ratings,
            device_preference=settings.neumf_device,
            seed=settings.als_random_seed,
            log=log,
        )
        seconds = round(time.perf_counter() - started, 1)
        meta = {
            **source,
            "nUsers": int(train.shape[0]),
            "nItems": int(train.shape[1]),
            "gmfDim": settings.neumf_gmf_dim,
            "layers": list(settings.neumf_mlp_layers),
            "negatives": settings.neumf_negatives,
            "batchSize": settings.neumf_batch_size,
            "learningRate": settings.neumf_learning_rate,
            "maxTrainRatings": settings.neumf_max_train_ratings,
            **info,
            "trainedAt": datetime.now(tz=timezone.utc).isoformat(),
            "seconds": seconds,
        }
        np.savez(
            paths.neumf_support,
            users=np.flatnonzero(np.diff(train.indptr) > 0),
            items=np.flatnonzero(np.diff(train.tocsc().indptr) > 0),
        )
        save_neumf(state, paths.neumf, paths.neumf_meta, meta)
        summary["neumf"] = {
            "bestEpoch": info["bestEpoch"],
            "bestValidationNdcg@10": info["bestValidationNdcg@10"],
            "device": info["device"],
            "seconds": seconds,
            "fitOn": fit_on,
        }
        log(f"  neumf: best round {info['bestEpoch']} ({seconds}s)")

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
